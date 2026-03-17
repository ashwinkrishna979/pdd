# core/gemini_client.py

"""
Unified Gemini API client for text and vision calls.
Uses `google-genai` SDK.
Enhanced rate limiting for free-tier (5-15 RPM).
Tracks daily request count.
"""

import os
import time
import threading
from typing import Optional, List
from PIL import Image

from google import genai
from google.genai import types
from google.genai.errors import APIError

from core.config import config


class GeminiClient:
    def __init__(self):
        self._client = None
        self._tracker = None

        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._min_request_interval = 60.0 / max(config.gemini.requests_per_minute, 1)
        self._daily_count = 0
        self._day_start = time.time()

        self._last_health_error: str = ""
        self._configure()

    def _configure(self):
        api_key = config.gemini.api_key
        if not api_key:
            self._client = None
            print(
                "    [Gemini] No API key set. Set GEMINI_API_KEY environment variable."
            )
            return
        try:
            self._client = genai.Client(api_key=api_key)
            self._last_health_error = ""
            print(
                f"    [Gemini] Configured (text: {config.gemini.text_model}, "
                f"vision: {config.gemini.vision_model}, "
                f"RPM: {config.gemini.requests_per_minute})"
            )
        except Exception as e:
            self._client = None
            self._last_health_error = f"{type(e).__name__}: {e}"
            print(f"    [Gemini] Configuration failed: {self._last_health_error}")

    def set_tracker(self, tracker):
        self._tracker = tracker

    def is_configured(self) -> bool:
        return self._client is not None

    def last_health_error(self) -> str:
        return self._last_health_error

    def _rate_limit(self):
        """Enforce rate limit with daily counter reset."""
        with self._lock:
            # Reset daily counter if new day
            now = time.time()
            if now - self._day_start > 86400:
                self._daily_count = 0
                self._day_start = now

            # Check daily limit
            if self._daily_count >= config.gemini.requests_per_day:
                print("    [Gemini] Daily request limit reached. Waiting 60s...")
                time.sleep(60)

            # Enforce per-minute spacing
            elapsed = now - self._last_request_time
            if elapsed < self._min_request_interval:
                wait = self._min_request_interval - elapsed
                time.sleep(wait)

            self._last_request_time = time.time()
            self._daily_count += 1

    def _prepare_image(self, image_path: str) -> Optional[Image.Image]:
        if not image_path or not os.path.exists(image_path):
            return None
        try:
            img = Image.open(image_path)
            max_w = config.image.max_width
            max_h = config.image.max_height

            if img.width > max_w or img.height > max_h:
                img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

            if img.mode == "RGBA":
                img = img.convert("RGB")
            elif img.mode not in ("RGB", "L"):
                img = img.convert("RGB")

            return img
        except Exception as e:
            print(f"    [Gemini] Image load error: {e}")
            return None

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        image_paths: Optional[List[str]] = None,
        audio_path: Optional[str] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        call_name: Optional[str] = None,
        max_retries: int = 3,
    ) -> Optional[str]:
        if not self._client:
            print("    [Gemini] Not configured. Set GEMINI_API_KEY.")
            return None

        has_images = bool(image_paths)
        has_audio = bool(audio_path)
        model_name = (
            config.gemini.vision_model
            if (has_images or has_audio)
            else config.gemini.text_model
        )
        temp = temperature if temperature is not None else config.llm.temperature
        max_tokens = max_output_tokens or config.llm.max_output_tokens

        contents = []

        # Upload audio if provided
        uploaded_audio = None
        if has_audio and os.path.exists(audio_path):
            try:
                print(f"    [Gemini] Uploading audio: {audio_path}")
                uploaded_audio = self._client.files.upload(file=audio_path)
                contents.append(uploaded_audio)
            except Exception as e:
                print(f"    [Gemini] Audio upload error: {e}")

        if has_images:
            for p in image_paths:
                img = self._prepare_image(p)
                if img:
                    contents.append(img)
            if image_paths and not any(isinstance(c, Image.Image) for c in contents):
                print("    [Gemini] No images could be loaded")

        contents.append(prompt)

        gen_kwargs = {
            "temperature": temp,
            "top_p": config.llm.top_p,
            "max_output_tokens": max_tokens,
        }
        if system_prompt:
            gen_kwargs["system_instruction"] = system_prompt

        gen_config = types.GenerateContentConfig(**gen_kwargs)

        for attempt in range(max_retries + 1):
            start = time.time()
            try:
                self._rate_limit()

                preview = prompt[:80].replace("\n", " ")
                img_str = f", {len(image_paths)} img" if has_images else ""
                print(
                    f"    [Gemini] Sending ({len(prompt)} chars{img_str}, model={model_name}): "
                    f'"{preview}..." (attempt {attempt + 1}, daily: {self._daily_count})'
                )

                resp = self._client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=gen_config,
                )

                elapsed = time.time() - start
                text = (resp.text or "").strip() or None

                prompt_tokens = 0
                response_tokens = 0
                if hasattr(resp, "usage_metadata") and resp.usage_metadata:
                    prompt_tokens = (
                        getattr(resp.usage_metadata, "prompt_token_count", 0) or 0
                    )
                    response_tokens = (
                        getattr(resp.usage_metadata, "candidates_token_count", 0) or 0
                    )

                if text:
                    token_info = (
                        f" [tokens: {prompt_tokens}>{response_tokens}]"
                        if prompt_tokens
                        else ""
                    )
                    print(
                        f"    [Gemini] {len(text)} chars in {elapsed:.1f}s{token_info}"
                    )
                else:
                    print(f"    [Gemini] Empty response after {elapsed:.1f}s")

                self._record(
                    call_name,
                    model_name,
                    prompt,
                    system_prompt,
                    text,
                    elapsed,
                    prompt_tokens,
                    response_tokens,
                    has_images or has_audio,
                )

                if uploaded_audio:
                    try:
                        self._client.files.delete(name=uploaded_audio.name)
                    except Exception as e:
                        print(f"    [Gemini] Failed to delete audio file: {e}")

                return text

            except APIError as e:
                if e.code == 429:
                    wait = 15 * (attempt + 1)
                    print(f"    [Gemini] Rate limited (429). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                if e.code in (500, 503):
                    wait = 5 * (attempt + 1)
                    print(f"    [Gemini] Server error ({e.code}). Waiting {wait}s...")
                    time.sleep(wait)
                    continue
                print(f"    [Gemini] API Error {e.code}: {e.message}")
                break
            except Exception as e:
                print(f"    [Gemini] Unexpected error: {type(e).__name__}: {e}")
                if attempt < max_retries:
                    time.sleep(5)
                    continue
                break

        self._record(
            call_name,
            model_name,
            prompt,
            system_prompt,
            None,
            time.time() - start,
            0,
            0,
            has_images,
        )
        return None

    def _record(
        self,
        call_name,
        model_name,
        prompt,
        system_prompt,
        response,
        duration,
        actual_prompt,
        actual_response,
        has_image,
    ):
        if self._tracker and call_name:
            self._tracker.record(
                call_name=call_name,
                model=model_name,
                prompt=prompt or "",
                response=response or "",
                duration=duration,
                system_prompt=system_prompt or "",
                actual_prompt_tokens=actual_prompt,
                actual_response_tokens=actual_response,
                has_image=has_image,
            )

    def is_available(self) -> bool:
        if not self._client:
            self._last_health_error = "Client not configured"
            return False

        try:
            it = self._client.models.list()
            for _ in it:
                break
            self._last_health_error = ""
            return True

        except APIError as e:
            self._last_health_error = f"{e.code} - {e.message}"
            if e.code == 429:
                return True
            if e.code in (401, 403):
                return False
            return False

        except Exception as e:
            self._last_health_error = f"{type(e).__name__}: {e}"
            return False


gemini_client = GeminiClient()
