"""Batch frame description using Gemini 2.0 Flash.

Sends all extracted frames (in video order) to Gemini as a single
multi-image request so the model can understand the full video context
and return per-frame descriptions.
"""

import json
import logging
import os

from google import genai
from google.genai import types
from PIL import Image

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a video workflow analyst. You will receive a sequence of frames "
    "extracted from a screen-recording video, in the exact order they appear. "
    "Analyse ALL frames together as a single continuous workflow. "
    "First, identify what the user is trying to accomplish across the entire video. "
    "Then, break the workflow into sequential steps. "
    "Finally, assign each frame to its corresponding step and describe what is "
    "happening in that frame within the context of the overall workflow.\n\n"
    "Return a JSON object with:\n"
    '  "workflow": "<what the user is trying to accomplish>",\n'
    '  "steps": [\n'
    '    {\n'
    '      "step_number": <1-based>,\n'
    '      "step_title": "<short title for this step>",\n'
    '      "frames": [\n'
    '        {\n'
    '          "frame_index": <0-based index matching input order>,\n'
    '          "description": "<detailed description of what is shown and what the user is doing in this frame, in the context of this step and the overall workflow>",\n'
    '          "search_keywords": "<10-15 words focusing on: 1) the APPLICATION or WEBSITE name shown (e.g. Excel, Chrome, Canada411), 2) the specific USER ACTION being performed (e.g. typing, clicking, selecting, copying, pasting, opening, scrolling), 3) the key PAGE or SCREEN visible (e.g. login page, search results, spreadsheet, settings panel). Example: user typing phone number in Canada411 reverse lookup search field>"\n'
    '        }\n'
    '      ]\n'
    '    }\n'
    '  ]\n'
    "Return ONLY the JSON object."
)


def _load_image(filepath: str) -> Image.Image | None:
    if not os.path.isfile(filepath):
        return None
    try:
        img = Image.open(filepath)
        img.thumbnail((1024, 1024), Image.LANCZOS)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        return img
    except Exception:
        logger.exception("Failed to load image %s", filepath)
        return None


def describe_frames(
    api_key: str,
    frame_infos: list[dict],
    model: str = "gemini-2.5-flash",
) -> dict[str, dict]:
    """Send all frames to Gemini in order and return a mapping of
    filepath → {step_number, step_title, description}.

    ``frame_infos`` must already be sorted by video order (scene_id,
    then label within a scene).
    """
    client = genai.Client(api_key=api_key)

    # Build contents: images interleaved with a small caption per frame
    contents: list = []
    valid_frames: list[dict] = []

    for idx, fi in enumerate(frame_infos):
        img = _load_image(fi["filepath"])
        if img is None:
            continue
        contents.append(img)
        contents.append(f"Frame {idx}: {fi['filename']}")
        valid_frames.append(fi)

    if not valid_frames:
        logger.warning("No valid frames to describe")
        return {}

    contents.append(
        f"Describe all {len(valid_frames)} frames above. "
        "Return a JSON array with one object per frame in the same order."
    )

    config = types.GenerateContentConfig(
        system_instruction=_SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=65536,
        response_mime_type="application/json",
    )

    logger.info(
        "Sending %d frames to Gemini (%s) for batch description",
        len(valid_frames),
        model,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config,
    )

    raw_text = (response.text or "").strip()

    # Strip markdown fences if present
    if raw_text.startswith("```"):
        raw_text = raw_text.split("\n", 1)[-1]
        if raw_text.endswith("```"):
            raw_text = raw_text[: -len("```")].rstrip()

    descriptions = json.loads(raw_text)

    # Map back to filepaths
    result: dict[str, dict] = {}
    for step in descriptions.get("steps", []):
        step_number = step.get("step_number", 0)
        step_title = step.get("step_title", "")
        for frame in step.get("frames", []):
            idx = frame.get("frame_index", -1)
            if 0 <= idx < len(valid_frames):
                fi = valid_frames[idx]
                result[fi["filepath"]] = {
                    "step_number": step_number,
                    "step_title": step_title,
                    "description": frame.get("description", ""),
                    "search_keywords": frame.get("search_keywords", ""),
                }

    logger.info(
        "Gemini described %d / %d frames (workflow: %s)",
        len(result),
        len(valid_frames),
        descriptions.get("workflow", "unknown"),
    )
    return result
