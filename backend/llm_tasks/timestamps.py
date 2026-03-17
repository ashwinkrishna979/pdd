# llm_tasks/timestamps.py

"""
Key timestamp identification and description paraphrasing.
Used by the audio pipeline to identify action moments in transcript.
Kept for frame extraction support.
"""

import re
import time
from typing import Dict, List

from core.gemini_client import gemini_client
from core.config import ACTION_KEYWORDS
from core.utils import timed, parse_numbered_steps, redact_pii_text
from llm_tasks.system_prompts import get_system_prompt


def identify_key_timestamps(
    transcript: str,
    transcript_path: str
) -> List[Dict]:
    """
    Identify key action timestamps from transcript.
    Uses keyword matching (no LLM call needed) to save API quota.
    """
    start = time.time()
    lines = []

    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(
                    r'\[(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\]\s+(.*)',
                    line.strip()
                )
                if m:
                    lines.append({
                        "timestamp": float(m.group(1)),
                        "text": m.group(3).strip()
                    })
    except Exception as e:
        print(f"    [Timestamps] Error reading transcript: {e}")
        return []

    if not lines:
        return []

    # Keyword matching (no LLM call — saves API quota)
    all_kw = set()
    for kl in ACTION_KEYWORDS.values():
        for kw in kl:
            all_kw.add(kw.lower())

    moments = []
    for tl in lines:
        if any(kw in tl["text"].lower() for kw in all_kw):
            moments.append({
                "timestamp": tl["timestamp"],
                "description": redact_pii_text(tl["text"])
            })

    # Deduplicate close timestamps
    if moments:
        deduped = [moments[0]]
        for km in moments[1:]:
            if abs(km["timestamp"] - deduped[-1]["timestamp"]) > 5.0:
                deduped.append(km)
        moments = deduped

    # Limit
    if len(moments) > 20:
        s = len(moments) // 20
        moments = moments[::s][:20]

    timed(f"Timestamps ({len(moments)})", start)
    return moments


def paraphrase_batch(
    texts: List[str],
    batch_size: int = 8
) -> List[str]:
    """
    Paraphrase frame descriptions into professional process steps.
    Larger batch size to reduce API calls.
    """
    if not texts:
        return []

    results = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        numbered = "\n".join(
            [f"{j+1}. {t[:120]}" for j, t in enumerate(batch)]
        )

        prompt = f"""Rewrite each description as a professional PDD process step.

RULES:
- Each step describes what THE SYSTEM does.
- Third person, present tense, active voice.
- 1 sentence each, starting with "The system..."
- NEVER include personal names or email addresses.

{numbered}

OUTPUT (numbered list only):
1."""

        response = gemini_client.generate(
            prompt=prompt,
            system_prompt=get_system_prompt(),
            call_name=f"Paraphrase_batch{i//batch_size + 1}"
        )

        batch_results = []
        if response:
            if not response.strip().startswith("1"):
                response = "1. " + response
            batch_results = parse_numbered_steps(response)

        for j in range(len(batch)):
            if j < len(batch_results):
                results.append(redact_pii_text(batch_results[j]))
            else:
                results.append(redact_pii_text(batch[j][:120]))

    return results