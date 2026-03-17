# llm/vision_describer.py

"""
Vision model frame description and transition analysis.
Uses Gemini vision to analyze screen transitions.
Used by the video pipeline.
"""

import re
import os
import cv2
import numpy as np
from typing import List, Dict, Optional

from core.gemini_client import gemini_client
from core.config import config
from core.utils import (
    clean_vision_response, build_operation_context,
    detect_auth_screen, get_auth_step_description
)
from llm_tasks.system_prompts import VISION_SYSTEM_PROMPT


def _combine_images_side_by_side(
    img1_path: str, img2_path: str, max_height: int = 800
) -> Optional[str]:
    """Combine two images side-by-side into a single image."""
    try:
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)
        if img1 is None or img2 is None:
            return None

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        target_height = min(max_height, max(h1, h2))
        if h1 != target_height:
            scale = target_height / h1
            img1 = cv2.resize(img1, (int(w1 * scale), target_height))
        if h2 != target_height:
            scale = target_height / h2
            img2 = cv2.resize(img2, (int(w2 * scale), target_height))

        combined = np.hstack((img1, img2))
        base, ext = os.path.splitext(img1_path)
        combined_path = f"{base}_combined{ext}"
        cv2.imwrite(combined_path, combined)
        return combined_path

    except Exception as e:
        print(f"    [Vision] Error combining images: {e}")
        return None


def _sanitize_vision_response(text: str) -> str:
    """Remove prompt instructions that leak into output."""
    if not text:
        return ""

    patterns_to_remove = [
        r'PART\s*\d+\s*[—\-:.]?\s*[A-Z\s]*:?\s*',
        r'INSTRUCTIONS?:.*?(?=The system|The user|$)',
        r'OUTPUT FORMAT:.*?(?=SCREEN:|ACTION:|$)',
        r'BEFORE\s+screen\s+state:.*?(?=AFTER|ACTION|$)',
        r'AFTER\s+screen\s+state:.*?(?=ACTION|SCREEN|$)',
        r'LEFT\s*=\s*BEFORE.*?(?=SCREEN|ACTION|$)',
        r'RIGHT\s*=\s*AFTER.*?(?=SCREEN|ACTION|$)',
        r'Write\s+in\s+third\s+person.*?(?=SCREEN|ACTION|The |$)',
        r'Be\s+SPECIFIC.*?(?=SCREEN|ACTION|The |$)',
        r'[Pp]lease\s+provide.*',
        r'I\s+need\s+more\s+information.*',
        r'I\s+cannot\s+determine.*',
    ]

    cleaned = text
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)

    cleaned = re.sub(r'\n\s*\n+', '\n', cleaned)
    cleaned = re.sub(r'[ \t]+', ' ', cleaned)
    lines = [line.strip() for line in cleaned.split('\n') if line.strip()]
    return '\n'.join(lines).strip()


def _extract_screen_action(response: str) -> Dict[str, str]:
    """Extract SCREEN and ACTION sections from vision response."""
    result = {"screen_description": "", "action_description": ""}

    if not response:
        return result

    cleaned = _sanitize_vision_response(response)

    screen_match = re.search(
        r'SCREEN:\s*(.*?)(?=ACTION:|$)',
        cleaned, re.DOTALL | re.IGNORECASE
    )
    action_match = re.search(
        r'ACTION:\s*(.*?)$',
        cleaned, re.DOTALL | re.IGNORECASE
    )

    if screen_match:
        result["screen_description"] = screen_match.group(1).strip()
    if action_match:
        result["action_description"] = action_match.group(1).strip()

    # Fallback
    if not result["action_description"]:
        sentences = re.split(r'(?<=[.!?])\s+', cleaned)
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 20:
                lower = sent.lower()
                if any(lower.startswith(p) for p in
                       ['the user ', 'user ', 'clicked ', 'selected ', 'entered ']):
                    result["action_description"] = sent
                    break

    if not result["action_description"] and cleaned and len(cleaned) > 20:
        result["action_description"] = cleaned[:200].strip()
        if not result["action_description"].endswith('.'):
            result["action_description"] += "."

    return result


def _select_prompt(
    change_type: str,
    operation_category: str = None,
    auth_info: Dict = None
) -> str:
    """Select appropriate prompt based on context."""
    format_instruction = """
OUTPUT EXACTLY IN THIS FORMAT (no other text):

SCREEN:
[Describe the RIGHT screenshot. You MUST explicitly state the exact Tab Name, Breadcrumb path, and any selected Dropdown values visible.]

ACTION:
[Describe EXACTLY what the user clicked or typed to go from LEFT to RIGHT screenshot. Quote EXACT button names and exact dropdown values.]

EXAMPLES OF GOOD ACTIONS:
- "Clicked the 'Home' tab."
- "Selected 'Regional Team' from the user group dropdown."
- "Clicked the 'Export to CSV' button located in the top right."
- "Clicked 'Download' and a browser file download popup appeared."
"""

    if auth_info and auth_info.get("is_auth"):
        auth_type = auth_info.get("auth_type", "login")
        return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER ({auth_type} screen).
{format_instruction}
Focus on: exact credential fields, "Sign In" / "Next" buttons, or SSO prompts."""

    if operation_category == "Excel":
        return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER (Excel operation).
{format_instruction}
Focus on: Excel function used, exact column headers clicked, exact filter values applied (e.g., "Unselected '(Blanks)'")."""

    return f"""Two screenshots side-by-side: LEFT = BEFORE, RIGHT = AFTER.
{format_instruction}
CRITICAL RULE: DO NOT SUMMARIZE. Read the exact text on the screen."""


def describe_transition(
    frame_before_path: str,
    frame_after_path: str,
    ocr_diff_summary: str = "",
    change_type: str = "",
    operation_category: str = None,
    call_index: int = 0,
    auth_info: Dict = None
) -> Dict[str, str]:
    """Describe a transition between two frames using vision model."""
    combined_path = _combine_images_side_by_side(frame_before_path, frame_after_path)
    if combined_path is None:
        combined_path = frame_after_path

    prompt = _select_prompt(change_type, operation_category, auth_info)

    if ocr_diff_summary:
        ocr_hint = ocr_diff_summary[:200]
        prompt += f"\n\nContext hint: {ocr_hint}"

    response = gemini_client.generate(
        prompt=prompt,
        image_paths=[combined_path],
        system_prompt=VISION_SYSTEM_PROMPT,
        call_name=f"Transition_{call_index}",
        max_retries=2
    )

    # Cleanup temp file
    if combined_path not in [frame_before_path, frame_after_path]:
        if os.path.exists(combined_path):
            try:
                os.remove(combined_path)
            except:
                pass

    result = _extract_screen_action(response)

    if not result["action_description"]:
        result["action_description"] = "The user performed an action on the screen."
    if not result["screen_description"]:
        result["screen_description"] = "Screen state changed."

    return result


def _build_rich_ocr_description(
    ocr_before: str,
    ocr_after: str,
    ocr_diff: Dict,
    change_info: Dict,
    operations: List[Dict],
    auth_info: Dict = None
) -> str:
    """Build a detailed change description from OCR data alone."""
    parts = []

    if auth_info and auth_info.get("is_auth"):
        auth_desc = get_auth_step_description(
            auth_info["auth_type"],
            auth_info.get("indicators", [])
        )
        parts.append(auth_desc)

    change_type = change_info.get("change_type", "")
    if change_type == "page_transition":
        parts.append("The user navigated to a different page or screen.")
    elif change_type == "modal_popup":
        parts.append("A dialog or popup appeared on screen.")
    elif change_type == "form_input":
        parts.append("The user entered data or interacted with a form element.")

    added = ocr_diff.get("added_words", [])
    removed = ocr_diff.get("removed_words", [])

    if added:
        parts.append(f"New text visible: {', '.join(added[:15])}")
    if removed:
        parts.append(f"Previous text removed: {', '.join(removed[:10])}")

    if operations:
        delta_ops = [op for op in operations if op.get("source") == "delta"]
        if delta_ops:
            op_names = [op["display_name"] for op in delta_ops]
            parts.append(f"Operations detected: {', '.join(op_names)}")

    if parts:
        return " ".join(parts)

    return "The user performed an action on the screen."


def analyze_transitions_smart(
    key_frames: List[Dict],
    ocr_diffs: List[Dict] = None,
    detected_operations: List[List[Dict]] = None,
    change_data: List[Dict] = None,
    auth_flags: List[Dict] = None
) -> List[Dict]:
    """
    Smart transition analysis with auth-awareness.
    Selects which transitions need vision calls vs OCR-only.
    """
    if len(key_frames) < 2:
        return []

    total_pairs = len(key_frames) - 1
    max_vision = config.llm.max_vision_calls
    min_vision = min(config.llm.min_vision_calls, total_pairs)
    ocr_threshold = config.llm.ocr_sufficient_threshold

    # Score each transition
    scored = []
    for i in range(total_pairs):
        score = 0.0
        reasons = []

        before_auth = auth_flags[i] if auth_flags and i < len(auth_flags) else {}
        after_auth = auth_flags[i + 1] if auth_flags and i + 1 < len(auth_flags) else {}

        if before_auth.get("is_auth") or after_auth.get("is_auth"):
            score += 120.0
            reasons.append("auth_screen")

        if i == 0:
            score += 100.0
            reasons.append("first_transition")
        if i == total_pairs - 1:
            score += 90.0
            reasons.append("last_transition")

        if change_data and i < len(change_data):
            magnitude = change_data[i].get("pixel_change_magnitude", 0)
            change_type = change_data[i].get("change_type", "")

            if change_type == "page_transition":
                score += 60.0
                reasons.append("page_transition")
            elif change_type == "modal_popup":
                score += 50.0
                reasons.append("modal_popup")
            elif magnitude > 0.3:
                score += 40.0
                reasons.append("large_change")

        if detected_operations and i < len(detected_operations):
            ops = detected_operations[i]
            if ops:
                score += 30.0
                reasons.append("operations_detected")

        if ocr_diffs and i < len(ocr_diffs):
            ocr_ratio = ocr_diffs[i].get("change_ratio", 0)
            if ocr_ratio > ocr_threshold:
                score -= 15.0
                reasons.append("ocr_sufficient")

        if i > 0 and i < total_pairs - 1 and i % 5 == 0:
            score += 10.0
            reasons.append("periodic_check")

        scored.append((i, score, reasons))

    scored.sort(key=lambda x: x[1], reverse=True)

    vision_indices = set()
    for idx, score, reasons in scored:
        if len(vision_indices) >= max_vision:
            is_auth = any("auth" in r for r in reasons)
            if not is_auth:
                break
        if score > 0 or len(vision_indices) < min_vision:
            vision_indices.add(idx)

    print(
        f"    [Vision] Smart selection: {len(vision_indices)} vision calls "
        f"for {total_pairs} transitions"
    )

    # Process all transitions
    transitions = []
    vision_count = 0

    for i in range(total_pairs):
        before = key_frames[i]
        after = key_frames[i + 1]

        ocr_summary = ""
        if ocr_diffs and i < len(ocr_diffs):
            diff = ocr_diffs[i]
            added = diff.get("added_words", [])[:15]
            removed = diff.get("removed_words", [])[:15]
            if added:
                ocr_summary += f"New text: {', '.join(added)}\n"
            if removed:
                ocr_summary += f"Removed text: {', '.join(removed)}\n"

        operation_category = None
        if detected_operations and i < len(detected_operations):
            ops = detected_operations[i]
            if ops:
                operation_category = ops[0]["category"]

        change_type = ""
        if change_data and i < len(change_data):
            change_type = change_data[i].get("change_type", "")

        trans_auth = {}
        if auth_flags:
            before_auth = auth_flags[i] if i < len(auth_flags) else {}
            after_auth = auth_flags[i + 1] if i + 1 < len(auth_flags) else {}
            if before_auth.get("is_auth") or after_auth.get("is_auth"):
                trans_auth = before_auth if before_auth.get("is_auth") else after_auth

        if i in vision_indices:
            vision_result = describe_transition(
                before["path"], after["path"],
                ocr_diff_summary=ocr_summary,
                change_type=change_type,
                operation_category=operation_category,
                call_index=vision_count,
                auth_info=trans_auth
            )
            vision_count += 1

            after["vision_description"] = vision_result["screen_description"]

            transitions.append({
                "frame_before": before,
                "frame_after": after,
                "change_description": vision_result["action_description"],
                "timestamp_before": before.get("timestamp", 0),
                "timestamp_after": after.get("timestamp", 0),
                "order": i,
                "used_vision": True,
                "auth_info": trans_auth
            })

        else:
            ocr_desc = _build_rich_ocr_description(
                before.get("ocr_text", ""),
                after.get("ocr_text", ""),
                ocr_diffs[i] if ocr_diffs and i < len(ocr_diffs) else {},
                change_data[i] if change_data and i < len(change_data) else {},
                detected_operations[i] if detected_operations and i < len(detected_operations) else [],
                trans_auth
            )

            after["vision_description"] = after.get("ocr_text", "")

            transitions.append({
                "frame_before": before,
                "frame_after": after,
                "change_description": ocr_desc,
                "timestamp_before": before.get("timestamp", 0),
                "timestamp_after": after.get("timestamp", 0),
                "order": i,
                "used_vision": False,
                "auth_info": trans_auth
            })

    print(f"    [Vision] Complete: {vision_count} vision, "
          f"{total_pairs - vision_count} OCR-only")
    return transitions


def identify_application(frame_path: str) -> str:
    """Identify the application shown in a frame."""
    prompt = """What application, website, or software tool is shown in this screenshot?

OUTPUT EXACTLY IN THIS FORMAT:
APPLICATION: [application name only, e.g., "Microsoft Excel", "Google Chrome - Salesforce"]

Do not include any other text."""

    response = gemini_client.generate(
        prompt=prompt,
        image_paths=[frame_path],
        call_name="IdentifyApp"
    )

    if response:
        match = re.search(r'APPLICATION:\s*(.+)', response, re.IGNORECASE)
        if match:
            name = match.group(1).strip().strip('"\'')
            if 2 < len(name) < 50:
                return name

        name = response.strip().split('\n')[0].strip().strip('"\'')
        name = re.sub(r'^APPLICATION:\s*', '', name, flags=re.IGNORECASE)
        if 2 < len(name) < 50:
            return name

    return ""