# core/utils.py

"""
Shared utility functions used across both pipelines.
Text processing, sampling, entity verification, operation detection,
auth detection, PII redaction, tone validation.
"""

import re
import time
import os
from typing import List, Set, Dict, Optional

from core.config import (
    config, EXCEL_OPERATIONS, WEB_OPERATIONS,
    GENERAL_OPERATIONS, AUTH_VISUAL_INDICATORS
)


# ============================================================
# Timing
# ============================================================

def timed(name: str, start: float):
    """Log elapsed time for a task."""
    print(f"    [{name}] done in {time.time() - start:.1f}s")


# ============================================================
# Text Sampling
# ============================================================

def safe_sample(text: str, max_len: int = None) -> str:
    """Take a safe-sized sample. Beginning + End."""
    max_len = max_len or config.llm.max_sample_text
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    first = int(max_len * 0.6)
    last = max_len - first - 30
    return text[:first] + "\n[...]\n" + text[-last:]


def split_into_chunks(text: str) -> List[str]:
    """Split text into overlapping chunks."""
    chunk_size = config.llm.chunk_size
    max_chunks = config.llm.max_chunks
    overlap = config.llm.overlap_size

    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text) and len(chunks) < max_chunks:
        end = min(start + chunk_size, len(text))
        if end < len(text):
            bp = text.rfind('. ', start + chunk_size - 300, end)
            if bp != -1:
                end = bp + 1
        chunks.append(text[start:end].strip())
        start = end - overlap
        if start <= 0 and chunks:
            break
    return chunks


# ============================================================
# PII Redaction
# ============================================================

# Common name patterns (first names) — kept small, extend as needed
_COMMON_FIRST_NAMES = {
    'james', 'john', 'robert', 'michael', 'william', 'david', 'richard',
    'joseph', 'thomas', 'charles', 'mary', 'patricia', 'jennifer', 'linda',
    'elizabeth', 'barbara', 'susan', 'jessica', 'sarah', 'karen', 'nancy',
    'daniel', 'matthew', 'anthony', 'mark', 'donald', 'steven', 'paul',
    'andrew', 'joshua', 'kenneth', 'kevin', 'brian', 'george', 'timothy',
    'ronald', 'edward', 'jason', 'jeffrey', 'ryan', 'jacob', 'gary',
    'nicholas', 'eric', 'jonathan', 'stephen', 'larry', 'justin', 'scott',
    'brandon', 'benjamin', 'samuel', 'raymond', 'gregory', 'frank', 'alexander',
    'patrick', 'jack', 'dennis', 'jerry', 'tyler', 'aaron', 'jose',
    'adam', 'nathan', 'henry', 'peter', 'zachary', 'douglas', 'harold',
    'amy', 'angela', 'melissa', 'brenda', 'anna', 'samantha', 'katherine',
    'christine', 'deborah', 'rachel', 'carolyn', 'janet', 'catherine',
    'maria', 'heather', 'diane', 'ruth', 'julie', 'olivia', 'joyce',
    'virginia', 'victoria', 'kelly', 'lauren', 'christina', 'joan',
    'evelyn', 'judith', 'megan', 'andrea', 'cheryl', 'hannah', 'jacqueline',
    'martha', 'gloria', 'teresa', 'ann', 'sara', 'madison', 'frances',
    'kathryn', 'janice', 'jean', 'abigail', 'alice', 'judy', 'sophia',
    'grace', 'denise', 'amber', 'doris', 'marilyn', 'danielle', 'beverly',
    'isabella', 'theresa', 'diana', 'natalie', 'brittany', 'charlotte',
    'marie', 'kayla', 'alexis', 'lori',
}

# Email pattern
_EMAIL_PATTERN = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

# Phone patterns (US/international)
_PHONE_PATTERNS = [
    re.compile(r'\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
    re.compile(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'),
    re.compile(r'\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b'),
    re.compile(r'\b\+\d{1,3}[-.\s]?\d{4,14}\b'),
]

# Name-like patterns: "Mr./Mrs./Dr. Lastname", "FirstName LastName"
_NAME_PATTERNS = [
    re.compile(r'\b(?:Mr|Mrs|Ms|Miss|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b'),
    re.compile(r'\b[A-Z][a-z]{2,15}\s+[A-Z][a-z]{2,20}\b'),
]


def redact_pii_text(text: str) -> str:
    """Redact PII from text: emails, phone numbers, personal names."""
    if not text or not config.redaction.enabled or not config.redaction.redact_in_text:
        return text

    placeholder = config.redaction.redaction_placeholder
    redacted = text

    # Redact emails
    if config.redaction.redact_emails:
        redacted = _EMAIL_PATTERN.sub(placeholder, redacted)

    # Redact phone numbers
    if config.redaction.redact_phone_numbers:
        for pattern in _PHONE_PATTERNS:
            redacted = pattern.sub(placeholder, redacted)

    # Redact names
    if config.redaction.redact_names:
        for pattern in _NAME_PATTERNS:
            matches = pattern.finditer(redacted)
            for match in sorted(matches, key=lambda m: m.start(), reverse=True):
                candidate = match.group()
                words = candidate.split()
                # Check if any word is a known first name
                has_name = any(w.lower() in _COMMON_FIRST_NAMES for w in words)
                # Also check "Title Lastname" pattern
                has_title = any(
                    w.lower().rstrip('.') in {'mr', 'mrs', 'ms', 'miss', 'dr', 'prof'}
                    for w in words
                )
                if has_name or has_title:
                    redacted = redacted[:match.start()] + placeholder + redacted[match.end():]

    return redacted


def redact_pii_from_image(image_path: str, ocr_boxes: List[Dict] = None) -> str:
    """
    Redact PII from screenshot by blacking out regions containing PII text.
    Returns path to redacted image (overwrites original).
    """
    if not config.redaction.enabled or not config.redaction.redact_in_screenshots:
        return image_path

    if not image_path or not os.path.exists(image_path):
        return image_path

    try:
        import cv2
        import numpy as np
    except ImportError:
        return image_path

    img = cv2.imread(image_path)
    if img is None:
        return image_path

    modified = False

    if ocr_boxes:
        for box in ocr_boxes:
            word = box.get("text", "")
            if _is_pii_word(word):
                x = box.get("x", 0)
                y = box.get("y", 0)
                w = box.get("w", 0)
                h = box.get("h", 0)
                pad = 3
                x1 = max(0, x - pad)
                y1 = max(0, y - pad)
                x2 = min(img.shape[1], x + w + pad)
                y2 = min(img.shape[0], y + h + pad)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
                modified = True
    else:
        # If no OCR boxes provided, try OCR to find PII regions
        try:
            from video.ocr_engine import OCR_AVAILABLE
            if OCR_AVAILABLE:
                import pytesseract
                from PIL import Image
                pil_img = Image.open(image_path)
                box_data = pytesseract.image_to_data(
                    pil_img, output_type=pytesseract.Output.DICT, config='--psm 6'
                )
                for i in range(len(box_data['text'])):
                    word = box_data['text'][i].strip()
                    conf = int(box_data['conf'][i])
                    if word and conf > 30 and _is_pii_word(word):
                        x = box_data['left'][i]
                        y = box_data['top'][i]
                        w = box_data['width'][i]
                        h = box_data['height'][i]
                        pad = 3
                        x1 = max(0, x - pad)
                        y1 = max(0, y - pad)
                        x2 = min(img.shape[1], x + w + pad)
                        y2 = min(img.shape[0], y + h + pad)
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
                        modified = True
        except Exception:
            pass

    if modified:
        cv2.imwrite(image_path, img)

    return image_path


def _is_pii_word(word: str) -> bool:
    """Check if a single word/token is PII."""
    if not word or len(word) < 2:
        return False

    # Email check
    if _EMAIL_PATTERN.match(word):
        return True

    # Phone fragments — skip, handled at sentence level
    # Name check
    if word.lower() in _COMMON_FIRST_NAMES and word[0].isupper():
        return True

    # "@" in word likely email fragment
    if '@' in word:
        return True

    return False


# ============================================================
# Tone / Style Validation
# ============================================================

_FIRST_PERSON_PATTERNS = re.compile(
    r'\b(I|we|my|our|me|us|myself|ourselves)\b', re.IGNORECASE
)
_INFORMAL_STARTS = [
    'i want', 'i need', 'we need', 'we want', 'let me', 'let us',
    'you should', 'you need', 'you can', 'please note',
    'as discussed', 'as mentioned', 'as per our', 'as we discussed',
    'in the meeting', 'during the call', 'the transcript',
    'the recording', 'the video', 'the speaker',
]


def enforce_tone(text: str) -> str:
    """
    Post-process text to enforce third-person, present-tense, formal tone.
    Removes first-person references and informal language.
    """
    if not text:
        return text

    lines = text.split('\n')
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            cleaned_lines.append(line)
            continue

        # Skip lines that are pure informal references
        lower = stripped.lower()
        if any(lower.startswith(p) for p in _INFORMAL_STARTS):
            continue

        # Replace first-person with third-person
        cleaned = _FIRST_PERSON_PATTERNS.sub(
            lambda m: _first_to_third(m.group()), stripped
        )

        # Remove meeting/transcript references
        cleaned = re.sub(
            r'\b(?:in the meeting|during the call|as discussed|'
            r'the transcript|the recording|the speaker said|'
            r'they mentioned|it was discussed)\b',
            '', cleaned, flags=re.IGNORECASE
        )

        # Clean up double spaces
        cleaned = re.sub(r'\s{2,}', ' ', cleaned).strip()

        if cleaned:
            cleaned_lines.append(cleaned)

    return '\n'.join(cleaned_lines)


def _first_to_third(word: str) -> str:
    """Convert first-person pronoun to third-person equivalent."""
    mapping = {
        'i': '', 'I': '',
        'we': '', 'We': '',
        'my': 'the', 'My': 'The',
        'our': 'the', 'Our': 'The',
        'me': '', 'Me': '',
        'us': '', 'Us': '',
        'myself': 'itself', 'ourselves': 'itself',
    }
    return mapping.get(word, word)


# ============================================================
# Entity Helpers
# ============================================================

def build_entity_hint(entities: Dict) -> str:
    """Build a hint string from extracted entities."""
    parts = []
    if entities.get("companies"):
        parts.append(f"Companies: {', '.join(entities['companies'])}")
    if entities.get("applications"):
        parts.append(f"Applications: {', '.join(entities['applications'])}")
    if entities.get("systems"):
        parts.append(f"Systems: {', '.join(entities['systems'])}")
    if parts:
        return "Entities from transcript: " + "; ".join(parts)
    return ""


def verify_entities_against_transcript(entities: Dict, transcript: str) -> Dict:
    """Remove entity names that don't appear in the transcript."""
    transcript_lower = transcript.lower()

    def _appears(name: str) -> bool:
        name_lower = name.lower().strip()
        if name_lower in transcript_lower:
            return True
        words = name_lower.split()
        if len(words) > 1:
            significant = [w for w in words if len(w) > 3]
            if significant and all(w in transcript_lower for w in significant):
                return True
        if len(name_lower) >= 4 and name_lower[:4] in transcript_lower:
            return True
        no_space = name_lower.replace(" ", "")
        if len(no_space) >= 4 and no_space in transcript_lower.replace(" ", ""):
            return True
        return False

    verified = {}
    for key, items in entities.items():
        if isinstance(items, list):
            verified_items = []
            for item in items:
                if _appears(item):
                    verified_items.append(item)
                else:
                    print(f"    [Entities] Removed hallucinated: '{item}'")
            verified[key] = verified_items
        else:
            verified[key] = items
    return verified


# ============================================================
# Step Parsing and Filtering
# ============================================================

# Only filter out pure coordination/meeting-scheduling steps,
# NOT steps that contain valid process action verbs
CONVERSATION_PHRASES = [
    'schedule meeting', 'set up a call', 'send invite',
    'book a meeting', 'touch base', 'circle back',
    'let me know', 'get back to', 'agenda item',
    'stakeholder meeting',
]

# Phrases that should NEVER be filtered even if they contain
# words that might look conversational
PROCESS_PRESERVE_PHRASES = [
    'check', 'verify', 'validate', 'filter', 'export', 'extract',
    'download', 'upload', 'navigate', 'click', 'select', 'enter',
    'login', 'log in', 'log out', 'logout', 'open', 'close',
    'save', 'update', 'remove', 'delete', 'create', 'generate',
    'process', 'submit', 'confirm', 'apply', 'search', 'copy',
    'paste', 'repeat', 'iterate', 'loop', 'for each', 'if ',
    'capture', 'record', 'log ', 'track', 'report', 'status',
    'revoke', 'assign', 'unassign', 'disable', 'enable',
    'query', 'script', 'execute', 'run', 'trigger',
]


def filter_conversation_steps(steps: List[str]) -> List[str]:
    """
    Remove steps that describe pure conversations/meetings, not process actions.
    Much less aggressive than before — preserves steps with process action verbs.
    """
    filtered = []
    for s in steps:
        s_lower = s.lower()

        # Always keep if it contains process action verbs
        has_process_action = any(phrase in s_lower for phrase in PROCESS_PRESERVE_PHRASES)
        if has_process_action:
            filtered.append(s)
            continue

        # Only remove if it matches pure conversation phrases
        is_conversation = any(phrase in s_lower for phrase in CONVERSATION_PHRASES)
        if is_conversation:
            print(f"    [Filter] Removed conversation step: '{s[:60]}...'")
            continue

        # Keep by default
        filtered.append(s)

    return filtered if filtered else steps


def parse_numbered_steps(text: str) -> List[str]:
    """Parse numbered steps from LLM response."""
    steps = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r'^[\d]+[\.\)]\s*', '', line).strip()
        cleaned = re.sub(r'^[-•*➤]\s*', '', cleaned).strip()
        cleaned = cleaned.strip('"')
        if not cleaned or len(cleaned) < 10:
            continue
        skip = [
            'here are', 'following', 'process steps', 'transcript',
            'note:', 'based on', 'the above', 'these are',
            'below', 'i have', 'let me', 'sure,', 'certainly',
            'wrong', 'correct', 'example',
            'use only', 'names from'
        ]
        if any(cleaned.lower().startswith(p) for p in skip):
            continue
        if cleaned.startswith('WRONG') or cleaned.startswith('RIGHT'):
            continue
        if cleaned.isupper() or cleaned.endswith(':'):
            continue
        steps.append(cleaned)
    return steps


def deduplicate_steps(steps: List[str]) -> List[str]:
    """
    Remove near-duplicate steps.
    Less aggressive — requires higher similarity threshold and
    preserves steps that act on different targets.
    """
    if len(steps) <= 1:
        return steps

    unique = []
    for s in steps:
        # Normalize for comparison
        key = re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()

        # Check against existing unique steps
        is_duplicate = False
        for existing in unique:
            existing_key = re.sub(r'[^a-z0-9 ]', '', existing.lower()).strip()

            # Only consider duplicate if very high word overlap
            words_new = set(key.split())
            words_existing = set(existing_key.split())

            if not words_new or not words_existing:
                continue

            # Calculate Jaccard similarity
            intersection = words_new & words_existing
            union = words_new | words_existing
            similarity = len(intersection) / len(union) if union else 0

            # Only mark as duplicate if >85% word overlap
            if similarity > 0.85:
                is_duplicate = True
                break

        if not is_duplicate and len(key) > 5:
            unique.append(s)

    return unique


def clean_vision_response(text: str) -> str:
    """Clean up vision model response text."""
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    return '\n'.join(lines).strip()


# ============================================================
# Auth / Login Detection
# ============================================================

def detect_auth_screen(ocr_text: str, vision_text: str = "") -> Dict:
    """Detect if a frame shows a login/logout/auth screen."""
    combined = f"{ocr_text} {vision_text}".lower()
    if not combined.strip():
        return {"is_auth": False, "auth_type": "", "confidence": 0.0, "indicators": []}

    found_indicators = []
    for indicator in AUTH_VISUAL_INDICATORS:
        if indicator in combined:
            found_indicators.append(indicator)

    if not found_indicators:
        return {"is_auth": False, "auth_type": "", "confidence": 0.0, "indicators": []}

    login_words = {"sign in", "log in", "login", "signin", "username", "password",
                   "credentials", "welcome", "sso", "authenticate"}
    logout_words = {"sign out", "log out", "logout", "signout", "signed out",
                    "goodbye", "end session", "session expired"}
    password_words = {"change password", "reset password", "forgot password",
                      "new password", "password expired", "update password"}
    mfa_words = {"mfa", "otp", "verification code", "two-factor", "2fa",
                 "multi-factor", "captcha"}

    login_hits = sum(1 for i in found_indicators if i in login_words or
                     any(lw in i for lw in login_words))
    logout_hits = sum(1 for i in found_indicators if i in logout_words or
                      any(lw in i for lw in logout_words))
    password_hits = sum(1 for i in found_indicators if i in password_words or
                        any(pw in i for pw in password_words))
    mfa_hits = sum(1 for i in found_indicators if i in mfa_words or
                   any(mw in i for mw in mfa_words))

    auth_type = "unknown_auth"
    if logout_hits > 0 and logout_hits >= login_hits:
        auth_type = "logout"
    elif password_hits > 0 and password_hits >= login_hits:
        auth_type = "password_change"
    elif mfa_hits > 0:
        auth_type = "mfa_verification"
    elif login_hits > 0:
        auth_type = "login"

    confidence = min(1.0, len(found_indicators) / 3.0)
    has_field = any(i in ["username", "user name", "password", "email", "user id", "userid"]
                    for i in found_indicators)
    has_action = any(i in ["sign in", "log in", "login", "submit", "continue",
                           "sign out", "log out", "logout"]
                     for i in found_indicators)
    if has_field and has_action:
        confidence = min(1.0, confidence + 0.3)

    return {
        "is_auth": confidence >= 0.3,
        "auth_type": auth_type,
        "confidence": confidence,
        "indicators": found_indicators
    }


def get_auth_step_description(auth_type: str, indicators: List[str] = None,
                               app_name: str = "") -> str:
    """Generate a detailed auth step description for PDD."""
    app = app_name or "the application"
    descriptions = {
        "login": (
            f"Navigate to the login page of {app}. "
            f"Enter the configured username/user ID and password. "
            f"Click the 'Sign In' button to authenticate."
        ),
        "logout": (
            f"Initiate the logout process from {app}. "
            f"Click the logout option to end the session."
        ),
        "mfa_verification": (
            f"Handle a multi-factor authentication prompt "
            f"in {app} by completing the verification challenge."
        ),
        "password_change": (
            f"Navigate to the password management section of {app} "
            f"and submit the password change."
        ),
        "unknown_auth": (
            f"Interact with an authentication screen in {app}."
        ),
    }
    return descriptions.get(auth_type, descriptions["unknown_auth"])


# ============================================================
# Operation Detection (Delta-based)
# ============================================================

def _extract_words(text: str) -> List[str]:
    """Extract meaningful words from text."""
    if not text:
        return []
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from',
        'are', 'was', 'not', 'but', 'all', 'can', 'will',
        'has', 'have', 'had', 'its', 'than', 'then', 'which',
        'would', 'could', 'should', 'into', 'over', 'under'
    }
    return [w for w in words if w not in stopwords]


def detect_operations_delta(
    ocr_before: str, ocr_after: str,
    change_description: str = ""
) -> List[Dict]:
    """Detect operations based on what CHANGED between frames."""
    detected = []
    words_before = set(_extract_words(ocr_before))
    words_after = set(_extract_words(ocr_after))
    added_words = words_after - words_before
    removed_words = words_before - words_after

    delta_text = ' '.join(added_words | removed_words)
    action_text = change_description.lower() if change_description else ""

    all_ops = {
        "Excel": EXCEL_OPERATIONS,
        "Web": WEB_OPERATIONS,
        "General": GENERAL_OPERATIONS,
    }

    for category, ops_dict in all_ops.items():
        for op_name, keywords in ops_dict.items():
            for kw in keywords:
                if kw in action_text:
                    detected.append({
                        "category": category,
                        "operation": op_name,
                        "keyword_matched": kw,
                        "display_name": format_operation_name(op_name),
                        "source": "vision_action",
                        "confidence": 0.9
                    })
                    break
                elif kw in delta_text.lower():
                    detected.append({
                        "category": category,
                        "operation": op_name,
                        "keyword_matched": kw,
                        "display_name": format_operation_name(op_name),
                        "source": "delta",
                        "confidence": 0.7
                    })
                    break

    # Deduplicate
    seen_ops = {}
    for op in detected:
        key = (op["category"], op["operation"])
        if key not in seen_ops or op["confidence"] > seen_ops[key]["confidence"]:
            seen_ops[key] = op
    return list(seen_ops.values())


def format_operation_name(op_name: str) -> str:
    """Convert operation key to display name."""
    display_map = {
        "vlookup": "VLOOKUP Formula", "hlookup": "HLOOKUP Formula",
        "filter": "Data Filter", "sort": "Data Sort",
        "pivot_table": "Pivot Table", "duplicate_removal": "Duplicate Removal",
        "formula": "Formula/Calculation", "copy_paste": "Copy & Paste",
        "find_replace": "Find & Replace", "chart": "Chart/Graph",
        "macro": "Macro Execution", "import_export": "Import/Export Data",
        "data_validation": "Data Validation",
        "login": "User Login/Authentication", "logout": "User Logout",
        "navigate": "Page Navigation", "search": "Search/Query",
        "form_fill": "Form Data Entry", "upload": "File Upload",
        "download": "File Download", "submit": "Form Submission",
        "select": "Selection", "modal_dialog": "Dialog Interaction",
        "open_application": "Application Launch",
        "close_application": "Application Close",
        "switch_window": "Window Switch", "email": "Email Operation",
        "file_operation": "File Management",
    }
    return display_map.get(op_name, op_name.replace('_', ' ').title())


def build_operation_context(operations: List[Dict]) -> str:
    """Build a context string from detected operations."""
    if not operations:
        return ""
    high_conf = [op for op in operations
                 if op.get("source") in ("delta", "vision_action")
                 and op.get("confidence", 0) >= 0.7]
    if not high_conf:
        return ""
    lines = ["Detected operations:"]
    for op in high_conf[:3]:
        lines.append(f"- {op['display_name']}")
    return "\n".join(lines)