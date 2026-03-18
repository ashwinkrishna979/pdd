"""Clean and compress OmniParser output into compact tokens for embedding and metadata."""

import ast
import re

# ---- Filtering config ----

_IGNORE_KEYWORDS = {
    "refresh", "reload", "wifi", "wi-fi", "signal",
    "sunny", "temperature", "eng",
    "notification", "browser", "edge",
    "excel", "word", "symbol", "logo",
    "bookmark", "unanswerable",
    "checkmark", "loading", "buffering",
    "folder", "file folder",
}

_USELESS_ICON_TEXT = {
    "a symbol or symbol",
    "a symbol or logo",
    "a bookmark",
    "the number 8",
    "the corner of an image",
}

_INTERESTING_TYPES = {"text", "icon"}
_MIN_TEXT_LENGTH = 2
_MAX_TEXT_LENGTH = 80


# ---- Helpers ----


def _parse_elements(raw_output: dict) -> dict:
    raw_elements = raw_output.get("elements", "")
    elements: dict = {}
    for line in raw_elements.split("\n"):
        if "{" not in line:
            continue
        try:
            obj_str = line.split(":", 1)[1].strip()
            obj = ast.literal_eval(obj_str)
            elements[f"elem_{len(elements)}"] = obj
        except Exception:
            continue
    return elements


def _normalize(text: str) -> str:
    text = text.strip().lower()
    return re.sub(r"\s+", " ", text)


def _is_noise(text: str) -> bool:
    if len(text) < _MIN_TEXT_LENGTH or len(text) > _MAX_TEXT_LENGTH:
        return True
    if any(k in text for k in _IGNORE_KEYWORDS):
        return True
    return text in _USELESS_ICON_TEXT


def _quantize_bbox(bbox: list[float]) -> str:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    h = "L" if cx < 0.33 else ("C" if cx < 0.66 else "R")
    v = "T" if cy < 0.33 else ("M" if cy < 0.66 else "B")
    return f"{v}-{h}"


def _compress(elements: dict) -> list[dict]:
    seen: set[str] = set()
    cleaned: list[dict] = []
    for e in elements.values():
        if e.get("type") not in _INTERESTING_TYPES:
            continue
        content = _normalize(e.get("content", ""))
        if _is_noise(content) or content in seen:
            continue
        seen.add(content)
        bbox = e.get("bbox")
        cleaned.append(
            {
                "text": content,
                "type": e["type"],
                "pos": _quantize_bbox(bbox) if bbox else None,
            }
        )
    return cleaned


def _to_tokens(cleaned: list[dict]) -> list[str]:
    tokens: list[str] = []
    for e in cleaned:
        prefix = "ICO" if e["type"] == "icon" else "TXT"
        pos = e["pos"] or ""
        tokens.append(f"{prefix}:{e['text']}@{pos}")
    return tokens


def _generate_summary(tokens: list[str]) -> dict:
    texts, actions = [], []
    for t in tokens:
        payload = t.split(":")[1].split("@")[0]
        if t.startswith("TXT"):
            texts.append(payload)
        elif t.startswith("ICO"):
            actions.append(payload)
    return {
        "visible_text": texts[:20],
        "possible_actions": actions[:10],
    }


# ---- Public API ----


def process_omniparser_output(frame_id: int, raw_output: dict) -> dict:
    """Full pipeline: parse → clean → tokenize → summarize."""
    elements = _parse_elements(raw_output)
    cleaned = _compress(elements)
    tokens = _to_tokens(cleaned)
    summary = _generate_summary(tokens)
    return {
        "frame_id": frame_id,
        "compact_elements": tokens,
        "summary": summary,
        "vector_metadata": {
            "frame_id": frame_id,
            "elements": tokens,
            "search_text": " ".join(tokens),
        },
    }
