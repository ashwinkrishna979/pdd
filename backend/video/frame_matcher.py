# audio/frame_matcher.py

"""
Frame-to-Step matching using OCR + Text Similarity.
Used by the audio pipeline to assign screenshot frames to detailed steps.

Pipeline:
1. OCR each frame to get on-screen text
2. For each detailed step, score all frames using OCR + transcript similarity
3. Assign best-matching frame to each step (no duplicates)
4. Fill unmatched steps chronologically
"""

import os
import re
from typing import List, Dict, Tuple, Optional, Set

from video.ocr_engine import ocr_batch, OCR_AVAILABLE


# ============================================================
# Text Similarity
# ============================================================

WORD_SYNONYMS = {
    'connect': {'login', 'log', 'sign', 'authenticate', 'access', 'open'},
    'login': {'connect', 'sign', 'log', 'authenticate', 'credentials', 'password'},
    'navigate': {'go', 'open', 'click', 'select', 'menu', 'tab', 'page', 'dashboard'},
    'extract': {'download', 'export', 'get', 'pull', 'fetch', 'retrieve', 'save'},
    'download': {'extract', 'export', 'save', 'get', 'fetch'},
    'validate': {'check', 'verify', 'confirm', 'ensure', 'review', 'inspect'},
    'filter': {'sort', 'search', 'find', 'select', 'criteria', 'column'},
    'process': {'handle', 'execute', 'perform', 'run', 'action', 'apply'},
    'generate': {'create', 'produce', 'build', 'make', 'report', 'output'},
    'report': {'generate', 'summary', 'output', 'results', 'details'},
    'update': {'modify', 'change', 'edit', 'set', 'status', 'save'},
    'credentials': {'login', 'password', 'username', 'user', 'authentication'},
    'application': {'portal', 'system', 'app', 'tool', 'platform', 'website'},
    'remove': {'delete', 'revoke', 'deactivate', 'disable', 'clear'},
    'license': {'licence', 'subscription', 'seat', 'entitlement'},
    'search': {'find', 'look', 'query', 'filter', 'locate'},
    'email': {'mail', 'notification', 'send', 'message', 'alert'},
}


def _extract_meaningful_words(text: str) -> Set[str]:
    """Extract meaningful words (lowercase, length > 2, no stopwords)."""
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from', 'are',
        'was', 'were', 'been', 'have', 'has', 'had', 'not', 'but',
        'all', 'can', 'will', 'would', 'should', 'could', 'may',
        'its', 'than', 'then', 'them', 'they', 'their', 'there',
        'each', 'which', 'when', 'where', 'how', 'who', 'whom',
        'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'over', 'some', 'any', 'also',
        'shall', 'must', 'using', 'based', 'upon'
    }
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    return set(words) - stopwords


def enhanced_similarity(text1: str, text2: str) -> float:
    """Enhanced similarity with synonym matching and importance weighting."""
    if not text1 or not text2:
        return 0.0

    words1 = _extract_meaningful_words(text1)
    words2 = _extract_meaningful_words(text2)

    if not words1 or not words2:
        return 0.0

    common_words = {
        'click', 'open', 'navigate', 'select', 'enter', 'system',
        'page', 'button', 'field', 'data', 'process', 'step',
        'application', 'portal', 'user', 'file', 'report',
        'status', 'update', 'check', 'verify', 'login', 'log',
    }

    direct_matches = words1 & words2

    synonym_matches = set()
    for w1 in words1:
        if w1 in direct_matches:
            continue
        synonyms = WORD_SYNONYMS.get(w1, set())
        for w2 in words2:
            if w2 in synonyms:
                synonym_matches.add(w1)
                break

    substring_matches = set()
    for w1 in words1:
        if w1 in direct_matches or w1 in synonym_matches:
            continue
        if len(w1) >= 5:
            for w2 in words2:
                if len(w2) >= 5 and w1[:5] == w2[:5]:
                    substring_matches.add(w1)
                    break

    all_words = words1 | words2
    if not all_words:
        return 0.0

    score = 0.0
    max_score = 0.0

    for word in all_words:
        if word in common_words:
            weight = 1.0
        elif len(word) >= 6:
            weight = 4.0
        else:
            weight = 2.0

        max_score += weight

        if word in direct_matches:
            score += weight
        elif word in synonym_matches:
            score += weight * 0.7
        elif word in substring_matches:
            score += weight * 0.5

    return score / max_score if max_score > 0 else 0.0


# ============================================================
# Frame-Step Scoring
# ============================================================

MIN_MATCH_SCORE = 0.02


def score_frame_against_step(
    frame_ocr_text: str,
    frame_transcript_text: str,
    step_description: str,
    ocr_weight: float = 0.6,
    transcript_weight: float = 0.4
) -> float:
    """Score how well a frame matches a step description."""
    ocr_score = enhanced_similarity(frame_ocr_text, step_description) if frame_ocr_text else 0.0
    transcript_score = enhanced_similarity(frame_transcript_text, step_description) if frame_transcript_text else 0.0

    if not frame_ocr_text:
        return transcript_score
    if not frame_transcript_text:
        return ocr_score

    weighted = (ocr_score * ocr_weight) + (transcript_score * transcript_weight)
    best_single = max(ocr_score, transcript_score)

    return (weighted * 0.7) + (best_single * 0.3)


# ============================================================
# Main Matching Logic
# ============================================================

def match_frames_to_steps(
    frame_candidates: List[Dict],
    detailed_steps: List[Dict],
    allow_reuse: bool = False
) -> List[Tuple[str, str]]:
    """Match frames to steps using OCR + transcript similarity."""
    if not frame_candidates or not detailed_steps:
        return []

    num_steps = len(detailed_steps)
    num_frames = len(frame_candidates)
    print(f"    [Matcher] Matching {num_frames} frames to {num_steps} steps...")

    # Build score matrix
    scores = []
    for step in detailed_steps:
        step_scores = []
        for frame in frame_candidates:
            score = score_frame_against_step(
                frame_ocr_text=frame.get("ocr", ""),
                frame_transcript_text=frame.get("transcript", ""),
                step_description=step.get("description", "")
            )
            step_scores.append(score)
        scores.append(step_scores)

    # Greedy assignment
    assigned = []
    used_frames: Set[int] = set()

    for step_idx in range(num_steps):
        best_frame_idx = -1
        best_score = -1.0

        for frame_idx in range(num_frames):
            if not allow_reuse and frame_idx in used_frames:
                continue
            if scores[step_idx][frame_idx] > best_score:
                best_score = scores[step_idx][frame_idx]
                best_frame_idx = frame_idx

        if best_frame_idx >= 0 and best_score >= MIN_MATCH_SCORE:
            frame = frame_candidates[best_frame_idx]
            assigned.append((
                frame["path"],
                detailed_steps[step_idx].get("description", "")
            ))
            if not allow_reuse:
                used_frames.add(best_frame_idx)
        else:
            assigned.append((
                "",
                detailed_steps[step_idx].get("description", "")
            ))

    matched = sum(1 for path, _ in assigned if path)
    print(f"    [Matcher] Matched: {matched}/{num_steps} steps")
    return assigned


def fill_unmatched_chronologically(
    assigned: List[Tuple[str, str]],
    frame_candidates: List[Dict],
    used_paths: Set[str] = None
) -> List[Tuple[str, str]]:
    """Fill unmatched steps with remaining frames in chronological order."""
    if used_paths is None:
        used_paths = set(path for path, _ in assigned if path)

    unused = [
        f for f in frame_candidates
        if f["path"] not in used_paths
    ]
    unused.sort(key=lambda f: f.get("timestamp", 0))
    unused_iter = iter(unused)

    filled = []
    for path, desc in assigned:
        if path:
            filled.append((path, desc))
        else:
            next_frame = next(unused_iter, None)
            if next_frame:
                filled.append((next_frame["path"], desc))
                used_paths.add(next_frame["path"])
            else:
                filled.append(("", desc))
    return filled


# ============================================================
# Complete Pipeline
# ============================================================

def match_pipeline(
    frame_candidates: List[Dict],
    detailed_steps: List[Dict],
    run_ocr: bool = True
) -> List[Tuple[str, str]]:
    """Complete frame matching pipeline."""
    if not frame_candidates:
        return [("", step.get("description", "")) for step in detailed_steps]
    if not detailed_steps:
        return []

    # Step 1: OCR
    if run_ocr and OCR_AVAILABLE:
        paths = [f["path"] for f in frame_candidates]
        ocr_results = ocr_batch(paths)
        for frame in frame_candidates:
            frame["ocr"] = ocr_results.get(frame["path"], {}).get("text", "")
    else:
        for frame in frame_candidates:
            frame["ocr"] = ""

    # Step 2: Score and match
    assigned = match_frames_to_steps(frame_candidates, detailed_steps)

    # Step 3: Fill unmatched chronologically
    assigned = fill_unmatched_chronologically(assigned, frame_candidates)

    return assigned


def build_candidates(
    frame_pairs: List[Tuple[str, str]],
    timestamps: List[float] = None
) -> List[Dict]:
    """Convert (path, transcript_text) pairs into candidate dicts."""
    candidates = []
    for i, (path, text) in enumerate(frame_pairs):
        ts = timestamps[i] if timestamps and i < len(timestamps) else i * 10.0
        candidates.append({
            "path": path,
            "transcript": text,
            "ocr": "",
            "timestamp": ts
        })
    return candidates