# video/ocr_engine.py

"""
Tesseract OCR engine for frame text extraction.
Shared by both audio (frame_matcher) and video (vision_describer) pipelines.
"""

import os
import re
from typing import List, Dict

try:
    import pytesseract
    from PIL import Image
    pytesseract.pytesseract.tesseract_cmd = (
        r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    )
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("    [OCR] pytesseract not available. Install: pip install pytesseract Pillow")


def ocr_frame(frame_path: str, with_boxes: bool = False) -> Dict:
    """Extract text from a single frame."""
    result = {"text": "", "boxes": [], "word_count": 0}

    if not OCR_AVAILABLE:
        return result

    if not os.path.exists(frame_path):
        return result

    try:
        image = Image.open(frame_path)
        text = pytesseract.image_to_string(image, config='--psm 6 --oem 3')
        text = _clean_ocr_text(text)
        result["text"] = text
        result["word_count"] = len(text.split()) if text else 0

        if with_boxes and text:
            box_data = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT, config='--psm 6'
            )
            boxes = []
            for i in range(len(box_data['text'])):
                word = box_data['text'][i].strip()
                conf = int(box_data['conf'][i])
                if word and conf > 30:
                    boxes.append({
                        "text": word,
                        "x": box_data['left'][i],
                        "y": box_data['top'][i],
                        "w": box_data['width'][i],
                        "h": box_data['height'][i],
                        "confidence": conf
                    })
            result["boxes"] = boxes

        return result

    except Exception as e:
        print(f"    [OCR] Error on {os.path.basename(frame_path)}: {e}")
        return result


def ocr_batch(
    frame_paths: List[str],
    with_boxes: bool = False
) -> Dict[str, Dict]:
    """OCR multiple frames sequentially."""
    results = {}
    total = len(frame_paths)

    if not OCR_AVAILABLE:
        print("    [OCR] Tesseract not available, returning empty results")
        return {fp: {"text": "", "boxes": [], "word_count": 0} for fp in frame_paths}

    print(f"    [OCR] Processing {total} frames...")

    for i, fp in enumerate(frame_paths):
        results[fp] = ocr_frame(fp, with_boxes=with_boxes)
        if (i + 1) % 10 == 0 or (i + 1) == total:
            print(f"    [OCR] {i + 1}/{total} done")

    non_empty = sum(1 for v in results.values() if v["text"].strip())
    print(f"    [OCR] {non_empty}/{total} frames had readable text")
    return results


def compute_text_diff(text_before: str, text_after: str) -> Dict:
    """Compute text difference between two OCR results."""
    words_before = set(_extract_words(text_before))
    words_after = set(_extract_words(text_after))

    added = words_after - words_before
    removed = words_before - words_after
    common = words_before & words_after

    total = len(words_before | words_after)
    change_ratio = len(added | removed) / total if total > 0 else 0.0

    return {
        "added_words": sorted(added),
        "removed_words": sorted(removed),
        "common_words": sorted(common),
        "change_ratio": change_ratio
    }


def _extract_words(text: str) -> List[str]:
    """Extract meaningful words from text."""
    if not text:
        return []
    words = re.findall(r'[a-zA-Z]{3,}', text.lower())
    stopwords = {
        'the', 'and', 'for', 'that', 'this', 'with', 'from',
        'are', 'was', 'not', 'but', 'all', 'can', 'will',
        'has', 'have', 'had', 'its', 'than', 'then'
    }
    return [w for w in words if w not in stopwords]


def _clean_ocr_text(text: str) -> str:
    """Clean OCR output."""
    if not text:
        return ""
    text = re.sub(r'[|]{2,}', ' ', text)
    text = re.sub(r'[_]{3,}', ' ', text)
    text = re.sub(r'[~`]{2,}', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    lines = text.split('\n')
    lines = [l.strip() for l in lines if len(l.strip()) > 2]
    return ' '.join(lines).strip()