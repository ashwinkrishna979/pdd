# video/change_detector.py

"""
Change detection between consecutive frames.
Pixel diff + OCR text diff for classifying transition types.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional

from video.ocr_engine import compute_text_diff


def compute_pixel_diff(
    frame1_path: str,
    frame2_path: str,
    threshold: int = 30
) -> Dict:
    """Compute pixel-level difference between two frames."""
    img1 = cv2.imread(frame1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(frame2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return {"change_magnitude": 0.0, "changed_regions": [], "diff_mask": None}

    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    diff = cv2.absdiff(img1, img2)
    _, diff_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        diff_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    min_area = (img1.shape[0] * img1.shape[1]) * 0.005
    changed_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            changed_regions.append({
                "x": x, "y": y, "w": w, "h": h, "area": area
            })

    changed_regions.sort(key=lambda r: r["area"], reverse=True)

    total_pixels = img1.shape[0] * img1.shape[1]
    changed_pixels = np.count_nonzero(diff_mask)
    change_magnitude = changed_pixels / total_pixels

    return {
        "change_magnitude": change_magnitude,
        "changed_regions": changed_regions[:10],
        "diff_mask": diff_mask
    }


def find_primary_change_region(
    changed_regions: List[Dict],
    frame_height: int,
    frame_width: int
) -> Optional[Dict]:
    """Find the most significant change region."""
    if not changed_regions:
        return None

    largest = changed_regions[0]
    frame_area = frame_height * frame_width

    if largest["area"] > frame_area * 0.6:
        return None

    return largest


def detect_changes_between_frames(
    key_frames: List[Dict],
    ocr_results: Dict[str, Dict]
) -> List[Dict]:
    """Detect changes between all consecutive frame pairs."""
    if len(key_frames) < 2:
        return []

    changes = []
    total_pairs = len(key_frames) - 1
    print(f"    [ChangeDetect] Analyzing {total_pairs} frame pairs...")

    for i in range(total_pairs):
        before = key_frames[i]
        after = key_frames[i + 1]

        before_path = before["path"]
        after_path = after["path"]

        pixel_result = compute_pixel_diff(before_path, after_path)

        before_text = ocr_results.get(before_path, {}).get("text", "")
        after_text = ocr_results.get(after_path, {}).get("text", "")
        text_diff = compute_text_diff(before_text, after_text)

        before_img = cv2.imread(before_path)
        frame_h = before_img.shape[0] if before_img is not None else 1080
        frame_w = before_img.shape[1] if before_img is not None else 1920

        primary_region = find_primary_change_region(
            pixel_result["changed_regions"], frame_h, frame_w
        )

        change_type = _classify_change(
            pixel_result["change_magnitude"],
            text_diff["change_ratio"],
            primary_region
        )

        changes.append({
            "pair_index": i,
            "frame_before_path": before_path,
            "frame_after_path": after_path,
            "timestamp_before": before.get("timestamp", 0),
            "timestamp_after": after.get("timestamp", 0),
            "pixel_change_magnitude": pixel_result["change_magnitude"],
            "changed_regions": pixel_result["changed_regions"],
            "primary_region": primary_region,
            "text_diff": text_diff,
            "change_type": change_type,
            "frame_height": frame_h,
            "frame_width": frame_w
        })

    type_counts = {}
    for c in changes:
        ct = c["change_type"]
        type_counts[ct] = type_counts.get(ct, 0) + 1
    print(f"    [ChangeDetect] Change types: {type_counts}")

    return changes


def _classify_change(
    pixel_magnitude: float,
    text_change_ratio: float,
    primary_region: Optional[Dict]
) -> str:
    """Classify the type of change between frames."""
    if pixel_magnitude > 0.5:
        return "page_transition"
    if pixel_magnitude > 0.15 and primary_region:
        region_ratio = primary_region["area"] / (1920 * 1080)
        if region_ratio < 0.3:
            return "modal_popup"
        return "partial_update"
    if pixel_magnitude > 0.03 and primary_region:
        if text_change_ratio < 0.3:
            return "form_input"
        return "menu_interaction"
    if pixel_magnitude > 0.01:
        return "minor_change"
    return "minor_change"