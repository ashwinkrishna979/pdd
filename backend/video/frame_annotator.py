# video/frame_annotator.py

"""
Frame annotation — draws red boxes, arrows, and labels on screenshots.
Used by the video pipeline to highlight detected actions.
"""

import os
import cv2
import numpy as np
from typing import Optional, Dict

from core.config import config


def annotate_frame(
    frame_path: str,
    output_dir: str,
    step_number: int,
    change_region: Optional[Dict] = None,
    action_label: str = "",
    enabled: bool = None
) -> str:
    """Annotate a frame with action highlight."""
    enabled = enabled if enabled is not None else config.annotation.enabled

    if not enabled:
        return frame_path

    if not os.path.exists(frame_path):
        return frame_path

    try:
        frame = cv2.imread(frame_path)
        if frame is None:
            return frame_path

        annotated = frame.copy()
        os.makedirs(output_dir, exist_ok=True)

        if change_region:
            annotated = _draw_region_highlight(
                annotated, change_region, step_number, action_label
            )
        else:
            annotated = _draw_step_label(annotated, step_number, action_label)

        filename = f"annotated_step_{step_number:03d}.jpg"
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, annotated)
        return output_path

    except Exception as e:
        print(f"    [Annotator] Error on step {step_number}: {e}, using raw frame")
        return frame_path


def _draw_region_highlight(
    frame: np.ndarray,
    region: Dict,
    step_number: int,
    label: str
) -> np.ndarray:
    """Draw red rectangle around the change region with label."""
    ann = config.annotation

    x = region.get("x", 0)
    y = region.get("y", 0)
    w = region.get("w", 100)
    h = region.get("h", 50)

    cv2.rectangle(frame, (x, y), (x + w, y + h), ann.box_color, ann.box_thickness)

    circle_x = x - 25 if x > 40 else x + w + 25
    circle_y = y + h // 2
    circle_y = max(25, min(circle_y, frame.shape[0] - 25))
    circle_x = max(25, min(circle_x, frame.shape[1] - 25))

    cv2.circle(frame, (circle_x, circle_y), 18, ann.box_color, -1)
    cv2.putText(
        frame, str(step_number),
        (circle_x - 8, circle_y + 6),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
        ann.label_text_color, 2
    )

    arrow_end_x = x if circle_x < x else x + w
    arrow_end_y = y + h // 2
    cv2.arrowedLine(
        frame,
        (circle_x, circle_y),
        (arrow_end_x, arrow_end_y),
        ann.arrow_color, 2,
        tipLength=0.15
    )

    if label:
        label_short = label[:60] + "..." if len(label) > 60 else label
        label_y = y + h + 25
        if label_y > frame.shape[0] - 10:
            label_y = y - 10

        (tw, th), _ = cv2.getTextSize(
            label_short, cv2.FONT_HERSHEY_SIMPLEX,
            ann.font_scale * 0.6, 1
        )
        cv2.rectangle(
            frame,
            (x, label_y - th - 5),
            (x + tw + 10, label_y + 5),
            ann.label_bg_color, -1
        )
        cv2.putText(
            frame, label_short,
            (x + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            ann.font_scale * 0.6,
            ann.label_text_color, 1
        )

    return frame


def _draw_step_label(
    frame: np.ndarray,
    step_number: int,
    label: str
) -> np.ndarray:
    """Draw step label in top-left corner when no specific region is known."""
    ann = config.annotation

    label_text = f"Step {step_number}"
    if label:
        label_text += f": {label[:50]}"

    bar_height = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(
        frame, label_text,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        ann.font_scale,
        ann.label_text_color,
        ann.font_thickness
    )

    return frame