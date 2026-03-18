# video/scene_detector.py

"""
SSIM-based scene change detection with dynamic threshold and adaptive sampling.
Used by the silent video pipeline to find key visual transitions.
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Optional
from collections import deque

from core.config import config


def compute_ssim_gray(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute SSIM between two grayscale frames."""
    if frame1 is None or frame2 is None:
        return 0.0

    if frame1.shape != frame2.shape:
        frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    frame1 = frame1.astype(np.float64)
    frame2 = frame2.astype(np.float64)

    mu1 = cv2.GaussianBlur(frame1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(frame2, (11, 11), 1.5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(frame1 ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(frame2 ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(frame1 * frame2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return float(ssim_map.mean())


def compute_histogram_diff(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Compute histogram correlation between two frames."""
    if frame1 is None or frame2 is None:
        return 0.0

    hist1 = cv2.calcHist([frame1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([frame2], [0], None, [256], [0, 256])

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)

    correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    return max(0.0, correlation)


def get_video_info(video_path: str) -> Dict:
    """Get video metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 0, "frames": 0, "duration": 0, "width": 0, "height": 0}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration = total_frames / fps if fps > 0 else 0

    return {
        "fps": fps,
        "frames": total_frames,
        "duration": duration,
        "width": width,
        "height": height
    }


def estimate_content_type(first_frame: np.ndarray) -> float:
    """
    Estimate optimal SSIM threshold based on frame content.
    Returns a threshold value (lower for UI-heavy, higher for presentations).
    """
    gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.count_nonzero(edges) / (gray.shape[0] * gray.shape[1])

    hsv = cv2.cvtColor(first_frame, cv2.COLOR_BGR2HSV)
    color_variance = np.var(hsv[:, :, 1])

    if edge_density > 0.1 and color_variance > 1000:
        return 0.75
    elif edge_density < 0.05 and color_variance < 500:
        return 0.92
    else:
        return config.frame.ssim_threshold


def detect_scene_changes(
    video_path: str,
    ssim_threshold: Optional[float] = None,
    min_gap_seconds: Optional[float] = None,
    sample_interval: Optional[float] = None,
    adaptive: bool = True
) -> List[Dict]:
    """
    Detect scene changes with dynamic threshold and adaptive sampling rate.
    """
    if ssim_threshold is None:
        ssim_threshold = config.frame.ssim_threshold
    min_gap = min_gap_seconds or config.frame.min_frame_gap_seconds
    base_interval = sample_interval or config.frame.sample_interval_seconds

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    [SceneDetect] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"    [SceneDetect] Video: {duration:.0f}s, {fps:.1f}fps, {total_frames} frames")

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return []
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    dynamic_threshold = estimate_content_type(first_frame)
    if ssim_threshold == config.frame.ssim_threshold:
        ssim_threshold = dynamic_threshold
    print(f"    [SceneDetect] Using SSIM threshold: {ssim_threshold:.2f}")

    current_interval = base_interval
    frame_skip = max(1, int(fps * current_interval))

    scene_changes = [{
        "timestamp": 0.0,
        "frame_index": 0,
        "ssim_score": 0.0,
        "frame": first_frame.copy()
    }]
    prev_gray = first_gray
    last_scene_time = 0.0

    recent_ssim = deque(maxlen=10)
    frames_checked = 0

    frame_idx = frame_skip
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timestamp = frame_idx / fps

        ssim = compute_ssim_gray(prev_gray, current_gray)
        frames_checked += 1
        recent_ssim.append(ssim)

        if adaptive and len(recent_ssim) == recent_ssim.maxlen:
            avg_ssim = sum(recent_ssim) / len(recent_ssim)
            if avg_ssim < ssim_threshold * 0.9:
                new_interval = max(0.2, current_interval * 0.8)
            elif avg_ssim > ssim_threshold * 1.1:
                new_interval = min(2.0, current_interval * 1.2)
            else:
                new_interval = current_interval

            if abs(new_interval - current_interval) > 0.1:
                current_interval = new_interval
                frame_skip = max(1, int(fps * current_interval))

        if ssim < ssim_threshold and (timestamp - last_scene_time) >= min_gap:
            hist_diff = compute_histogram_diff(prev_gray, current_gray)
            if hist_diff < 0.95:
                scene_changes.append({
                    "timestamp": timestamp,
                    "frame_index": frame_idx,
                    "ssim_score": ssim,
                    "frame": frame.copy()
                })
                last_scene_time = timestamp

        prev_gray = current_gray
        frame_idx += frame_skip

        if frames_checked % 100 == 0:
            print(f"    [SceneDetect] Checked {frames_checked} samples, "
                  f"found {len(scene_changes)} scenes...")

    cap.release()
    print(f"    [SceneDetect] Checked {frames_checked} samples, "
          f"found {len(scene_changes)} scenes")
    return scene_changes