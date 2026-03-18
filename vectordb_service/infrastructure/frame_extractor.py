import os

import cv2


def extract_scene_frames(
    video_path: str, scenes: list, output_dir: str
) -> list[dict]:
    """Extract start/mid/end frames from each detected scene."""
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frames: list[dict] = []

    for i, scene in enumerate(scenes):
        start = scene[0].get_frames()
        end = scene[1].get_frames()

        start_frame = start + 2
        end_frame = end - 2
        mid_frame = (start + end) // 2

        for label, frame_no in [
            ("start", start_frame),
            ("mid", mid_frame),
            ("end", end_frame),
        ]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()

            if ret:
                filename = f"scene_{i}_{label}.png"
                filepath = os.path.join(output_dir, filename)
                cv2.imwrite(filepath, frame)
                frames.append(
                    {
                        "scene_id": i,
                        "frame_no": frame_no,
                        "label": label,
                        "filename": filename,
                        "filepath": filepath,
                    }
                )

    cap.release()
    return frames
