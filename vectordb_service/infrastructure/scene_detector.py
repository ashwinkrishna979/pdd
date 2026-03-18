from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def detect_scenes(video_path: str, threshold: float = 9.0) -> list:
    """Detect scene boundaries in a video using content-based detection."""
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video)
    return scene_manager.get_scene_list()
