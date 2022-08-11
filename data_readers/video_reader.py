"""This data reader reads a single video.
"""
import dataclasses
import os

import cv2
import numpy as np
from torch.utils.data import Dataset


@dataclasses.dataclass
class VideoDetails:
    frame_count: int = 0
    fps: int = 0
    width: int = 0
    height: int = 0


def get_video_details(video_path: str) -> VideoDetails:
    """Find details about a video.

    Args:
        video_path: Path of the video.

    Returns:
        Object containing video metadata.
    """
    cap = cv2.VideoCapture(video_path)
    video_details = VideoDetails(
        frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        fps=int(cap.get(cv2.CAP_PROP_FPS)),
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    )
    cap.release()
    return video_details


def get_video_frame(video_path: str, frame_num: int, factor: int = 1,
                    attempts: int = 0) -> np.ndarray:
    """Retrieves a video frame from the video.

    Args:
        video_path: Path to the video.
        frame_num: 0-indexed frame to retrieve.
        factor: Resize factor, defaults to 1.
        attempts: Number of attempts to read the frame.

    Returns:
        The loaded video frame as a numpy array.
    """
    capture = cv2.VideoCapture(video_path)
    ret = capture.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    if not ret:
        print("Capture set failed", ret)
    ret, frame = capture.read()
    current_attempt = 0
    while not ret and current_attempt < attempts:
        ret, frame = capture.read()
        current_attempt += 1
    if not ret:
        print("Error loading frame", ret)
        capture.release()
        exit(0)
    capture.release()
    frame = frame[:, :, ::-1] / 255
    if factor != 1:
        frame = cv2.resize(frame, (0, 0), fx=1 / factor, fy=1 / factor)
    return frame


class VideoReader(Dataset):
    """Video Reader"""

    def __init__(self, video_path: str, max_frames: int = -1,
                 start_frame: int = 0):
        """Create a dataset reader from your CARLA dataset

        Args:
          video_path: Path to the video.
          max_frames: Max size of the dataset,
          start_frame: Initial starting frame.
        """
        if not os.path.isfile(video_path):
            raise ValueError(f"File not found: {video_path}")
        self.video_path = video_path
        self.start_frame = start_frame
        self.video_details = get_video_details(video_path)
        if max_frames >= 0:
            self.video_details.frame_count = min(max_frames + self.start_frame,
                                                 self.video_details.frame_count)

    def __len__(self):
        """Get length of the data."""
        return self.video_details.frame_count - self.start_frame

    def __getitem__(self, idx):
        """Gets a single item from the dataset.

        Args:
          idx: Index of the item in the dataset.

        Returns:
          Dictionary including the video frame.
        """
        frame = get_video_frame(self.video_path, idx + self.start_frame)
        return {
            "frame": frame.astype(np.float32),
            "t": idx
        }
