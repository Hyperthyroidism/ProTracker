from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def get_video_info(video_path: str) -> Dict[str, float]:
    """
    Get basic information of a video.

    Args:
        video_path: Path to input video.

    Returns:
        Dictionary containing fps, width, height, frame_count and duration.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        duration = 0.0
    else:
        duration = frame_count / fps

    cap.release()

    return {
        "fps": float(fps),
        "width": int(width),
        "height": int(height),
        "frame_count": int(frame_count),
        "duration": float(duration),
    }


def read_video_frames(
    video_path: str,
    max_frames: Optional[int] = None,
    frame_interval: int = 1,
    resize: Optional[Tuple[int, int]] = None,
) -> List[np.ndarray]:
    """
    Read frames from a video.

    Args:
        video_path: Path to input video.
        max_frames: Maximum number of frames to read. If None, read all frames.
        frame_interval: Interval for frame sampling. 1 means reading every frame.
        resize: Optional resize size in (width, height) format.

    Returns:
        List of video frames in BGR format.
    """
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if frame_interval <= 0:
        raise ValueError("frame_interval must be larger than 0.")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frames = []
    frame_id = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_id % frame_interval == 0:
            if resize is not None:
                frame = cv2.resize(frame, resize)

            frames.append(frame)
            saved_count += 1

            if max_frames is not None and saved_count >= max_frames:
                break

        frame_id += 1

    cap.release()

    return frames


def write_video_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 25.0,
    codec: str = "mp4v",
) -> None:
    """
    Write frames to a video file.

    Args:
        frames: List of frames in BGR format.
        output_path: Output video path.
        fps: Output video fps.
        codec: FourCC codec string.
    """
    if len(frames) == 0:
        raise ValueError("frames is empty. Cannot write video.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    height, width = frames[0].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    for frame in frames:
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, (width, height))

        writer.write(frame)

    writer.release()


def save_frames_as_images(
    frames: List[np.ndarray],
    output_dir: str,
    prefix: str = "frame",
    start_index: int = 0,
    ext: str = ".jpg",
) -> None:
    """
    Save video frames as image files.

    Args:
        frames: List of frames in BGR format.
        output_dir: Directory to save images.
        prefix: Image filename prefix.
        start_index: Start frame index.
        ext: Image extension, such as .jpg or .png.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, frame in enumerate(frames):
        frame_id = start_index + idx
        save_path = output_dir / f"{prefix}_{frame_id:06d}{ext}"
        cv2.imwrite(str(save_path), frame)


def read_image_sequence(
    image_dir: str,
    extensions: Optional[List[str]] = None,
) -> List[np.ndarray]:
    """
    Read an image sequence from a directory.

    Args:
        image_dir: Directory containing image frames.
        extensions: Image extensions to read.

    Returns:
        List of frames in BGR format.
    """
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if extensions is None:
        extensions = [".jpg", ".jpeg", ".png", ".bmp"]

    image_paths = []

    for ext in extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))

    image_paths = sorted(image_paths)

    frames = []

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        frames.append(frame)

    return frames


def convert_video_to_frames(
    video_path: str,
    output_dir: str,
    frame_interval: int = 1,
    prefix: str = "frame",
    ext: str = ".jpg",
) -> None:
    """
    Convert a video into image frames.

    Args:
        video_path: Input video path.
        output_dir: Directory to save extracted frames.
        frame_interval: Sampling interval.
        prefix: Saved image prefix.
        ext: Saved image extension.
    """
    frames = read_video_frames(
        video_path=video_path,
        frame_interval=frame_interval,
    )

    save_frames_as_images(
        frames=frames,
        output_dir=output_dir,
        prefix=prefix,
        ext=ext,
    )


def draw_text(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 0.6,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw text on an image.

    Args:
        image: Input image.
        text: Text to draw.
        position: Text position.
        color: Text color in BGR format.
        font_scale: Font scale.
        thickness: Text thickness.

    Returns:
        Image with text.
    """
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return image