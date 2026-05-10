import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


def load_mot_results(result_path: str) -> List[List[float]]:
    """
    Load MOT-format tracking results.

    MOT format:
        frame_id, track_id, x, y, width, height, confidence, class_id, visibility

    Args:
        result_path: Path to MOT-format result txt file.

    Returns:
        List of MOT-format result lines.
    """
    result_path = Path(result_path)

    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    results = []

    with open(result_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 6:
                continue

            values = [float(x) for x in parts]

            while len(values) < 9:
                values.append(1.0)

            results.append(values[:9])

    return results


def group_results_by_frame(
    results: List[List[float]],
) -> Dict[int, List[List[float]]]:
    """
    Group MOT results by frame id.

    Args:
        results: MOT-format result list.

    Returns:
        Dictionary mapping frame id to result lines.
    """
    grouped_results: Dict[int, List[List[float]]] = {}

    for line in results:
        frame_id = int(line[0])
        grouped_results.setdefault(frame_id, []).append(line)

    return grouped_results


def get_color(track_id: int) -> Tuple[int, int, int]:
    """
    Generate a deterministic color according to track id.

    Args:
        track_id: Track identity.

    Returns:
        BGR color tuple.
    """
    np.random.seed(track_id)
    color = np.random.randint(0, 255, size=3).tolist()

    return int(color[0]), int(color[1]), int(color[2])


def draw_tracking_results(
    frame,
    frame_results: List[List[float]],
    show_confidence: bool = False,
    line_width: int = 2,
):
    """
    Draw tracking results on one frame.

    Args:
        frame: Input video frame.
        frame_results: Tracking results of current frame.
        show_confidence: Whether to show confidence score.
        line_width: Bounding box line width.

    Returns:
        Visualized frame.
    """
    vis_frame = frame.copy()

    for result in frame_results:
        _, track_id, x, y, w, h, confidence, class_id, visibility = result

        track_id = int(track_id)
        class_id = int(class_id)

        x1 = int(x)
        y1 = int(y)
        x2 = int(x + w)
        y2 = int(y + h)

        color = get_color(track_id)

        cv2.rectangle(
            vis_frame,
            (x1, y1),
            (x2, y2),
            color,
            line_width,
        )

        if show_confidence:
            text = f"ID {track_id} | {confidence:.2f}"
        else:
            text = f"ID {track_id}"

        text_x = x1
        text_y = max(0, y1 - 5)

        cv2.putText(
            vis_frame,
            text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return vis_frame


def visualize_video(
    video_path: str,
    result_path: str,
    output_path: str,
    show_confidence: bool = False,
    line_width: int = 2,
) -> None:
    """
    Visualize tracking results on a video.

    Args:
        video_path: Input video path.
        result_path: MOT-format result txt path.
        output_path: Output video path.
        show_confidence: Whether to show confidence score.
        line_width: Bounding box line width.
    """
    video_path = Path(video_path)
    result_path = Path(result_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if not result_path.exists():
        raise FileNotFoundError(f"Result file not found: {result_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = load_mot_results(str(result_path))
    results_by_frame = group_results_by_frame(results)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create video writer: {output_path}")

    frame_idx = 1

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_results = results_by_frame.get(frame_idx, [])

        vis_frame = draw_tracking_results(
            frame=frame,
            frame_results=frame_results,
            show_confidence=show_confidence,
            line_width=line_width,
        )

        writer.write(vis_frame)

        if frame_idx % 50 == 0:
            print(f"Processed frame {frame_idx}")

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"Visualization saved to: {output_path}")


def visualize_image_sequence(
    image_dir: str,
    result_path: str,
    output_dir: str,
    show_confidence: bool = False,
    line_width: int = 2,
) -> None:
    """
    Visualize tracking results on an image sequence.

    Args:
        image_dir: Directory containing image frames.
        result_path: MOT-format result txt path.
        output_dir: Directory to save visualized images.
        show_confidence: Whether to show confidence score.
        line_width: Bounding box line width.
    """
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = load_mot_results(result_path)
    results_by_frame = group_results_by_frame(results)

    image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
    image_paths = []

    for ext in image_extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise FileNotFoundError(f"No image frames found in: {image_dir}")

    for frame_idx, image_path in enumerate(image_paths, start=1):
        frame = cv2.imread(str(image_path))

        if frame is None:
            continue

        frame_results = results_by_frame.get(frame_idx, [])

        vis_frame = draw_tracking_results(
            frame=frame,
            frame_results=frame_results,
            show_confidence=show_confidence,
            line_width=line_width,
        )

        save_path = output_dir / image_path.name
        cv2.imwrite(str(save_path), vis_frame)

        if frame_idx % 50 == 0:
            print(f"Processed image frame {frame_idx}")

    print(f"Visualized images saved to: {output_dir}")


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Visualize ProTracker MOT-format tracking results."
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Input video path."
    )

    parser.add_argument(
        "--image-dir",
        type=str,
        default=None,
        help="Input image sequence directory."
    )

    parser.add_argument(
        "--result",
        type=str,
        required=True,
        help="MOT-format result txt file."
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output video path or output image directory."
    )

    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Show confidence score on visualization."
    )

    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding box line width."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.video is not None:
        visualize_video(
            video_path=args.video,
            result_path=args.result,
            output_path=args.output,
            show_confidence=args.show_confidence,
            line_width=args.line_width,
        )

    elif args.image_dir is not None:
        visualize_image_sequence(
            image_dir=args.image_dir,
            result_path=args.result,
            output_dir=args.output,
            show_confidence=args.show_confidence,
            line_width=args.line_width,
        )

    else:
        raise ValueError("Please provide either --video or --image-dir.")


if __name__ == "__main__":
    main()