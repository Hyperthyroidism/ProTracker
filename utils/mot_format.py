from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def detection_to_mot_line(
    detection: Dict[str, Any],
    frame_id: Optional[int] = None,
    track_id: Optional[int] = None,
) -> List[float]:
    """
    Convert one detection or tracking result to a MOT-format line.

    MOT format:
        frame_id, track_id, x, y, width, height, confidence, class_id, visibility

    Args:
        detection: Detection or tracking result dictionary.
        frame_id: Optional frame id. If None, use detection["frame_id"].
        track_id: Optional track id. If None, use detection["track_id"].

    Returns:
        MOT-format line.
    """
    if frame_id is None:
        frame_id = int(detection.get("frame_id", 0))

    if track_id is None:
        track_id = int(detection.get("track_id", -1))

    if "bbox_xywh" in detection:
        x, y, w, h = detection["bbox_xywh"]

    elif "bbox_xyxy" in detection:
        x1, y1, x2, y2 = detection["bbox_xyxy"]
        x = x1
        y = y1
        w = x2 - x1
        h = y2 - y1

    else:
        raise KeyError("Detection must contain bbox_xywh or bbox_xyxy.")

    confidence = float(detection.get("confidence", 1.0))
    class_id = int(detection.get("class_id", 0))
    visibility = float(detection.get("visibility", 1.0))

    mot_line = [
        float(frame_id),
        float(track_id),
        float(x),
        float(y),
        float(w),
        float(h),
        float(confidence),
        float(class_id),
        float(visibility),
    ]

    return mot_line


def detections_to_mot_lines(
    detections: List[Dict[str, Any]],
    frame_id: Optional[int] = None,
) -> List[List[float]]:
    """
    Convert a list of detections to MOT-format lines.

    Args:
        detections: Detection list.
        frame_id: Optional frame id.

    Returns:
        List of MOT-format lines.
    """
    mot_lines = []

    for detection in detections:
        track_id = detection.get("track_id", -1)

        if int(track_id) < 0:
            continue

        mot_line = detection_to_mot_line(
            detection=detection,
            frame_id=frame_id,
        )
        mot_lines.append(mot_line)

    return mot_lines


def save_mot_results(
    mot_results: List[List[float]],
    save_path: str,
) -> None:
    """
    Save MOT-format tracking results to a txt file.

    Args:
        mot_results: MOT-format result list.
        save_path: Output txt file path.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mot_results = sorted(
        mot_results,
        key=lambda x: (int(x[0]), int(x[1])),
    )

    with open(save_path, "w", encoding="utf-8") as f:
        for line in mot_results:
            line_str = ",".join(
                [
                    str(int(value)) if idx in [0, 1, 7] else f"{float(value):.6f}"
                    for idx, value in enumerate(line)
                ]
            )
            f.write(line_str + "\n")


def load_mot_results(file_path: str) -> List[List[float]]:
    """
    Load MOT-format results from a txt file.

    Args:
        file_path: MOT result txt file path.

    Returns:
        MOT-format result list.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"MOT result file not found: {file_path}")

    results = []

    with open(file_path, "r", encoding="utf-8") as f:
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


def group_mot_results_by_frame(
    mot_results: List[List[float]],
) -> Dict[int, List[List[float]]]:
    """
    Group MOT results by frame id.

    Args:
        mot_results: MOT-format result list.

    Returns:
        Dictionary mapping frame id to result lines.
    """
    grouped_results: Dict[int, List[List[float]]] = {}

    for line in mot_results:
        frame_id = int(line[0])
        grouped_results.setdefault(frame_id, []).append(line)

    return grouped_results


def group_mot_results_by_track(
    mot_results: List[List[float]],
) -> Dict[int, List[List[float]]]:
    """
    Group MOT results by track id.

    Args:
        mot_results: MOT-format result list.

    Returns:
        Dictionary mapping track id to result lines.
    """
    grouped_results: Dict[int, List[List[float]]] = {}

    for line in mot_results:
        track_id = int(line[1])
        grouped_results.setdefault(track_id, []).append(line)

    return grouped_results


def convert_mot_to_detections(
    mot_results: List[List[float]],
) -> List[Dict[str, Any]]:
    """
    Convert MOT-format lines to detection dictionaries.

    Args:
        mot_results: MOT-format result list.

    Returns:
        Detection dictionary list.
    """
    detections = []

    for line in mot_results:
        frame_id, track_id, x, y, w, h, conf, class_id, visibility = line[:9]

        detection = {
            "frame_id": int(frame_id),
            "track_id": int(track_id),
            "bbox_xywh": np.array([x, y, w, h], dtype=np.float32),
            "bbox_xyxy": np.array([x, y, x + w, y + h], dtype=np.float32),
            "confidence": float(conf),
            "class_id": int(class_id),
            "visibility": float(visibility),
            "class_name": "vessel",
        }

        detections.append(detection)

    return detections


def save_frame_results(
    frame_id: int,
    detections: List[Dict[str, Any]],
    save_path: str,
    append: bool = True,
) -> None:
    """
    Save one frame's tracking results to a MOT-format txt file.

    Args:
        frame_id: Current frame id.
        detections: Detection or tracking result list.
        save_path: Output txt path.
        append: Whether to append results to existing file.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if append else "w"

    mot_lines = detections_to_mot_lines(
        detections=detections,
        frame_id=frame_id,
    )

    with open(save_path, mode, encoding="utf-8") as f:
        for line in mot_lines:
            line_str = ",".join(
                [
                    str(int(value)) if idx in [0, 1, 7] else f"{float(value):.6f}"
                    for idx, value in enumerate(line)
                ]
            )
            f.write(line_str + "\n")


def filter_mot_results_by_confidence(
    mot_results: List[List[float]],
    confidence_threshold: float = 0.0,
) -> List[List[float]]:
    """
    Filter MOT results by confidence score.

    Args:
        mot_results: MOT-format result list.
        confidence_threshold: Confidence threshold.

    Returns:
        Filtered MOT result list.
    """
    filtered_results = []

    for line in mot_results:
        confidence = float(line[6])

        if confidence >= confidence_threshold:
            filtered_results.append(line)

    return filtered_results


def filter_mot_results_by_track_length(
    mot_results: List[List[float]],
    min_track_length: int = 3,
) -> List[List[float]]:
    """
    Filter MOT results by track length.

    Args:
        mot_results: MOT-format result list.
        min_track_length: Minimum valid track length.

    Returns:
        Filtered MOT result list.
    """
    grouped_by_track = group_mot_results_by_track(mot_results)

    valid_track_ids = {
        track_id
        for track_id, lines in grouped_by_track.items()
        if len(lines) >= min_track_length
    }

    filtered_results = [
        line
        for line in mot_results
        if int(line[1]) in valid_track_ids
    ]

    return filtered_results


def summarize_mot_results(
    mot_results: List[List[float]],
) -> Dict[str, int]:
    """
    Summarize MOT-format tracking results.

    Args:
        mot_results: MOT-format result list.

    Returns:
        Summary dictionary.
    """
    if len(mot_results) == 0:
        return {
            "num_frames": 0,
            "num_tracks": 0,
            "num_boxes": 0,
        }

    frame_ids = [int(line[0]) for line in mot_results]
    track_ids = [int(line[1]) for line in mot_results]

    summary = {
        "num_frames": len(set(frame_ids)),
        "num_tracks": len(set(track_ids)),
        "num_boxes": len(mot_results),
    }

    return summary