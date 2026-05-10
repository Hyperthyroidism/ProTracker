import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_mot_file(file_path: str) -> List[List[float]]:
    """
    Load a MOT-format txt file.

    MOT format:
        frame_id, track_id, x, y, width, height, confidence, class_id, visibility

    Args:
        file_path: MOT-format txt path.

    Returns:
        List of MOT-format records.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"MOT file not found: {file_path}")

    records = []

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

            records.append(values[:9])

    return records


def group_by_frame(records: List[List[float]]) -> Dict[int, List[List[float]]]:
    """
    Group MOT records by frame id.

    Args:
        records: MOT-format records.

    Returns:
        Dictionary mapping frame id to records.
    """
    grouped = {}

    for record in records:
        frame_id = int(record[0])
        grouped.setdefault(frame_id, []).append(record)

    return grouped


def xywh_to_xyxy(record: List[float]) -> np.ndarray:
    """
    Convert MOT record box from xywh to xyxy.

    Args:
        record: MOT-format record.

    Returns:
        Bounding box in xyxy format.
    """
    x, y, w, h = record[2:6]

    return np.array(
        [x, y, x + w, y + h],
        dtype=np.float32,
    )


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Calculate IoU between two boxes.

    Args:
        box_a: First box in xyxy format.
        box_b: Second box in xyxy format.

    Returns:
        IoU value.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return float(inter_area / union)


def match_frame(
    gt_records: List[List[float]],
    pred_records: List[List[float]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int, int]:
    """
    Match predictions and ground truth objects in one frame.

    Args:
        gt_records: Ground truth records of one frame.
        pred_records: Prediction records of one frame.
        iou_threshold: IoU threshold for matching.

    Returns:
        Tuple of:
            matches, false_positives, false_negatives, id_switches
    """
    if len(gt_records) == 0 and len(pred_records) == 0:
        return 0, 0, 0, 0

    if len(gt_records) == 0:
        return 0, len(pred_records), 0, 0

    if len(pred_records) == 0:
        return 0, 0, len(gt_records), 0

    candidates = []

    for gt_idx, gt in enumerate(gt_records):
        gt_box = xywh_to_xyxy(gt)
        gt_id = int(gt[1])

        for pred_idx, pred in enumerate(pred_records):
            pred_box = xywh_to_xyxy(pred)
            pred_id = int(pred[1])

            iou = box_iou(gt_box, pred_box)

            if iou >= iou_threshold:
                candidates.append((iou, gt_idx, pred_idx, gt_id, pred_id))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = 0
    id_switches = 0

    for iou, gt_idx, pred_idx, gt_id, pred_id in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches += 1

        # In this lightweight version, IDSW is counted when the matched
        # prediction ID is different from the ground truth ID in the same frame.
        if gt_id != pred_id:
            id_switches += 1

    false_positives = len(pred_records) - len(matched_pred)
    false_negatives = len(gt_records) - len(matched_gt)

    return matches, false_positives, false_negatives, id_switches


def evaluate_one_sequence(
    gt_file: str,
    pred_file: str,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate MOTA for one sequence.

    Args:
        gt_file: Ground truth txt file.
        pred_file: Prediction txt file.
        iou_threshold: IoU threshold.

    Returns:
        Metric dictionary.
    """
    gt_records = load_mot_file(gt_file)
    pred_records = load_mot_file(pred_file)

    gt_by_frame = group_by_frame(gt_records)
    pred_by_frame = group_by_frame(pred_records)

    all_frame_ids = sorted(
        set(gt_by_frame.keys()) | set(pred_by_frame.keys())
    )

    total_gt = len(gt_records)
    total_matches = 0
    total_fp = 0
    total_fn = 0
    total_idsw = 0

    for frame_id in all_frame_ids:
        gt_frame = gt_by_frame.get(frame_id, [])
        pred_frame = pred_by_frame.get(frame_id, [])

        matches, fp, fn, idsw = match_frame(
            gt_records=gt_frame,
            pred_records=pred_frame,
            iou_threshold=iou_threshold,
        )

        total_matches += matches
        total_fp += fp
        total_fn += fn
        total_idsw += idsw

    if total_gt == 0:
        mota = 0.0
    else:
        mota = 1.0 - (total_fn + total_fp + total_idsw) / total_gt

    result = {
        "MOTA": float(mota),
        "GT": float(total_gt),
        "Matches": float(total_matches),
        "FP": float(total_fp),
        "FN": float(total_fn),
        "IDSW": float(total_idsw),
    }

    return result


def find_sequence_pairs(gt_dir: str, pred_dir: str) -> List[Tuple[Path, Path]]:
    """
    Find sequence pairs with the same file names.

    Args:
        gt_dir: Ground truth directory.
        pred_dir: Prediction directory.

    Returns:
        List of gt-pred file pairs.
    """
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground truth directory not found: {gt_dir}")

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")

    pairs = []

    for gt_file in sorted(gt_dir.glob("*.txt")):
        pred_file = pred_dir / gt_file.name

        if pred_file.exists():
            pairs.append((gt_file, pred_file))
        else:
            print(f"Prediction file missing for sequence: {gt_file.name}")

    return pairs


def evaluate_mota(
    gt_dir: str,
    pred_dir: str,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate MOTA on multiple sequences.

    Args:
        gt_dir: Ground truth directory.
        pred_dir: Prediction directory.
        iou_threshold: IoU threshold.

    Returns:
        Overall metric dictionary.
    """
    pairs = find_sequence_pairs(gt_dir, pred_dir)

    if len(pairs) == 0:
        print("No matched sequence files found.")
        return {
            "MOTA": 0.0,
            "GT": 0.0,
            "Matches": 0.0,
            "FP": 0.0,
            "FN": 0.0,
            "IDSW": 0.0,
        }

    total_gt = 0.0
    total_matches = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_idsw = 0.0

    print("\nMOTA Evaluation")
    print("-" * 70)

    for gt_file, pred_file in pairs:
        result = evaluate_one_sequence(
            gt_file=str(gt_file),
            pred_file=str(pred_file),
            iou_threshold=iou_threshold,
        )

        total_gt += result["GT"]
        total_matches += result["Matches"]
        total_fp += result["FP"]
        total_fn += result["FN"]
        total_idsw += result["IDSW"]

        print(
            f"{gt_file.stem:25s} "
            f"MOTA: {result['MOTA']:.4f} | "
            f"GT: {int(result['GT'])} | "
            f"FP: {int(result['FP'])} | "
            f"FN: {int(result['FN'])} | "
            f"IDSW: {int(result['IDSW'])}"
        )

    if total_gt == 0:
        overall_mota = 0.0
    else:
        overall_mota = 1.0 - (total_fn + total_fp + total_idsw) / total_gt

    overall = {
        "MOTA": float(overall_mota),
        "GT": float(total_gt),
        "Matches": float(total_matches),
        "FP": float(total_fp),
        "FN": float(total_fn),
        "IDSW": float(total_idsw),
    }

    print("-" * 70)
    print(
        f"{'Overall':25s} "
        f"MOTA: {overall['MOTA']:.4f} | "
        f"GT: {int(overall['GT'])} | "
        f"FP: {int(overall['FP'])} | "
        f"FN: {int(overall['FN'])} | "
        f"IDSW: {int(overall['IDSW'])}"
    )

    return overall


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate MOTA for ProTracker results."
    )

    parser.add_argument(
        "--gt",
        type=str,
        required=True,
        help="Ground truth directory."
    )

    parser.add_argument(
        "--pred",
        type=str,
        required=True,
        help="Prediction result directory."
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for matching."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    evaluate_mota(
        gt_dir=args.gt,
        pred_dir=args.pred,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()