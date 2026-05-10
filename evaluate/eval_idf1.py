import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_mot_file(file_path: str) -> List[List[float]]:
    """
    Load MOT-format tracking results.

    MOT format:
        frame_id, track_id, x, y, width, height, confidence, class_id, visibility

    Args:
        file_path: MOT-format txt file.

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
    Convert MOT box from xywh to xyxy.

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
) -> List[Tuple[int, int]]:
    """
    Match predictions and ground truth objects in one frame.

    Args:
        gt_records: Ground truth records of one frame.
        pred_records: Prediction records of one frame.
        iou_threshold: IoU threshold for matching.

    Returns:
        Matched pairs:
            [(gt_id, pred_id), ...]
    """
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
    matches = []

    for iou, gt_idx, pred_idx, gt_id, pred_id in candidates:
        if gt_idx in matched_gt or pred_idx in matched_pred:
            continue

        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        matches.append((gt_id, pred_id))

    return matches


def build_identity_match_table(
    gt_records: List[List[float]],
    pred_records: List[List[float]],
    iou_threshold: float = 0.5,
) -> Dict[Tuple[int, int], int]:
    """
    Build a table of identity matches between ground truth IDs and predicted IDs.

    Args:
        gt_records: Ground truth records.
        pred_records: Prediction records.
        iou_threshold: IoU threshold.

    Returns:
        Dictionary mapping (gt_id, pred_id) to matched count.
    """
    gt_by_frame = group_by_frame(gt_records)
    pred_by_frame = group_by_frame(pred_records)

    all_frame_ids = sorted(
        set(gt_by_frame.keys()) | set(pred_by_frame.keys())
    )

    match_table: Dict[Tuple[int, int], int] = {}

    for frame_id in all_frame_ids:
        gt_frame = gt_by_frame.get(frame_id, [])
        pred_frame = pred_by_frame.get(frame_id, [])

        matches = match_frame(
            gt_records=gt_frame,
            pred_records=pred_frame,
            iou_threshold=iou_threshold,
        )

        for gt_id, pred_id in matches:
            key = (int(gt_id), int(pred_id))
            match_table[key] = match_table.get(key, 0) + 1

    return match_table


def greedy_identity_assignment(
    match_table: Dict[Tuple[int, int], int],
) -> Dict[int, int]:
    """
    Assign predicted identities to ground truth identities using greedy matching.

    Args:
        match_table: Dictionary mapping (gt_id, pred_id) to matched count.

    Returns:
        Dictionary mapping gt_id to pred_id.
    """
    candidates = []

    for (gt_id, pred_id), count in match_table.items():
        candidates.append((count, gt_id, pred_id))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    assigned_gt = set()
    assigned_pred = set()
    assignment = {}

    for count, gt_id, pred_id in candidates:
        if gt_id in assigned_gt or pred_id in assigned_pred:
            continue

        assigned_gt.add(gt_id)
        assigned_pred.add(pred_id)
        assignment[int(gt_id)] = int(pred_id)

    return assignment


def calculate_id_metrics(
    gt_records: List[List[float]],
    pred_records: List[List[float]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calculate simplified IDF1, IDP, and IDR.

    Args:
        gt_records: Ground truth records.
        pred_records: Prediction records.
        iou_threshold: IoU threshold.

    Returns:
        Identity metric dictionary.
    """
    match_table = build_identity_match_table(
        gt_records=gt_records,
        pred_records=pred_records,
        iou_threshold=iou_threshold,
    )

    assignment = greedy_identity_assignment(match_table)

    idtp = 0

    for gt_id, pred_id in assignment.items():
        idtp += match_table.get((gt_id, pred_id), 0)

    total_gt = len(gt_records)
    total_pred = len(pred_records)

    idfn = total_gt - idtp
    idfp = total_pred - idtp

    if idtp + idfp == 0:
        idp = 0.0
    else:
        idp = idtp / (idtp + idfp)

    if idtp + idfn == 0:
        idr = 0.0
    else:
        idr = idtp / (idtp + idfn)

    if idp + idr == 0:
        idf1 = 0.0
    else:
        idf1 = 2.0 * idp * idr / (idp + idr)

    result = {
        "IDF1": float(idf1),
        "IDP": float(idp),
        "IDR": float(idr),
        "IDTP": float(idtp),
        "IDFP": float(idfp),
        "IDFN": float(idfn),
        "GT": float(total_gt),
        "Pred": float(total_pred),
    }

    return result


def evaluate_one_sequence(
    gt_file: str,
    pred_file: str,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate IDF1 for one sequence.

    Args:
        gt_file: Ground truth txt file.
        pred_file: Prediction txt file.
        iou_threshold: IoU threshold.

    Returns:
        ID metric dictionary.
    """
    gt_records = load_mot_file(gt_file)
    pred_records = load_mot_file(pred_file)

    result = calculate_id_metrics(
        gt_records=gt_records,
        pred_records=pred_records,
        iou_threshold=iou_threshold,
    )

    return result


def find_sequence_pairs(gt_dir: str, pred_dir: str) -> List[Tuple[Path, Path]]:
    """
    Find ground truth and prediction sequence pairs.

    Args:
        gt_dir: Ground truth directory.
        pred_dir: Prediction directory.

    Returns:
        List of matched gt-pred file pairs.
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


def evaluate_idf1(
    gt_dir: str,
    pred_dir: str,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate IDF1 on multiple sequences.

    Args:
        gt_dir: Ground truth directory.
        pred_dir: Prediction directory.
        iou_threshold: IoU threshold.

    Returns:
        Overall ID metric dictionary.
    """
    pairs = find_sequence_pairs(gt_dir, pred_dir)

    if len(pairs) == 0:
        print("No matched sequence files found.")
        return {
            "IDF1": 0.0,
            "IDP": 0.0,
            "IDR": 0.0,
            "IDTP": 0.0,
            "IDFP": 0.0,
            "IDFN": 0.0,
        }

    total_idtp = 0.0
    total_idfp = 0.0
    total_idfn = 0.0
    total_gt = 0.0
    total_pred = 0.0

    print("\nIDF1 Evaluation")
    print("-" * 80)

    for gt_file, pred_file in pairs:
        result = evaluate_one_sequence(
            gt_file=str(gt_file),
            pred_file=str(pred_file),
            iou_threshold=iou_threshold,
        )

        total_idtp += result["IDTP"]
        total_idfp += result["IDFP"]
        total_idfn += result["IDFN"]
        total_gt += result["GT"]
        total_pred += result["Pred"]

        print(
            f"{gt_file.stem:25s} "
            f"IDF1: {result['IDF1']:.4f} | "
            f"IDP: {result['IDP']:.4f} | "
            f"IDR: {result['IDR']:.4f} | "
            f"IDTP: {int(result['IDTP'])} | "
            f"IDFP: {int(result['IDFP'])} | "
            f"IDFN: {int(result['IDFN'])}"
        )

    if total_idtp + total_idfp == 0:
        overall_idp = 0.0
    else:
        overall_idp = total_idtp / (total_idtp + total_idfp)

    if total_idtp + total_idfn == 0:
        overall_idr = 0.0
    else:
        overall_idr = total_idtp / (total_idtp + total_idfn)

    if overall_idp + overall_idr == 0:
        overall_idf1 = 0.0
    else:
        overall_idf1 = 2.0 * overall_idp * overall_idr / (overall_idp + overall_idr)

    overall = {
        "IDF1": float(overall_idf1),
        "IDP": float(overall_idp),
        "IDR": float(overall_idr),
        "IDTP": float(total_idtp),
        "IDFP": float(total_idfp),
        "IDFN": float(total_idfn),
        "GT": float(total_gt),
        "Pred": float(total_pred),
    }

    print("-" * 80)
    print(
        f"{'Overall':25s} "
        f"IDF1: {overall['IDF1']:.4f} | "
        f"IDP: {overall['IDP']:.4f} | "
        f"IDR: {overall['IDR']:.4f} | "
        f"IDTP: {int(overall['IDTP'])} | "
        f"IDFP: {int(overall['IDFP'])} | "
        f"IDFN: {int(overall['IDFN'])}"
    )

    return overall


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate IDF1 for ProTracker results."
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

    evaluate_idf1(
        gt_dir=args.gt,
        pred_dir=args.pred,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()