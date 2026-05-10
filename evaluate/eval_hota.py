import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_mot_file(file_path: str) -> List[List[float]]:
    """
    Load MOT-format tracking results.

    MOT format:
        frame_id, track_id, x, y, width, height, confidence, class_id, visibility
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
    """
    grouped = {}

    for record in records:
        frame_id = int(record[0])
        grouped.setdefault(frame_id, []).append(record)

    return grouped


def xywh_to_xyxy(record: List[float]) -> np.ndarray:
    """
    Convert MOT box from xywh to xyxy.
    """
    x, y, w, h = record[2:6]

    return np.array(
        [x, y, x + w, y + h],
        dtype=np.float32,
    )


def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Calculate IoU between two boxes.
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, float(inter_x2 - inter_x1))
    inter_h = max(0.0, float(inter_y2 - inter_y1))
    inter_area = inter_w * inter_h

    area_a = max(0.0, float(ax2 - ax1)) * max(0.0, float(ay2 - ay1))
    area_b = max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return float(inter_area / union)


def match_frame(
    gt_records: List[List[float]],
    pred_records: List[List[float]],
    iou_threshold: float = 0.5,
) -> Tuple[List[Tuple[int, int, int, int, float]], int, int]:
    """
    Match predictions and ground truth objects in one frame.

    Returns:
        matches:
            [(gt_index, pred_index, gt_id, pred_id, iou), ...]
        fp:
            false positives
        fn:
            false negatives
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
        matches.append((gt_idx, pred_idx, gt_id, pred_id, float(iou)))

    fp = len(pred_records) - len(matched_pred)
    fn = len(gt_records) - len(matched_gt)

    return matches, fp, fn


def build_identity_match_table(
    gt_by_frame: Dict[int, List[List[float]]],
    pred_by_frame: Dict[int, List[List[float]]],
    iou_threshold: float = 0.5,
) -> Dict[Tuple[int, int], int]:
    """
    Build identity matching table:
        (gt_id, pred_id) -> matched frame count
    """
    match_table = {}

    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    for frame_id in all_frames:
        gt_frame = gt_by_frame.get(frame_id, [])
        pred_frame = pred_by_frame.get(frame_id, [])

        matches, _, _ = match_frame(
            gt_records=gt_frame,
            pred_records=pred_frame,
            iou_threshold=iou_threshold,
        )

        for _, _, gt_id, pred_id, _ in matches:
            key = (int(gt_id), int(pred_id))
            match_table[key] = match_table.get(key, 0) + 1

    return match_table


def greedy_identity_assignment(
    match_table: Dict[Tuple[int, int], int],
) -> Dict[int, int]:
    """
    Greedy assignment between ground-truth identities and predicted identities.
    """
    candidates = []

    for (gt_id, pred_id), count in match_table.items():
        candidates.append((count, gt_id, pred_id))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    used_gt = set()
    used_pred = set()
    assignment = {}

    for count, gt_id, pred_id in candidates:
        if gt_id in used_gt or pred_id in used_pred:
            continue

        used_gt.add(gt_id)
        used_pred.add(pred_id)
        assignment[int(gt_id)] = int(pred_id)

    return assignment


def evaluate_one_sequence(
    gt_file: str,
    pred_file: str,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate simplified HOTA, DetA and AssA for one sequence.

    Note:
        This is a lightweight implementation for project checking.
        For strict benchmark evaluation, TrackEval is recommended.
    """
    gt_records = load_mot_file(gt_file)
    pred_records = load_mot_file(pred_file)

    gt_by_frame = group_by_frame(gt_records)
    pred_by_frame = group_by_frame(pred_records)

    all_frames = sorted(set(gt_by_frame.keys()) | set(pred_by_frame.keys()))

    total_gt = len(gt_records)
    total_pred = len(pred_records)

    tp = 0
    fp = 0
    fn = 0

    for frame_id in all_frames:
        gt_frame = gt_by_frame.get(frame_id, [])
        pred_frame = pred_by_frame.get(frame_id, [])

        matches, frame_fp, frame_fn = match_frame(
            gt_records=gt_frame,
            pred_records=pred_frame,
            iou_threshold=iou_threshold,
        )

        tp += len(matches)
        fp += frame_fp
        fn += frame_fn

    if tp + fp + fn == 0:
        deta = 0.0
    else:
        deta = tp / (tp + fp + fn)

    match_table = build_identity_match_table(
        gt_by_frame=gt_by_frame,
        pred_by_frame=pred_by_frame,
        iou_threshold=iou_threshold,
    )

    assignment = greedy_identity_assignment(match_table)

    idtp = 0

    for gt_id, pred_id in assignment.items():
        idtp += match_table.get((gt_id, pred_id), 0)

    idfp = total_pred - idtp
    idfn = total_gt - idtp

    if idtp + idfp + idfn == 0:
        assa = 0.0
    else:
        assa = idtp / (idtp + idfp + idfn)

    hota = float(np.sqrt(max(deta, 0.0) * max(assa, 0.0)))

    result = {
        "HOTA": float(hota),
        "DetA": float(deta),
        "AssA": float(assa),
        "TP": float(tp),
        "FP": float(fp),
        "FN": float(fn),
        "IDTP": float(idtp),
        "IDFP": float(idfp),
        "IDFN": float(idfn),
        "GT": float(total_gt),
        "Pred": float(total_pred),
    }

    return result


def find_sequence_pairs(gt_dir: str, pred_dir: str) -> List[Tuple[Path, Path]]:
    """
    Find ground truth and prediction sequence pairs.
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


def evaluate_hota(
    gt_dir: str,
    pred_dir: str,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluate simplified HOTA on multiple sequences.
    """
    pairs = find_sequence_pairs(gt_dir, pred_dir)

    if len(pairs) == 0:
        print("No matched sequence files found.")
        return {
            "HOTA": 0.0,
            "DetA": 0.0,
            "AssA": 0.0,
            "TP": 0.0,
            "FP": 0.0,
            "FN": 0.0,
            "IDTP": 0.0,
            "IDFP": 0.0,
            "IDFN": 0.0,
        }

    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_idtp = 0.0
    total_idfp = 0.0
    total_idfn = 0.0

    print("\nHOTA Evaluation")
    print("-" * 80)

    for gt_file, pred_file in pairs:
        result = evaluate_one_sequence(
            gt_file=str(gt_file),
            pred_file=str(pred_file),
            iou_threshold=iou_threshold,
        )

        total_tp += result["TP"]
        total_fp += result["FP"]
        total_fn += result["FN"]
        total_idtp += result["IDTP"]
        total_idfp += result["IDFP"]
        total_idfn += result["IDFN"]

        print(
            f"{gt_file.stem:25s} "
            f"HOTA: {result['HOTA']:.4f} | "
            f"DetA: {result['DetA']:.4f} | "
            f"AssA: {result['AssA']:.4f} | "
            f"TP: {int(result['TP'])} | "
            f"FP: {int(result['FP'])} | "
            f"FN: {int(result['FN'])}"
        )

    if total_tp + total_fp + total_fn == 0:
        overall_deta = 0.0
    else:
        overall_deta = total_tp / (total_tp + total_fp + total_fn)

    if total_idtp + total_idfp + total_idfn == 0:
        overall_assa = 0.0
    else:
        overall_assa = total_idtp / (total_idtp + total_idfp + total_idfn)

    overall_hota = float(np.sqrt(max(overall_deta, 0.0) * max(overall_assa, 0.0)))

    overall = {
        "HOTA": float(overall_hota),
        "DetA": float(overall_deta),
        "AssA": float(overall_assa),
        "TP": float(total_tp),
        "FP": float(total_fp),
        "FN": float(total_fn),
        "IDTP": float(total_idtp),
        "IDFP": float(total_idfp),
        "IDFN": float(total_idfn),
    }

    print("-" * 80)
    print(
        f"{'Overall':25s} "
        f"HOTA: {overall['HOTA']:.4f} | "
        f"DetA: {overall['DetA']:.4f} | "
        f"AssA: {overall['AssA']:.4f} | "
        f"TP: {int(overall['TP'])} | "
        f"FP: {int(overall['FP'])} | "
        f"FN: {int(overall['FN'])}"
    )

    return overall


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate simplified HOTA for ProTracker results."
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

    evaluate_hota(
        gt_dir=args.gt,
        pred_dir=args.pred,
        iou_threshold=args.iou_threshold,
    )


if __name__ == "__main__":
    main()