from typing import Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, list, tuple]


def xyxy_to_xywh(box: ArrayLike) -> np.ndarray:
    """
    Convert bounding box from xyxy format to xywh format.

    Args:
        box: Bounding box in [x1, y1, x2, y2] format.

    Returns:
        Bounding box in [x, y, w, h] format.
    """
    box = np.asarray(box, dtype=np.float32)

    x1, y1, x2, y2 = box

    return np.array(
        [
            x1,
            y1,
            x2 - x1,
            y2 - y1,
        ],
        dtype=np.float32,
    )


def xywh_to_xyxy(box: ArrayLike) -> np.ndarray:
    """
    Convert bounding box from xywh format to xyxy format.

    Args:
        box: Bounding box in [x, y, w, h] format.

    Returns:
        Bounding box in [x1, y1, x2, y2] format.
    """
    box = np.asarray(box, dtype=np.float32)

    x, y, w, h = box

    return np.array(
        [
            x,
            y,
            x + w,
            y + h,
        ],
        dtype=np.float32,
    )


def cxcywh_to_xyxy(box: ArrayLike) -> np.ndarray:
    """
    Convert bounding box from cxcywh format to xyxy format.

    Args:
        box: Bounding box in [cx, cy, w, h] format.

    Returns:
        Bounding box in [x1, y1, x2, y2] format.
    """
    box = np.asarray(box, dtype=np.float32)

    cx, cy, w, h = box

    return np.array(
        [
            cx - w / 2.0,
            cy - h / 2.0,
            cx + w / 2.0,
            cy + h / 2.0,
        ],
        dtype=np.float32,
    )


def xyxy_to_cxcywh(box: ArrayLike) -> np.ndarray:
    """
    Convert bounding box from xyxy format to cxcywh format.

    Args:
        box: Bounding box in [x1, y1, x2, y2] format.

    Returns:
        Bounding box in [cx, cy, w, h] format.
    """
    box = np.asarray(box, dtype=np.float32)

    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0

    return np.array(
        [
            cx,
            cy,
            w,
            h,
        ],
        dtype=np.float32,
    )


def box_area(box: ArrayLike, fmt: str = "xyxy") -> float:
    """
    Calculate bounding box area.

    Args:
        box: Bounding box.
        fmt: Box format. Supported values: "xyxy" and "xywh".

    Returns:
        Box area.
    """
    box = np.asarray(box, dtype=np.float32)

    if fmt == "xyxy":
        x1, y1, x2, y2 = box
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))

    elif fmt == "xywh":
        _, _, w, h = box
        w = max(0.0, float(w))
        h = max(0.0, float(h))

    else:
        raise ValueError(f"Unsupported box format: {fmt}")

    return w * h


def box_center(box: ArrayLike, fmt: str = "xyxy") -> np.ndarray:
    """
    Calculate bounding box center.

    Args:
        box: Bounding box.
        fmt: Box format. Supported values: "xyxy" and "xywh".

    Returns:
        Center point in [cx, cy] format.
    """
    box = np.asarray(box, dtype=np.float32)

    if fmt == "xyxy":
        x1, y1, x2, y2 = box
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

    elif fmt == "xywh":
        x, y, w, h = box
        cx = x + w / 2.0
        cy = y + h / 2.0

    else:
        raise ValueError(f"Unsupported box format: {fmt}")

    return np.array([cx, cy], dtype=np.float32)


def box_iou(box_a: ArrayLike, box_b: ArrayLike) -> float:
    """
    Calculate IoU between two bounding boxes in xyxy format.

    Args:
        box_a: First box in [x1, y1, x2, y2] format.
        box_b: Second box in [x1, y1, x2, y2] format.

    Returns:
        IoU value.
    """
    box_a = np.asarray(box_a, dtype=np.float32)
    box_b = np.asarray(box_b, dtype=np.float32)

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, float(inter_x2 - inter_x1))
    inter_h = max(0.0, float(inter_y2 - inter_y1))
    inter_area = inter_w * inter_h

    area_a = box_area(box_a, fmt="xyxy")
    area_b = box_area(box_b, fmt="xyxy")

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return float(inter_area / union)


def pairwise_iou(boxes_a: ArrayLike, boxes_b: ArrayLike) -> np.ndarray:
    """
    Calculate pairwise IoU between two sets of boxes.

    Args:
        boxes_a: Boxes with shape [N, 4] in xyxy format.
        boxes_b: Boxes with shape [M, 4] in xyxy format.

    Returns:
        IoU matrix with shape [N, M].
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float32)
    boxes_b = np.asarray(boxes_b, dtype=np.float32)

    if boxes_a.ndim == 1:
        boxes_a = boxes_a[None, :]

    if boxes_b.ndim == 1:
        boxes_b = boxes_b[None, :]

    iou_matrix = np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            iou_matrix[i, j] = box_iou(box_a, box_b)

    return iou_matrix


def clip_box(
    box: ArrayLike,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Clip bounding box into image boundary.

    Args:
        box: Bounding box in xyxy format.
        image_shape: Image shape in (height, width) format.

    Returns:
        Clipped bounding box.
    """
    box = np.asarray(box, dtype=np.float32).copy()

    if image_shape is None:
        return box

    height, width = image_shape

    box[0] = np.clip(box[0], 0, width - 1)
    box[1] = np.clip(box[1], 0, height - 1)
    box[2] = np.clip(box[2], 0, width - 1)
    box[3] = np.clip(box[3], 0, height - 1)

    return box


def expand_box(
    box: ArrayLike,
    ratio: float = 1.15,
    image_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Expand a bounding box around its center.

    Args:
        box: Bounding box in xyxy format.
        ratio: Expansion ratio.
        image_shape: Optional image shape in (height, width) format.

    Returns:
        Expanded bounding box in xyxy format.
    """
    box = np.asarray(box, dtype=np.float32)

    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, float(x2 - x1))
    h = max(1.0, float(y2 - y1))

    new_w = w * ratio
    new_h = h * ratio

    expanded = np.array(
        [
            cx - new_w / 2.0,
            cy - new_h / 2.0,
            cx + new_w / 2.0,
            cy + new_h / 2.0,
        ],
        dtype=np.float32,
    )

    expanded = clip_box(expanded, image_shape)

    return expanded


def normalize_box(
    box: ArrayLike,
    image_shape: Tuple[int, int],
    fmt: str = "xyxy",
) -> np.ndarray:
    """
    Normalize a bounding box by image width and height.

    Args:
        box: Bounding box.
        image_shape: Image shape in (height, width) format.
        fmt: Box format. Supported values: "xyxy" and "xywh".

    Returns:
        Normalized bounding box.
    """
    box = np.asarray(box, dtype=np.float32).copy()
    height, width = image_shape

    if fmt == "xyxy":
        box[[0, 2]] /= max(width, 1)
        box[[1, 3]] /= max(height, 1)

    elif fmt == "xywh":
        box[[0, 2]] /= max(width, 1)
        box[[1, 3]] /= max(height, 1)

    else:
        raise ValueError(f"Unsupported box format: {fmt}")

    return box


def denormalize_box(
    box: ArrayLike,
    image_shape: Tuple[int, int],
    fmt: str = "xyxy",
) -> np.ndarray:
    """
    Denormalize a bounding box according to image width and height.

    Args:
        box: Normalized bounding box.
        image_shape: Image shape in (height, width) format.
        fmt: Box format. Supported values: "xyxy" and "xywh".

    Returns:
        Denormalized bounding box.
    """
    box = np.asarray(box, dtype=np.float32).copy()
    height, width = image_shape

    if fmt == "xyxy":
        box[[0, 2]] *= width
        box[[1, 3]] *= height

    elif fmt == "xywh":
        box[[0, 2]] *= width
        box[[1, 3]] *= height

    else:
        raise ValueError(f"Unsupported box format: {fmt}")

    return box


def is_valid_box(
    box: ArrayLike,
    min_area: float = 1.0,
    fmt: str = "xyxy",
) -> bool:
    """
    Check whether a bounding box is valid.

    Args:
        box: Bounding box.
        min_area: Minimum valid area.
        fmt: Box format. Supported values: "xyxy" and "xywh".

    Returns:
        True if the box is valid, otherwise False.
    """
    area = box_area(box, fmt=fmt)

    if area < min_area:
        return False

    return True