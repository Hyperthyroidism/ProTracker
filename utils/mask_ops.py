from typing import Dict, List, Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, list, tuple]


def calculate_mask_area(mask: ArrayLike) -> float:
    """
    Calculate the area of a binary mask.

    Args:
        mask: Binary mask.

    Returns:
        Mask area, namely the number of foreground pixels.
    """
    mask = np.asarray(mask)

    if mask.ndim > 2:
        mask = np.squeeze(mask)

    area = float(np.sum(mask > 0))

    return area


def mask_to_box(mask: ArrayLike) -> Optional[np.ndarray]:
    """
    Convert a binary mask to a bounding box.

    Args:
        mask: Binary mask.

    Returns:
        Bounding box in [x1, y1, x2, y2] format.
        If the mask is empty, return None.
    """
    mask = np.asarray(mask)

    if mask.ndim > 2:
        mask = np.squeeze(mask)

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        return None

    x1 = float(np.min(xs))
    y1 = float(np.min(ys))
    x2 = float(np.max(xs))
    y2 = float(np.max(ys))

    return np.array([x1, y1, x2, y2], dtype=np.float32)


def masks_to_boxes(masks: Union[List[np.ndarray], Dict[int, np.ndarray]]) -> Dict[int, Optional[np.ndarray]]:
    """
    Convert multiple masks to bounding boxes.

    Args:
        masks: A list of masks or a dictionary mapping object id to mask.

    Returns:
        Dictionary mapping mask index or object id to bounding box.
    """
    boxes = {}

    if isinstance(masks, dict):
        for object_id, mask in masks.items():
            boxes[int(object_id)] = mask_to_box(mask)
    else:
        for idx, mask in enumerate(masks):
            boxes[int(idx)] = mask_to_box(mask)

    return boxes


def mask_to_xywh(mask: ArrayLike) -> Optional[np.ndarray]:
    """
    Convert a binary mask to an xywh bounding box.

    Args:
        mask: Binary mask.

    Returns:
        Bounding box in [x, y, w, h] format.
        If the mask is empty, return None.
    """
    box_xyxy = mask_to_box(mask)

    if box_xyxy is None:
        return None

    x1, y1, x2, y2 = box_xyxy

    return np.array(
        [
            x1,
            y1,
            x2 - x1,
            y2 - y1,
        ],
        dtype=np.float32,
    )


def resize_mask(
    mask: ArrayLike,
    target_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Resize a binary mask to target shape.

    Args:
        mask: Binary mask.
        target_shape: Target shape in (height, width) format.

    Returns:
        Resized binary mask.
    """
    import cv2

    mask = np.asarray(mask)

    if mask.ndim > 2:
        mask = np.squeeze(mask)

    target_h, target_w = target_shape

    resized = cv2.resize(
        mask.astype(np.uint8),
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST,
    )

    return (resized > 0).astype(np.uint8)


def binarize_mask(
    mask: ArrayLike,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Binarize a mask or mask logits.

    Args:
        mask: Input mask or mask logits.
        threshold: Threshold for binarization.

    Returns:
        Binary mask.
    """
    mask = np.asarray(mask)

    if mask.ndim > 2:
        mask = np.squeeze(mask)

    binary_mask = (mask > threshold).astype(np.uint8)

    return binary_mask


def merge_masks(
    masks: List[np.ndarray],
    mode: str = "union",
) -> np.ndarray:
    """
    Merge multiple masks into one mask.

    Args:
        masks: List of binary masks.
        mode: Merge mode. Supported values: "union" and "intersection".

    Returns:
        Merged binary mask.
    """
    if len(masks) == 0:
        raise ValueError("masks is empty.")

    masks = [np.asarray(mask).astype(np.uint8) for mask in masks]

    if mode == "union":
        merged = np.zeros_like(masks[0], dtype=np.uint8)

        for mask in masks:
            merged = np.logical_or(merged, mask > 0)

        return merged.astype(np.uint8)

    if mode == "intersection":
        merged = np.ones_like(masks[0], dtype=np.uint8)

        for mask in masks:
            merged = np.logical_and(merged, mask > 0)

        return merged.astype(np.uint8)

    raise ValueError(f"Unsupported merge mode: {mode}")


def calculate_mask_iou(
    mask_a: ArrayLike,
    mask_b: ArrayLike,
) -> float:
    """
    Calculate IoU between two binary masks.

    Args:
        mask_a: First binary mask.
        mask_b: Second binary mask.

    Returns:
        Mask IoU value.
    """
    mask_a = np.asarray(mask_a)

    if mask_a.ndim > 2:
        mask_a = np.squeeze(mask_a)

    mask_b = np.asarray(mask_b)

    if mask_b.ndim > 2:
        mask_b = np.squeeze(mask_b)

    if mask_a.shape != mask_b.shape:
        raise ValueError(
            f"Mask shapes are different: {mask_a.shape} vs {mask_b.shape}"
        )

    mask_a = mask_a > 0
    mask_b = mask_b > 0

    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()

    if union == 0:
        return 0.0

    return float(intersection / union)


def filter_small_masks(
    masks: Dict[int, np.ndarray],
    min_area: float = 16.0,
) -> Dict[int, np.ndarray]:
    """
    Remove masks whose area is smaller than min_area.

    Args:
        masks: Dictionary mapping object id to binary mask.
        min_area: Minimum valid mask area.

    Returns:
        Filtered mask dictionary.
    """
    filtered_masks = {}

    for object_id, mask in masks.items():
        area = calculate_mask_area(mask)

        if area >= min_area:
            filtered_masks[int(object_id)] = mask

    return filtered_masks


def apply_mask_to_image(
    image: np.ndarray,
    mask: ArrayLike,
    alpha: float = 0.5,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Overlay a binary mask on an image.

    Args:
        image: Input image in BGR format.
        mask: Binary mask.
        alpha: Blending ratio.
        color: Mask color in BGR format.

    Returns:
        Image with mask overlay.
    """
    image = image.copy()
    mask = np.asarray(mask)

    if mask.ndim > 2:
        mask = np.squeeze(mask)

    mask = mask > 0

    overlay = image.copy()
    overlay[mask] = color

    image = (image * (1 - alpha) + overlay * alpha).astype(np.uint8)

    return image


def masks_to_mot_results(
    frame_id: int,
    masks: Dict[int, np.ndarray],
    confidence: float = 1.0,
    class_id: int = 0,
    visibility: float = 1.0,
) -> List[List[float]]:
    """
    Convert frame-level masks to MOT-format results.

    Args:
        frame_id: Frame index. MOT format usually starts from 1.
        masks: Dictionary mapping track id to mask.
        confidence: Confidence score.
        class_id: Object class id.
        visibility: Visibility ratio.

    Returns:
        MOT-format result list.
    """
    mot_results = []

    for track_id, mask in masks.items():
        box_xywh = mask_to_xywh(mask)

        if box_xywh is None:
            continue

        x, y, w, h = box_xywh

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

        mot_results.append(mot_line)

    return mot_results