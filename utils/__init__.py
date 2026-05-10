"""
Utility modules for ProTracker.

This package contains common utility functions for video processing,
bounding box operations, mask operations, MOT-format conversion, and logging.
"""

from .video_io import read_video_frames, write_video_frames, get_video_info
from .box_ops import (
    xyxy_to_xywh,
    xywh_to_xyxy,
    box_area,
    box_iou,
    clip_box,
    expand_box,
)
from .mask_ops import (
    mask_to_box,
    masks_to_boxes,
    calculate_mask_area,
)
from .mot_format import (
    save_mot_results,
    load_mot_results,
    detection_to_mot_line,
)
from .logger import setup_logger

__all__ = [
    "read_video_frames",
    "write_video_frames",
    "get_video_info",
    "xyxy_to_xywh",
    "xywh_to_xyxy",
    "box_area",
    "box_iou",
    "clip_box",
    "expand_box",
    "mask_to_box",
    "masks_to_boxes",
    "calculate_mask_area",
    "save_mot_results",
    "load_mot_results",
    "detection_to_mot_line",
    "setup_logger",
]