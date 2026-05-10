"""
Model modules for ProTracker.

This package contains the main model wrappers used in the ProTracker framework,
including the YOLO11 detector wrapper, SAM2 predictor wrapper, target-aware
refinement module, and the complete ProTracker model.
"""

from .yolo_detector import YOLODetector
from .sam2_predictor import SAM2Predictor
from .target_refinement import TargetRefinementModule
from .protracker import ProTracker

__all__ = [
    "YOLODetector",
    "SAM2Predictor",
    "TargetRefinementModule",
    "ProTracker",
]