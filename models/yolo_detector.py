from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


class YOLODetector:
    """
    YOLO11 detector wrapper for ProTracker.

    This module is used to detect vessel candidates from UAV video frames.
    The raw detection results from YOLO are converted into a unified format
    for the following graph-based prompt optimizer and SAM2 tracking modules.
    """

    def __init__(
        self,
        model_path: str,
        image_size: int = 640,
        confidence_threshold: float = 0.25,
        nms_threshold: float = 0.70,
        device: str = "cuda",
        class_names: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            model_path: Path to the YOLO11 weight file.
            image_size: Input image size for YOLO inference.
            confidence_threshold: Detection confidence threshold.
            nms_threshold: NMS IoU threshold.
            device: Inference device, such as cuda or cpu.
            class_names: Class name list. For this project, it is usually ["vessel"].
        """
        self.model_path = Path(model_path)
        self.image_size = image_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.device = self._build_device(device)
        self.class_names = class_names if class_names is not None else ["vessel"]

        self.model = self._load_model()

    def _build_device(self, device: str) -> str:
        """
        Build inference device.

        Args:
            device: Device name from config.

        Returns:
            Available device name.
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. YOLODetector uses CPU instead.")
            return "cpu"

        return device

    def _load_model(self) -> Any:
        """
        Load YOLO11 model.

        Returns:
            YOLO model object.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO weight file not found: {self.model_path}")

        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "Failed to import ultralytics. "
                "Please install it with `pip install ultralytics`, "
                "or make sure external/ultralytics is correctly configured."
            ) from e

        model = YOLO(str(self.model_path))
        return model

    @staticmethod
    def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
        """
        Convert box format from xyxy to xywh.

        Args:
            box: Bounding box in [x1, y1, x2, y2] format.

        Returns:
            Bounding box in [x, y, w, h] format.
        """
        x1, y1, x2, y2 = box
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Detect vessels in a single frame.

        Args:
            frame: Input image frame in BGR or RGB format.
            frame_id: Current frame index.

        Returns:
            A list of detection dictionaries. Each detection has the following fields:

            {
                "frame_id": int,
                "bbox_xyxy": np.ndarray,  # [x1, y1, x2, y2]
                "bbox_xywh": np.ndarray,  # [x, y, w, h]
                "confidence": float,
                "class_id": int,
                "class_name": str,
            }
        """
        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            iou=self.nms_threshold,
            device=self.device,
            verbose=False,
        )

        if len(results) == 0:
            return []

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes_xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

        detections = []

        for box_xyxy, conf, cls_id in zip(boxes_xyxy, confidences, class_ids):
            box_xyxy = box_xyxy.astype(np.float32)
            box_xywh = self.xyxy_to_xywh(box_xyxy)

            if cls_id < len(self.class_names):
                class_name = self.class_names[cls_id]
            else:
                class_name = "unknown"

            detection = {
                "frame_id": int(frame_id),
                "bbox_xyxy": box_xyxy,
                "bbox_xywh": box_xywh,
                "confidence": float(conf),
                "class_id": int(cls_id),
                "class_name": class_name,
            }

            detections.append(detection)

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        start_frame_id: int = 0,
    ) -> List[List[Dict[str, Any]]]:
        """
        Detect vessels in a batch of frames.

        Args:
            frames: List of input frames.
            start_frame_id: Frame index of the first frame.

        Returns:
            A list where each element contains detections of one frame.
        """
        all_detections = []

        for idx, frame in enumerate(frames):
            frame_id = start_frame_id + idx
            detections = self.detect_frame(frame, frame_id=frame_id)
            all_detections.append(detections)

        return all_detections

    def format_as_mot(
        self,
        detections: List[Dict[str, Any]],
        default_track_id: int = -1,
    ) -> List[List[float]]:
        """
        Convert detection results into MOT-like format.

        Args:
            detections: Detection list returned by detect_frame.
            default_track_id: Temporary ID before tracking association.

        Returns:
            MOT-like result list:
            [frame_id, track_id, x, y, w, h, confidence, class_id, visibility]
        """
        mot_results = []

        for det in detections:
            x, y, w, h = det["bbox_xywh"]
            mot_line = [
                float(det["frame_id"]),
                float(default_track_id),
                float(x),
                float(y),
                float(w),
                float(h),
                float(det["confidence"]),
                float(det["class_id"]),
                1.0,
            ]
            mot_results.append(mot_line)

        return mot_results

    def __call__(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Make the detector callable.

        Args:
            frame: Input frame.
            frame_id: Current frame index.

        Returns:
            Detection results.
        """
        return self.detect_frame(frame, frame_id=frame_id)