from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from models.yolo_detector import YOLODetector
from models.sam2_predictor import SAM2Predictor
from models.target_refinement import TargetRefinementModule


class ProTracker:
    """
    ProTracker main model.

    ProTracker is a prompt-oriented multi-vessel tracking framework for UAV-based
    waterway scenes. It integrates the following modules:

        1. YOLO11 vessel detector
        2. Graph-based cascaded prompt optimizer
        3. Target-aware refinement module
        4. SAM2-based video segmentation and tracking module

    This class provides a unified interface for training, testing, and inference.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Configuration dictionary loaded from yaml files.
        """
        self.config = config

        project_cfg = config.get("project", {})
        self.device = project_cfg.get("device", "cuda")

        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. ProTracker uses CPU instead.")
            self.device = "cpu"

        self.detector = self._build_detector()
        self.graph_prompt_optimizer = self._build_graph_prompt_optimizer()
        self.target_refiner = self._build_target_refiner()
        self.sam2_predictor = self._build_sam2_predictor()

    def _build_detector(self) -> Optional[YOLODetector]:
        """
        Build YOLO11 vessel detector.

        Returns:
            YOLODetector object or None.
        """
        detector_cfg = self.config.get("detector", {})
        enabled = detector_cfg.get("enabled", True)

        if not enabled:
            return None

        model_path = detector_cfg.get("model_path", "weights/yolo11/yolo11_vessel.pt")
        image_size = detector_cfg.get("image_size", 640)
        confidence_threshold = detector_cfg.get("confidence_threshold", 0.25)
        nms_threshold = detector_cfg.get("nms_threshold", 0.70)
        device = detector_cfg.get("device", self.device)

        dataset_cfg = self.config.get("dataset", {})
        class_names = dataset_cfg.get("class_names", ["vessel"])

        detector = YOLODetector(
            model_path=model_path,
            image_size=image_size,
            confidence_threshold=confidence_threshold,
            nms_threshold=nms_threshold,
            device=device,
            class_names=class_names,
        )

        return detector

    def _build_graph_prompt_optimizer(self) -> Optional[torch.nn.Module]:
        """
        Build graph-based cascaded prompt optimizer.

        Returns:
            GraphPromptOptimizer object or None.
        """
        optimizer_cfg = self.config.get("graph_prompt_optimizer", {})
        enabled = optimizer_cfg.get("enabled", True)

        if not enabled:
            return None

        try:
            from networks.prompt_optimizer import GraphPromptOptimizer
        except ImportError as e:
            raise ImportError(
                "Failed to import GraphPromptOptimizer. "
                "Please make sure networks/prompt_optimizer.py exists."
            ) from e

        model = GraphPromptOptimizer(self.config)
        model = model.to(self.device)

        checkpoint_path = optimizer_cfg.get("checkpoint_path", None)

        if checkpoint_path is not None:
            checkpoint_path = Path(checkpoint_path)

            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

                if isinstance(checkpoint, dict) and "model" in checkpoint:
                    model.load_state_dict(checkpoint["model"], strict=False)
                else:
                    model.load_state_dict(checkpoint, strict=False)

                print(f"Graph prompt optimizer checkpoint loaded from: {checkpoint_path}")
            else:
                print(f"Graph prompt optimizer checkpoint not found: {checkpoint_path}")

        model.eval()
        return model

    def _build_target_refiner(self) -> Optional[TargetRefinementModule]:
        """
        Build target-aware refinement module.

        Returns:
            TargetRefinementModule object or None.
        """
        refinement_cfg = self.config.get("target_refinement", {})
        enabled = refinement_cfg.get("enabled", True)

        if not enabled:
            return None

        refiner = TargetRefinementModule(
            min_box_area=refinement_cfg.get("min_box_area", 16),
            expand_ratio=refinement_cfg.get("expand_ratio", 1.15),
            refine_small_targets=refinement_cfg.get("refine_small_targets", True),
            refine_occluded_targets=refinement_cfg.get("refine_occluded_targets", True),
            supplement_missing_targets=refinement_cfg.get("supplement_missing_targets", True),
        )

        return refiner

    def _build_sam2_predictor(self) -> Optional[SAM2Predictor]:
        """
        Build SAM2 predictor.

        Returns:
            SAM2Predictor object or None.
        """
        sam2_cfg = self.config.get("sam2", {})
        enabled = sam2_cfg.get("enabled", True)

        if not enabled:
            return None

        predictor = SAM2Predictor(
            config_path=sam2_cfg.get(
                "config_path",
                "external/sam2/configs/sam2_hiera_l.yaml"
            ),
            checkpoint_path=sam2_cfg.get(
                "checkpoint_path",
                "weights/sam2/sam2_hiera_large.pt"
            ),
            model_type=sam2_cfg.get("model_type", "sam2_hiera_large"),
            device=sam2_cfg.get("device", self.device),
            prompt_type=sam2_cfg.get("prompt_type", "box"),
            use_mask_prompt=sam2_cfg.get("use_mask_prompt", True),
            memory_enabled=sam2_cfg.get("memory_enabled", True),
        )

        return predictor

    @staticmethod
    def detections_to_prompts(
        detections: List[Dict[str, Any]],
        start_track_id: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Convert detection results to initial prompts.

        This function is used as a fallback when graph-based prompt optimization
        is not available.

        Args:
            detections: Detection results from YOLODetector.
            start_track_id: Start ID for temporary prompt identities.

        Returns:
            Prompt list.
        """
        prompts = []

        for idx, det in enumerate(detections):
            prompt = {
                "frame_id": det.get("frame_id", 0),
                "track_id": det.get("track_id", start_track_id + idx),
                "bbox_xyxy": np.asarray(det["bbox_xyxy"], dtype=np.float32),
                "bbox_xywh": np.asarray(det["bbox_xywh"], dtype=np.float32),
                "confidence": float(det.get("confidence", 1.0)),
                "class_id": int(det.get("class_id", 0)),
                "class_name": det.get("class_name", "vessel"),
                "source": "detector",
            }

            prompts.append(prompt)

        return prompts

    def detect_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Detect vessel candidates in one frame.

        Args:
            frame: Input video frame.
            frame_id: Current frame index.

        Returns:
            Detection list.
        """
        if self.detector is None:
            return []

        detections = self.detector.detect_frame(
            frame=frame,
            frame_id=frame_id,
        )

        return detections

    def optimize_prompts(
        self,
        detections: List[Dict[str, Any]],
        frame_id: int = 0,
        track_manager: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate optimized vessel prompts using graph-based cascaded prompt optimizer.

        Args:
            detections: Vessel detections of the current frame.
            frame_id: Current frame index.
            track_manager: Track manager that stores existing trajectories.

        Returns:
            Optimized prompt list.
        """
        if len(detections) == 0:
            return []

        if self.graph_prompt_optimizer is None:
            return self.detections_to_prompts(detections)

        with torch.no_grad():
            if hasattr(self.graph_prompt_optimizer, "generate_prompts"):
                prompts = self.graph_prompt_optimizer.generate_prompts(
                    detections=detections,
                    frame_id=frame_id,
                    track_manager=track_manager,
                )
            else:
                prompts = self.detections_to_prompts(detections)

        return prompts

    def refine_prompts(
        self,
        prompts: List[Dict[str, Any]],
        image_shape: Optional[Any] = None,
        track_histories: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Refine optimized prompts before feeding them into SAM2.

        Args:
            prompts: Prompt list.
            image_shape: Image shape in (height, width) format.
            track_histories: Track history dictionary.

        Returns:
            Refined prompt list.
        """
        if self.target_refiner is None:
            return prompts

        refined_prompts = self.target_refiner(
            prompts=prompts,
            image_shape=image_shape,
            track_histories=track_histories,
        )

        return refined_prompts

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        track_manager: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Process one video frame before SAM2 propagation.

        The frame-level process contains:

            1. YOLO11 vessel detection
            2. Graph-based prompt optimization
            3. Target-aware prompt refinement

        Args:
            frame: Input video frame.
            frame_id: Current frame index.
            track_manager: Track manager.

        Returns:
            Frame-level result dictionary.
        """
        image_shape = frame.shape[:2]

        detections = self.detect_frame(
            frame=frame,
            frame_id=frame_id,
        )

        optimized_prompts = self.optimize_prompts(
            detections=detections,
            frame_id=frame_id,
            track_manager=track_manager,
        )

        track_histories = None

        if track_manager is not None and hasattr(track_manager, "get_histories"):
            track_histories = track_manager.get_histories()

        refined_prompts = self.refine_prompts(
            prompts=optimized_prompts,
            image_shape=image_shape,
            track_histories=track_histories,
        )

        frame_result = {
            "frame_id": frame_id,
            "detections": detections,
            "optimized_prompts": optimized_prompts,
            "refined_prompts": refined_prompts,
        }

        return frame_result

    def run_sam2_tracking(
        self,
        video_path: str,
        prompts: List[Dict[str, Any]],
        initial_frame_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Run SAM2-based video segmentation and tracking.

        Args:
            video_path: Input video path.
            prompts: Refined prompt list.
            initial_frame_idx: Initial frame index where prompts are added.

        Returns:
            Video segmentation results.
        """
        if self.sam2_predictor is None:
            return []

        segments = self.sam2_predictor(
            video_path=video_path,
            prompts=prompts,
            initial_frame_idx=initial_frame_idx,
        )

        return segments

    def build_initial_prompts_from_video(
        self,
        frames: List[np.ndarray],
        track_manager: Optional[Any] = None,
        initial_frame_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Build initial prompts from a selected video frame.

        Args:
            frames: List of video frames.
            track_manager: Track manager.
            initial_frame_idx: Index of the frame used to initialize prompts.

        Returns:
            Refined initial prompts.
        """
        if len(frames) == 0:
            return []

        if initial_frame_idx < 0 or initial_frame_idx >= len(frames):
            initial_frame_idx = 0

        initial_frame = frames[initial_frame_idx]

        frame_result = self.process_frame(
            frame=initial_frame,
            frame_id=initial_frame_idx,
            track_manager=track_manager,
        )

        return frame_result["refined_prompts"]

    def run_video(
        self,
        video_path: str,
        frames: Optional[List[np.ndarray]] = None,
        track_manager: Optional[Any] = None,
        initial_frame_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Run ProTracker on one video.

        Args:
            video_path: Input video path.
            frames: Optional preloaded video frames.
            track_manager: Track manager.
            initial_frame_idx: Initial prompt frame index.

        Returns:
            Video-level result dictionary.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        if frames is None:
            from utils.video_io import read_video_frames

            frames = read_video_frames(str(video_path))

        initial_prompts = self.build_initial_prompts_from_video(
            frames=frames,
            track_manager=track_manager,
            initial_frame_idx=initial_frame_idx,
        )

        sam2_segments = self.run_sam2_tracking(
            video_path=str(video_path),
            prompts=initial_prompts,
            initial_frame_idx=initial_frame_idx,
        )

        result = {
            "video_path": str(video_path),
            "initial_frame_idx": initial_frame_idx,
            "initial_prompts": initial_prompts,
            "sam2_segments": sam2_segments,
        }

        return result

    def train(self) -> None:
        """
        Placeholder training interface.

        The detailed training process is implemented in train.py and
        networks/prompt_optimizer.py.
        """
        if self.graph_prompt_optimizer is None:
            raise RuntimeError("Graph prompt optimizer is not enabled.")

        self.graph_prompt_optimizer.train()

    def eval(self) -> None:
        """
        Set all learnable modules to evaluation mode.
        """
        if self.graph_prompt_optimizer is not None:
            self.graph_prompt_optimizer.eval()

    def __call__(
        self,
        video_path: str,
        frames: Optional[List[np.ndarray]] = None,
        track_manager: Optional[Any] = None,
        initial_frame_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Make ProTracker callable.

        Args:
            video_path: Input video path.
            frames: Optional preloaded frames.
            track_manager: Track manager.
            initial_frame_idx: Initial frame index for prompt initialization.

        Returns:
            Video-level result dictionary.
        """
        return self.run_video(
            video_path=video_path,
            frames=frames,
            track_manager=track_manager,
            initial_frame_idx=initial_frame_idx,
        )