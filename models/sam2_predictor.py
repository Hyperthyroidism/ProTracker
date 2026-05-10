from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


class SAM2Predictor:
    """
    SAM2 predictor wrapper for ProTracker.

    This module provides a unified interface for SAM2-based vessel segmentation
    and video tracking. It receives optimized prompts from ProTracker and returns
    fine-grained vessel masks.
    """

    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        model_type: str = "sam2_hiera_large",
        device: str = "cuda",
        prompt_type: str = "box",
        use_mask_prompt: bool = True,
        memory_enabled: bool = True,
    ) -> None:
        """
        Args:
            config_path: Path to the SAM2 config file.
            checkpoint_path: Path to the SAM2 checkpoint.
            model_type: SAM2 model type.
            device: Inference device, such as cuda or cpu.
            prompt_type: Prompt type, such as box, point, or mask.
            use_mask_prompt: Whether to use mask prompts.
            memory_enabled: Whether to use video memory mechanism.
        """
        self.config_path = Path(config_path)
        self.checkpoint_path = Path(checkpoint_path)
        self.model_type = model_type
        self.device = self._build_device(device)
        self.prompt_type = prompt_type
        self.use_mask_prompt = use_mask_prompt
        self.memory_enabled = memory_enabled

        self.predictor = self._load_predictor()

    def _build_device(self, device: str) -> str:
        """
        Build inference device.

        Args:
            device: Device name from config.

        Returns:
            Available device name.
        """
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA is not available. SAM2Predictor uses CPU instead.")
            return "cpu"

        return device

    def _load_predictor(self) -> Any:
        """
        Load SAM2 predictor.

        Returns:
            SAM2 predictor object.

        Notes:
            The official SAM2 repository may provide slightly different APIs
            across versions. This wrapper keeps the SAM2 call isolated so that
            only this file needs to be modified if the official API changes.
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {self.checkpoint_path}")

        if not self.config_path.exists():
            raise FileNotFoundError(f"SAM2 config file not found: {self.config_path}")

        try:
            from sam2.build_sam import build_sam2_video_predictor
        except ImportError as e:
            raise ImportError(
                "Failed to import SAM2. Please make sure the SAM2 repository "
                "is installed or correctly placed under external/sam2."
            ) from e

        predictor = build_sam2_video_predictor(
            config_file=str(self.config_path),
            ckpt_path=str(self.checkpoint_path),
            device=self.device,
        )

        return predictor

    @staticmethod
    def box_xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
        """
        Convert a box from xywh format to xyxy format.

        Args:
            box: Bounding box in [x, y, w, h] format.

        Returns:
            Bounding box in [x1, y1, x2, y2] format.
        """
        x, y, w, h = box
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    @staticmethod
    def normalize_box(box: np.ndarray) -> np.ndarray:
        """
        Ensure that the box is in float32 xyxy format.

        Args:
            box: Bounding box.

        Returns:
            Normalized bounding box.
        """
        box = np.asarray(box, dtype=np.float32)

        if box.shape[0] != 4:
            raise ValueError(f"Invalid box shape: {box.shape}")

        return box

    def init_video_state(self, video_path: str) -> Any:
        """
        Initialize SAM2 video inference state.

        Args:
            video_path: Path to the input video.

        Returns:
            SAM2 inference state.
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Input video not found: {video_path}")

        inference_state = self.predictor.init_state(video_path=str(video_path))
        return inference_state

    def reset_state(self, inference_state: Any) -> None:
        """
        Reset SAM2 inference state.

        Args:
            inference_state: SAM2 inference state.
        """
        if hasattr(self.predictor, "reset_state"):
            self.predictor.reset_state(inference_state)

    def add_box_prompt(
        self,
        inference_state: Any,
        frame_idx: int,
        object_id: int,
        box: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add a box prompt for one vessel object.

        Args:
            inference_state: SAM2 inference state.
            frame_idx: Frame index.
            object_id: Object identity.
            box: Bounding box in xyxy format.

        Returns:
            Tuple of object ids and mask logits returned by SAM2.
        """
        box = self.normalize_box(box)

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            box=box,
        )

        return out_obj_ids, out_mask_logits

    def add_point_prompt(
        self,
        inference_state: Any,
        frame_idx: int,
        object_id: int,
        points: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add point prompts for one vessel object.

        Args:
            inference_state: SAM2 inference state.
            frame_idx: Frame index.
            object_id: Object identity.
            points: Point coordinates with shape [N, 2].
            labels: Point labels with shape [N]. Positive point is usually 1,
                    negative point is usually 0.

        Returns:
            Tuple of object ids and mask logits returned by SAM2.
        """
        points = np.asarray(points, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int32)

        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=object_id,
            points=points,
            labels=labels,
        )

        return out_obj_ids, out_mask_logits

    def add_prompts(
        self,
        inference_state: Any,
        frame_idx: int,
        prompts: List[Dict[str, Any]],
    ) -> Dict[int, np.ndarray]:
        """
        Add multiple prompts to SAM2.

        Args:
            inference_state: SAM2 inference state.
            frame_idx: Frame index.
            prompts: Prompt list. Each prompt dictionary may contain:

                {
                    "track_id": int,
                    "bbox_xyxy": np.ndarray,
                    "points": np.ndarray,
                    "point_labels": np.ndarray,
                    "mask": np.ndarray,
                }

        Returns:
            A dictionary mapping object id to mask logits.
        """
        mask_logits_dict = {}

        for prompt in prompts:
            object_id = int(prompt.get("track_id", prompt.get("object_id", -1)))

            if object_id < 0:
                continue

            if self.prompt_type == "box" and "bbox_xyxy" in prompt:
                box = prompt["bbox_xyxy"]
                out_obj_ids, out_mask_logits = self.add_box_prompt(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    object_id=object_id,
                    box=box,
                )

            elif self.prompt_type == "point" and "points" in prompt:
                points = prompt["points"]
                labels = prompt.get(
                    "point_labels",
                    np.ones((len(points),), dtype=np.int32)
                )
                out_obj_ids, out_mask_logits = self.add_point_prompt(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    object_id=object_id,
                    points=points,
                    labels=labels,
                )

            elif "bbox_xyxy" in prompt:
                box = prompt["bbox_xyxy"]
                out_obj_ids, out_mask_logits = self.add_box_prompt(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    object_id=object_id,
                    box=box,
                )

            else:
                continue

            for obj_id, mask_logit in zip(out_obj_ids, out_mask_logits):
                mask_logits_dict[int(obj_id)] = mask_logit

        return mask_logits_dict

    def propagate_video(
        self,
        inference_state: Any,
    ) -> List[Dict[str, Any]]:
        """
        Propagate masks across the whole video using SAM2.

        Args:
            inference_state: SAM2 inference state.

        Returns:
            A list of frame-level mask results. Each element contains:

            {
                "frame_id": int,
                "object_ids": List[int],
                "masks": Dict[int, np.ndarray],
            }
        """
        video_segments = []

        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(
            inference_state
        ):
            frame_masks = {}

            for obj_id, mask_logit in zip(out_obj_ids, out_mask_logits):
                if isinstance(mask_logit, torch.Tensor):
                    mask = (mask_logit > 0.0).detach().cpu().numpy()
                else:
                    mask = np.asarray(mask_logit > 0.0)

                mask = np.squeeze(mask).astype(np.uint8)
                frame_masks[int(obj_id)] = mask

            frame_result = {
                "frame_id": int(out_frame_idx),
                "object_ids": [int(obj_id) for obj_id in out_obj_ids],
                "masks": frame_masks,
            }

            video_segments.append(frame_result)

        return video_segments

    def predict_with_prompts(
        self,
        video_path: str,
        initial_frame_idx: int,
        prompts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Run SAM2 video prediction with initial prompts.

        Args:
            video_path: Input video path.
            initial_frame_idx: Frame index where prompts are added.
            prompts: Optimized prompts generated by ProTracker.

        Returns:
            Video segmentation results.
        """
        inference_state = self.init_video_state(video_path)

        if self.memory_enabled:
            self.reset_state(inference_state)

        self.add_prompts(
            inference_state=inference_state,
            frame_idx=initial_frame_idx,
            prompts=prompts,
        )

        video_segments = self.propagate_video(inference_state)

        return video_segments

    def __call__(
        self,
        video_path: str,
        prompts: List[Dict[str, Any]],
        initial_frame_idx: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Make the SAM2 predictor callable.

        Args:
            video_path: Input video path.
            prompts: Prompt list.
            initial_frame_idx: Initial prompt frame index.

        Returns:
            Video segmentation results.
        """
        return self.predict_with_prompts(
            video_path=video_path,
            initial_frame_idx=initial_frame_idx,
            prompts=prompts,
        )