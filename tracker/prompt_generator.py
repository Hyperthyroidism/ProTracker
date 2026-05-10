from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class PromptGenerator:
    """
    Prompt generator for ProTracker.

    This module converts detections, tracks, and graph-optimized results into
    SAM2-compatible prompts. In ProTracker, the prompt generator acts as the
    bridge between the graph-based cascaded prompt optimizer and the SAM2-based
    video segmentation module.
    """

    def __init__(
        self,
        prompt_type: str = "box",
        use_mask_prompt: bool = True,
        point_mode: str = "center",
    ) -> None:
        """
        Args:
            prompt_type: Prompt type used for SAM2. Supported values:
                         "box", "point", and "mixed".
            use_mask_prompt: Whether to keep mask prompt if it is available.
            point_mode: Method for generating point prompts. Currently supports
                        "center".
        """
        self.prompt_type = prompt_type
        self.use_mask_prompt = use_mask_prompt
        self.point_mode = point_mode

    @staticmethod
    def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
        """
        Convert box from xywh to xyxy.

        Args:
            box: Bounding box in [x, y, w, h] format.

        Returns:
            Bounding box in [x1, y1, x2, y2] format.
        """
        x, y, w, h = box
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    @staticmethod
    def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
        """
        Convert box from xyxy to xywh.

        Args:
            box: Bounding box in [x1, y1, x2, y2] format.

        Returns:
            Bounding box in [x, y, w, h] format.
        """
        x1, y1, x2, y2 = box
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    @staticmethod
    def box_center(box_xyxy: np.ndarray) -> np.ndarray:
        """
        Calculate the center point of a bounding box.

        Args:
            box_xyxy: Bounding box in xyxy format.

        Returns:
            Center point with shape [1, 2].
        """
        x1, y1, x2, y2 = box_xyxy
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        return np.array([[cx, cy]], dtype=np.float32)

    @staticmethod
    def clip_box(
        box_xyxy: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Clip box into image boundary.

        Args:
            box_xyxy: Bounding box in xyxy format.
            image_shape: Image shape in (height, width) format.

        Returns:
            Clipped bounding box.
        """
        box_xyxy = np.asarray(box_xyxy, dtype=np.float32)

        if image_shape is None:
            return box_xyxy

        height, width = image_shape

        box_xyxy[0] = np.clip(box_xyxy[0], 0, width - 1)
        box_xyxy[1] = np.clip(box_xyxy[1], 0, height - 1)
        box_xyxy[2] = np.clip(box_xyxy[2], 0, width - 1)
        box_xyxy[3] = np.clip(box_xyxy[3], 0, height - 1)

        return box_xyxy

    def get_box_from_item(
        self,
        item: Dict[str, Any],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Get xyxy box from detection, track, or optimized prompt.

        Args:
            item: Input dictionary.
            image_shape: Image shape in (height, width) format.

        Returns:
            Bounding box in xyxy format. If no valid box exists, return None.
        """
        if "bbox_xyxy" in item:
            box_xyxy = np.asarray(item["bbox_xyxy"], dtype=np.float32)
        elif "bbox_xywh" in item:
            box_xyxy = self.xywh_to_xyxy(
                np.asarray(item["bbox_xywh"], dtype=np.float32)
            )
        else:
            return None

        box_xyxy = self.clip_box(box_xyxy, image_shape)

        return box_xyxy

    def generate_box_prompt(
        self,
        item: Dict[str, Any],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a box prompt for SAM2.

        Args:
            item: Detection, track, or optimized result dictionary.
            image_shape: Image shape in (height, width) format.

        Returns:
            Prompt dictionary.
        """
        box_xyxy = self.get_box_from_item(item, image_shape)

        if box_xyxy is None:
            return None

        prompt = {
            "frame_id": int(item.get("frame_id", 0)),
            "track_id": int(item.get("track_id", item.get("object_id", -1))),
            "bbox_xyxy": box_xyxy,
            "bbox_xywh": self.xyxy_to_xywh(box_xyxy),
            "confidence": float(item.get("confidence", 1.0)),
            "class_id": int(item.get("class_id", 0)),
            "class_name": item.get("class_name", "vessel"),
            "prompt_type": "box",
            "source": item.get("source", "prompt_generator"),
        }

        return prompt

    def generate_point_prompt(
        self,
        item: Dict[str, Any],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a point prompt for SAM2.

        Args:
            item: Detection, track, or optimized result dictionary.
            image_shape: Image shape in (height, width) format.

        Returns:
            Prompt dictionary.
        """
        box_xyxy = self.get_box_from_item(item, image_shape)

        if box_xyxy is None:
            return None

        if self.point_mode == "center":
            points = self.box_center(box_xyxy)
        else:
            points = self.box_center(box_xyxy)

        point_labels = np.ones((points.shape[0],), dtype=np.int32)

        prompt = {
            "frame_id": int(item.get("frame_id", 0)),
            "track_id": int(item.get("track_id", item.get("object_id", -1))),
            "bbox_xyxy": box_xyxy,
            "bbox_xywh": self.xyxy_to_xywh(box_xyxy),
            "points": points,
            "point_labels": point_labels,
            "confidence": float(item.get("confidence", 1.0)),
            "class_id": int(item.get("class_id", 0)),
            "class_name": item.get("class_name", "vessel"),
            "prompt_type": "point",
            "source": item.get("source", "prompt_generator"),
        }

        return prompt

    def generate_mask_prompt(
        self,
        item: Dict[str, Any],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a mask prompt if a mask is available.

        Args:
            item: Input dictionary that may contain mask.
            image_shape: Image shape in (height, width) format.

        Returns:
            Prompt dictionary.
        """
        if "mask" not in item:
            return None

        box_xyxy = self.get_box_from_item(item, image_shape)

        prompt = {
            "frame_id": int(item.get("frame_id", 0)),
            "track_id": int(item.get("track_id", item.get("object_id", -1))),
            "mask": np.asarray(item["mask"]).astype(np.uint8),
            "confidence": float(item.get("confidence", 1.0)),
            "class_id": int(item.get("class_id", 0)),
            "class_name": item.get("class_name", "vessel"),
            "prompt_type": "mask",
            "source": item.get("source", "prompt_generator"),
        }

        if box_xyxy is not None:
            prompt["bbox_xyxy"] = box_xyxy
            prompt["bbox_xywh"] = self.xyxy_to_xywh(box_xyxy)

        return prompt

    def generate_prompt(
        self,
        item: Dict[str, Any],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Generate one SAM2-compatible prompt.

        Args:
            item: Detection, track, or graph-optimized result dictionary.
            image_shape: Image shape in (height, width) format.

        Returns:
            Prompt dictionary.
        """
        if self.use_mask_prompt and "mask" in item:
            mask_prompt = self.generate_mask_prompt(
                item=item,
                image_shape=image_shape,
            )

            if mask_prompt is not None:
                return mask_prompt

        if self.prompt_type == "box":
            return self.generate_box_prompt(
                item=item,
                image_shape=image_shape,
            )

        if self.prompt_type == "point":
            return self.generate_point_prompt(
                item=item,
                image_shape=image_shape,
            )

        if self.prompt_type == "mixed":
            box_prompt = self.generate_box_prompt(
                item=item,
                image_shape=image_shape,
            )

            point_prompt = self.generate_point_prompt(
                item=item,
                image_shape=image_shape,
            )

            if box_prompt is None and point_prompt is None:
                return None

            if box_prompt is None:
                return point_prompt

            if point_prompt is not None:
                box_prompt["points"] = point_prompt["points"]
                box_prompt["point_labels"] = point_prompt["point_labels"]
                box_prompt["prompt_type"] = "mixed"

            return box_prompt

        return self.generate_box_prompt(
            item=item,
            image_shape=image_shape,
        )

    def generate_prompts(
        self,
        items: List[Dict[str, Any]],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate prompt list.

        Args:
            items: Detection, track, or graph-optimized result list.
            image_shape: Image shape in (height, width) format.

        Returns:
            Prompt list.
        """
        prompts = []

        for item in items:
            prompt = self.generate_prompt(
                item=item,
                image_shape=image_shape,
            )

            if prompt is None:
                continue

            if prompt.get("track_id", -1) < 0:
                continue

            prompts.append(prompt)

        return prompts

    def merge_prompts(
        self,
        base_prompts: List[Dict[str, Any]],
        refined_prompts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Merge base prompts and refined prompts.

        Args:
            base_prompts: Original prompt list.
            refined_prompts: Refined prompt list.

        Returns:
            Merged prompt list.
        """
        if refined_prompts is None:
            return base_prompts

        prompt_dict = {}

        for prompt in base_prompts:
            track_id = int(prompt.get("track_id", -1))
            if track_id >= 0:
                prompt_dict[track_id] = prompt

        for prompt in refined_prompts:
            track_id = int(prompt.get("track_id", -1))
            if track_id >= 0:
                prompt_dict[track_id] = prompt

        merged_prompts = list(prompt_dict.values())
        merged_prompts = sorted(
            merged_prompts,
            key=lambda x: int(x.get("track_id", 0)),
        )

        return merged_prompts

    def __call__(
        self,
        items: List[Dict[str, Any]],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Make the prompt generator callable.

        Args:
            items: Detection, track, or graph-optimized result list.
            image_shape: Image shape in (height, width) format.

        Returns:
            Prompt list.
        """
        return self.generate_prompts(
            items=items,
            image_shape=image_shape,
        )