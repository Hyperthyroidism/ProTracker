from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class TargetRefinementModule:
    """
    Target-aware refinement module for ProTracker.

    This module refines vessel prompts before they are sent to SAM2.
    It is mainly used to handle small vessels, partially occluded vessels,
    inaccurate detection boxes, and weak target candidates.
    """

    def __init__(
        self,
        min_box_area: float = 16.0,
        expand_ratio: float = 1.15,
        refine_small_targets: bool = True,
        refine_occluded_targets: bool = True,
        supplement_missing_targets: bool = True,
    ) -> None:
        """
        Args:
            min_box_area: Minimum bounding box area. Boxes smaller than this value
                          will be treated as weak or small targets.
            expand_ratio: Ratio used to slightly enlarge prompt boxes.
            refine_small_targets: Whether to refine small vessel prompts.
            refine_occluded_targets: Whether to refine occluded vessel prompts.
            supplement_missing_targets: Whether to supplement missing targets
                                        using track history.
        """
        self.min_box_area = min_box_area
        self.expand_ratio = expand_ratio
        self.refine_small_targets = refine_small_targets
        self.refine_occluded_targets = refine_occluded_targets
        self.supplement_missing_targets = supplement_missing_targets

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
    def box_area(box_xyxy: np.ndarray) -> float:
        """
        Calculate box area.

        Args:
            box_xyxy: Bounding box in [x1, y1, x2, y2] format.

        Returns:
            Box area.
        """
        x1, y1, x2, y2 = box_xyxy
        w = max(0.0, float(x2 - x1))
        h = max(0.0, float(y2 - y1))
        return w * h

    @staticmethod
    def clip_box(
        box_xyxy: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Clip bounding box into image boundary.

        Args:
            box_xyxy: Bounding box in [x1, y1, x2, y2] format.
            image_shape: Image shape in (height, width) format.

        Returns:
            Clipped bounding box.
        """
        box_xyxy = box_xyxy.astype(np.float32)

        if image_shape is None:
            return box_xyxy

        height, width = image_shape

        box_xyxy[0] = np.clip(box_xyxy[0], 0, width - 1)
        box_xyxy[1] = np.clip(box_xyxy[1], 0, height - 1)
        box_xyxy[2] = np.clip(box_xyxy[2], 0, width - 1)
        box_xyxy[3] = np.clip(box_xyxy[3], 0, height - 1)

        return box_xyxy

    def expand_box(
        self,
        box_xyxy: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
        ratio: Optional[float] = None,
    ) -> np.ndarray:
        """
        Expand bounding box around its center.

        Args:
            box_xyxy: Bounding box in [x1, y1, x2, y2] format.
            image_shape: Image shape in (height, width) format.
            ratio: Expansion ratio. If None, self.expand_ratio is used.

        Returns:
            Expanded bounding box.
        """
        if ratio is None:
            ratio = self.expand_ratio

        x1, y1, x2, y2 = box_xyxy.astype(np.float32)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        new_w = w * ratio
        new_h = h * ratio

        refined_box = np.array(
            [
                cx - new_w / 2.0,
                cy - new_h / 2.0,
                cx + new_w / 2.0,
                cy + new_h / 2.0,
            ],
            dtype=np.float32,
        )

        refined_box = self.clip_box(refined_box, image_shape)
        return refined_box

    def refine_small_box(
        self,
        box_xyxy: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Refine small vessel box.

        Small vessels usually have incomplete or unstable detection boxes.
        This function slightly enlarges small boxes to provide more context
        for SAM2 prompt-based segmentation.

        Args:
            box_xyxy: Bounding box in xyxy format.
            image_shape: Image shape in (height, width) format.

        Returns:
            Refined bounding box.
        """
        area = self.box_area(box_xyxy)

        if area < self.min_box_area:
            refined_box = self.expand_box(
                box_xyxy,
                image_shape=image_shape,
                ratio=max(self.expand_ratio, 1.30),
            )
        else:
            refined_box = self.expand_box(
                box_xyxy,
                image_shape=image_shape,
                ratio=self.expand_ratio,
            )

        return refined_box

    def refine_occluded_box(
        self,
        box_xyxy: np.ndarray,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Refine occluded vessel box.

        For partially occluded vessels, a slightly larger prompt box can help
        SAM2 recover more complete vessel regions.

        Args:
            box_xyxy: Bounding box in xyxy format.
            image_shape: Image shape in (height, width) format.

        Returns:
            Refined bounding box.
        """
        refined_box = self.expand_box(
            box_xyxy,
            image_shape=image_shape,
            ratio=max(self.expand_ratio, 1.20),
        )

        return refined_box

    @staticmethod
    def estimate_box_from_history(
        track_history: List[Dict[str, Any]],
    ) -> Optional[np.ndarray]:
        """
        Estimate a missing target box from track history.

        This is a simple motion-based estimation. It uses the last two boxes
        of a track to predict the next position.

        Args:
            track_history: Historical records of one track. Each element should
                           contain "bbox_xyxy".

        Returns:
            Estimated bounding box in xyxy format. If history is not enough,
            return None.
        """
        if track_history is None or len(track_history) == 0:
            return None

        if len(track_history) == 1:
            last_box = track_history[-1].get("bbox_xyxy", None)
            if last_box is None:
                return None
            return np.asarray(last_box, dtype=np.float32)

        last_box = np.asarray(track_history[-1].get("bbox_xyxy"), dtype=np.float32)
        prev_box = np.asarray(track_history[-2].get("bbox_xyxy"), dtype=np.float32)

        velocity = last_box - prev_box
        estimated_box = last_box + velocity

        return estimated_box.astype(np.float32)

    def supplement_missing_prompts(
        self,
        prompts: List[Dict[str, Any]],
        track_histories: Optional[Dict[int, List[Dict[str, Any]]]] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Supplement missing prompts according to track histories.

        Args:
            prompts: Current prompt list.
            track_histories: Dictionary mapping track id to historical records.
            image_shape: Image shape in (height, width) format.

        Returns:
            Prompt list with supplemented prompts.
        """
        if not self.supplement_missing_targets:
            return prompts

        if track_histories is None:
            return prompts

        existing_ids = set()

        for prompt in prompts:
            if "track_id" in prompt:
                existing_ids.add(int(prompt["track_id"]))

        supplemented_prompts = list(prompts)

        for track_id, history in track_histories.items():
            if int(track_id) in existing_ids:
                continue

            estimated_box = self.estimate_box_from_history(history)

            if estimated_box is None:
                continue

            estimated_box = self.clip_box(estimated_box, image_shape)
            estimated_box = self.expand_box(estimated_box, image_shape=image_shape)

            new_prompt = {
                "track_id": int(track_id),
                "bbox_xyxy": estimated_box,
                "bbox_xywh": self.xyxy_to_xywh(estimated_box),
                "confidence": 0.30,
                "class_id": 0,
                "class_name": "vessel",
                "source": "history_supplement",
            }

            supplemented_prompts.append(new_prompt)

        return supplemented_prompts

    def refine_prompt(
        self,
        prompt: Dict[str, Any],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        Refine a single vessel prompt.

        Args:
            prompt: Prompt dictionary.
            image_shape: Image shape in (height, width) format.

        Returns:
            Refined prompt dictionary.
        """
        refined_prompt = dict(prompt)

        if "bbox_xyxy" in refined_prompt:
            box_xyxy = np.asarray(refined_prompt["bbox_xyxy"], dtype=np.float32)
        elif "bbox_xywh" in refined_prompt:
            box_xyxy = self.xywh_to_xyxy(
                np.asarray(refined_prompt["bbox_xywh"], dtype=np.float32)
            )
        else:
            return refined_prompt

        area = self.box_area(box_xyxy)

        if self.refine_small_targets and area < self.min_box_area:
            box_xyxy = self.refine_small_box(box_xyxy, image_shape=image_shape)
            refined_prompt["refine_type"] = "small_target"

        elif self.refine_occluded_targets and refined_prompt.get("occluded", False):
            box_xyxy = self.refine_occluded_box(box_xyxy, image_shape=image_shape)
            refined_prompt["refine_type"] = "occluded_target"

        else:
            box_xyxy = self.expand_box(box_xyxy, image_shape=image_shape)
            refined_prompt["refine_type"] = "normal_expand"

        refined_prompt["bbox_xyxy"] = box_xyxy
        refined_prompt["bbox_xywh"] = self.xyxy_to_xywh(box_xyxy)
        refined_prompt["source"] = refined_prompt.get("source", "target_refinement")

        return refined_prompt

    def refine_prompts(
        self,
        prompts: List[Dict[str, Any]],
        image_shape: Optional[Tuple[int, int]] = None,
        track_histories: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Refine a list of vessel prompts.

        Args:
            prompts: Prompt list.
            image_shape: Image shape in (height, width) format.
            track_histories: Track histories used to supplement missing prompts.

        Returns:
            Refined prompt list.
        """
        refined_prompts = []

        for prompt in prompts:
            refined_prompt = self.refine_prompt(
                prompt=prompt,
                image_shape=image_shape,
            )
            refined_prompts.append(refined_prompt)

        refined_prompts = self.supplement_missing_prompts(
            prompts=refined_prompts,
            track_histories=track_histories,
            image_shape=image_shape,
        )

        return refined_prompts

    def __call__(
        self,
        prompts: List[Dict[str, Any]],
        image_shape: Optional[Tuple[int, int]] = None,
        track_histories: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Make the refinement module callable.

        Args:
            prompts: Prompt list.
            image_shape: Image shape in (height, width) format.
            track_histories: Track histories used to supplement missing prompts.

        Returns:
            Refined prompt list.
        """
        return self.refine_prompts(
            prompts=prompts,
            image_shape=image_shape,
            track_histories=track_histories,
        )