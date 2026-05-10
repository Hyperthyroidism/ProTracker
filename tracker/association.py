from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class AssociationModule:
    """
    Association module for ProTracker.

    This module matches current vessel detections with existing tracks.
    It provides a simple and stable association interface that can combine:

        1. IoU similarity
        2. Center distance similarity
        3. Detection confidence
        4. Optional graph edge scores from GraphPromptOptimizer

    In the complete ProTracker framework, the graph-based prompt optimizer
    provides stronger association information, while this module provides
    a unified matching interface for track management.
    """

    def __init__(
        self,
        association_threshold: float = 0.5,
        iou_weight: float = 0.5,
        distance_weight: float = 0.3,
        confidence_weight: float = 0.2,
        graph_score_weight: float = 0.5,
        max_center_distance: float = 100.0,
    ) -> None:
        """
        Args:
            association_threshold: Minimum score required for a valid match.
            iou_weight: Weight of IoU similarity.
            distance_weight: Weight of center-distance similarity.
            confidence_weight: Weight of detection confidence.
            graph_score_weight: Weight of graph association score.
            max_center_distance: Maximum center distance used for normalization.
        """
        self.association_threshold = association_threshold
        self.iou_weight = iou_weight
        self.distance_weight = distance_weight
        self.confidence_weight = confidence_weight
        self.graph_score_weight = graph_score_weight
        self.max_center_distance = max_center_distance

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
    def get_box_xyxy(item: Dict[str, Any]) -> np.ndarray:
        """
        Get xyxy bounding box from a detection or track-like dictionary.

        Args:
            item: Dictionary containing bbox_xyxy or bbox_xywh.

        Returns:
            Bounding box in xyxy format.
        """
        if "bbox_xyxy" in item:
            return np.asarray(item["bbox_xyxy"], dtype=np.float32)

        if "bbox_xywh" in item:
            return AssociationModule.xywh_to_xyxy(
                np.asarray(item["bbox_xywh"], dtype=np.float32)
            )

        raise KeyError("Input item must contain bbox_xyxy or bbox_xywh.")

    @staticmethod
    def box_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """
        Calculate IoU between two boxes.

        Args:
            box_a: Bounding box in xyxy format.
            box_b: Bounding box in xyxy format.

        Returns:
            IoU value.
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union = area_a + area_b - inter_area

        if union <= 0:
            return 0.0

        return float(inter_area / union)

    @staticmethod
    def box_center(box: np.ndarray) -> Tuple[float, float]:
        """
        Calculate the center of a bounding box.

        Args:
            box: Bounding box in xyxy format.

        Returns:
            Center coordinates.
        """
        x1, y1, x2, y2 = box
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

    def center_distance_similarity(
        self,
        box_a: np.ndarray,
        box_b: np.ndarray,
    ) -> float:
        """
        Calculate center-distance similarity.

        Args:
            box_a: First bounding box in xyxy format.
            box_b: Second bounding box in xyxy format.

        Returns:
            Similarity score in [0, 1].
        """
        ax, ay = self.box_center(box_a)
        bx, by = self.box_center(box_b)

        distance = np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

        similarity = 1.0 - min(distance / max(self.max_center_distance, 1e-6), 1.0)

        return float(similarity)

    def get_graph_score(
        self,
        det_idx: int,
        track_idx: int,
        edge_output: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """
        Get graph association score from GraphPromptOptimizer output.

        Args:
            det_idx: Detection index.
            track_idx: Track index.
            edge_output: Output dictionary from graph network.

        Returns:
            Graph association score if available, otherwise None.
        """
        if edge_output is None:
            return None

        if "association_scores" in edge_output:
            scores = edge_output["association_scores"]
            try:
                return float(scores[det_idx][track_idx])
            except Exception:
                return None

        if "edge_scores" in edge_output and "edge_index" in edge_output:
            edge_scores = edge_output["edge_scores"]
            edge_index = edge_output["edge_index"]

            try:
                if hasattr(edge_scores, "detach"):
                    edge_scores = edge_scores.detach().cpu().numpy()

                if hasattr(edge_index, "detach"):
                    edge_index = edge_index.detach().cpu().numpy()

                edge_scores = np.asarray(edge_scores).reshape(-1)
                edge_index = np.asarray(edge_index)

                for edge_id in range(edge_index.shape[1]):
                    src = int(edge_index[0, edge_id])
                    dst = int(edge_index[1, edge_id])

                    if src == det_idx and dst == track_idx:
                        return float(edge_scores[edge_id])
            except Exception:
                return None

        return None

    def compute_pair_score(
        self,
        detection: Dict[str, Any],
        track_item: Dict[str, Any],
        det_idx: int = 0,
        track_idx: int = 0,
        edge_output: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Compute association score between one detection and one track.

        Args:
            detection: Current detection dictionary.
            track_item: Existing track dictionary.
            det_idx: Detection index.
            track_idx: Track index.
            edge_output: Optional graph output.

        Returns:
            Association score.
        """
        det_box = self.get_box_xyxy(detection)
        track_box = self.get_box_xyxy(track_item)

        iou_score = self.box_iou(det_box, track_box)
        distance_score = self.center_distance_similarity(det_box, track_box)
        confidence_score = float(detection.get("confidence", 1.0))

        base_score = (
            self.iou_weight * iou_score
            + self.distance_weight * distance_score
            + self.confidence_weight * confidence_score
        )

        graph_score = self.get_graph_score(
            det_idx=det_idx,
            track_idx=track_idx,
            edge_output=edge_output,
        )

        if graph_score is not None:
            score = (
                (1.0 - self.graph_score_weight) * base_score
                + self.graph_score_weight * graph_score
            )
        else:
            score = base_score

        return float(score)

    def build_score_matrix(
        self,
        detections: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]],
        edge_output: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Build association score matrix.

        Args:
            detections: Current detection list.
            tracks: Existing track list.
            edge_output: Optional graph output.

        Returns:
            Score matrix with shape [num_detections, num_tracks].
        """
        num_dets = len(detections)
        num_tracks = len(tracks)

        score_matrix = np.zeros((num_dets, num_tracks), dtype=np.float32)

        for det_idx, detection in enumerate(detections):
            for track_idx, track_item in enumerate(tracks):
                score = self.compute_pair_score(
                    detection=detection,
                    track_item=track_item,
                    det_idx=det_idx,
                    track_idx=track_idx,
                    edge_output=edge_output,
                )
                score_matrix[det_idx, track_idx] = score

        return score_matrix

    def greedy_match(
        self,
        score_matrix: np.ndarray,
    ) -> List[Tuple[int, int]]:
        """
        Greedy matching based on association scores.

        Args:
            score_matrix: Score matrix with shape [num_detections, num_tracks].

        Returns:
            List of matched pairs: (detection_index, track_index).
        """
        if score_matrix.size == 0:
            return []

        matched_pairs = []
        used_dets = set()
        used_tracks = set()

        candidates = []

        num_dets, num_tracks = score_matrix.shape

        for det_idx in range(num_dets):
            for track_idx in range(num_tracks):
                score = float(score_matrix[det_idx, track_idx])

                if score >= self.association_threshold:
                    candidates.append((score, det_idx, track_idx))

        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

        for score, det_idx, track_idx in candidates:
            if det_idx in used_dets or track_idx in used_tracks:
                continue

            matched_pairs.append((det_idx, track_idx))
            used_dets.add(det_idx)
            used_tracks.add(track_idx)

        return matched_pairs

    def associate(
        self,
        detections: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]],
        edge_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Associate current detections with existing tracks.

        Args:
            detections: Current detection list.
            tracks: Existing track list.
            edge_output: Optional graph output.

        Returns:
            Association result dictionary.
        """
        score_matrix = self.build_score_matrix(
            detections=detections,
            tracks=tracks,
            edge_output=edge_output,
        )

        matched_pairs = self.greedy_match(score_matrix)

        matched_det_indices = {pair[0] for pair in matched_pairs}
        matched_track_indices = {pair[1] for pair in matched_pairs}

        unmatched_detections = [
            idx for idx in range(len(detections))
            if idx not in matched_det_indices
        ]

        unmatched_tracks = [
            idx for idx in range(len(tracks))
            if idx not in matched_track_indices
        ]

        result = {
            "matches": matched_pairs,
            "unmatched_detections": unmatched_detections,
            "unmatched_tracks": unmatched_tracks,
            "score_matrix": score_matrix,
        }

        return result

    def __call__(
        self,
        detections: List[Dict[str, Any]],
        tracks: List[Dict[str, Any]],
        edge_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make the association module callable.

        Args:
            detections: Current detection list.
            tracks: Existing track list.
            edge_output: Optional graph output.

        Returns:
            Association result dictionary.
        """
        return self.associate(
            detections=detections,
            tracks=tracks,
            edge_output=edge_output,
        )