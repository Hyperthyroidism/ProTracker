from typing import Any, Dict, List, Optional

import numpy as np


class Track:
    """
    Single vessel track.

    A Track stores the identity, latest bounding box, confidence, state,
    and historical records of one vessel target.
    """

    def __init__(
        self,
        track_id: int,
        detection: Dict[str, Any],
        frame_id: int,
    ) -> None:
        """
        Args:
            track_id: Unique identity number of this track.
            detection: Initial detection dictionary.
            frame_id: Current frame index.
        """
        self.track_id = int(track_id)
        self.start_frame = int(frame_id)
        self.last_frame = int(frame_id)

        self.bbox_xyxy = self._get_box_xyxy(detection)
        self.bbox_xywh = self._xyxy_to_xywh(self.bbox_xyxy)

        self.confidence = float(detection.get("confidence", 1.0))
        self.class_id = int(detection.get("class_id", 0))
        self.class_name = detection.get("class_name", "vessel")

        self.age = 1
        self.hits = 1
        self.time_since_update = 0
        self.is_active = True

        self.history: List[Dict[str, Any]] = []
        self.add_history(detection, frame_id)

    @staticmethod
    def _xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
        """
        Convert xyxy box to xywh box.

        Args:
            box: Bounding box in [x1, y1, x2, y2] format.

        Returns:
            Bounding box in [x, y, w, h] format.
        """
        x1, y1, x2, y2 = box
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    @staticmethod
    def _xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
        """
        Convert xywh box to xyxy box.

        Args:
            box: Bounding box in [x, y, w, h] format.

        Returns:
            Bounding box in [x1, y1, x2, y2] format.
        """
        x, y, w, h = box
        return np.array([x, y, x + w, y + h], dtype=np.float32)

    def _get_box_xyxy(self, detection: Dict[str, Any]) -> np.ndarray:
        """
        Get xyxy box from detection.

        Args:
            detection: Detection dictionary.

        Returns:
            Bounding box in xyxy format.
        """
        if "bbox_xyxy" in detection:
            return np.asarray(detection["bbox_xyxy"], dtype=np.float32)

        if "bbox_xywh" in detection:
            return self._xywh_to_xyxy(
                np.asarray(detection["bbox_xywh"], dtype=np.float32)
            )

        raise KeyError("Detection must contain bbox_xyxy or bbox_xywh.")

    def add_history(
        self,
        detection: Dict[str, Any],
        frame_id: int,
    ) -> None:
        """
        Add one historical record.

        Args:
            detection: Detection dictionary.
            frame_id: Current frame index.
        """
        box_xyxy = self._get_box_xyxy(detection)
        box_xywh = self._xyxy_to_xywh(box_xyxy)

        record = {
            "frame_id": int(frame_id),
            "track_id": self.track_id,
            "bbox_xyxy": box_xyxy,
            "bbox_xywh": box_xywh,
            "confidence": float(detection.get("confidence", self.confidence)),
            "class_id": int(detection.get("class_id", self.class_id)),
            "class_name": detection.get("class_name", self.class_name),
        }

        self.history.append(record)

    def update(
        self,
        detection: Dict[str, Any],
        frame_id: int,
    ) -> None:
        """
        Update this track with a new detection.

        Args:
            detection: Matched detection dictionary.
            frame_id: Current frame index.
        """
        self.bbox_xyxy = self._get_box_xyxy(detection)
        self.bbox_xywh = self._xyxy_to_xywh(self.bbox_xyxy)

        self.confidence = float(detection.get("confidence", self.confidence))
        self.class_id = int(detection.get("class_id", self.class_id))
        self.class_name = detection.get("class_name", self.class_name)

        self.last_frame = int(frame_id)
        self.age += 1
        self.hits += 1
        self.time_since_update = 0
        self.is_active = True

        self.add_history(detection, frame_id)

    def mark_missed(self) -> None:
        """
        Mark this track as not updated in the current frame.
        """
        self.age += 1
        self.time_since_update += 1

    def deactivate(self) -> None:
        """
        Deactivate this track.
        """
        self.is_active = False

    def predict_next_box(self) -> np.ndarray:
        """
        Predict next box using a simple linear motion model.

        Returns:
            Predicted bounding box in xyxy format.
        """
        if len(self.history) < 2:
            return self.bbox_xyxy.copy()

        last_box = np.asarray(self.history[-1]["bbox_xyxy"], dtype=np.float32)
        prev_box = np.asarray(self.history[-2]["bbox_xyxy"], dtype=np.float32)

        velocity = last_box - prev_box
        predicted_box = last_box + velocity

        return predicted_box.astype(np.float32)

    def to_detection(self) -> Dict[str, Any]:
        """
        Convert this track to a detection-like dictionary.

        Returns:
            Detection-like dictionary.
        """
        return {
            "frame_id": self.last_frame,
            "track_id": self.track_id,
            "bbox_xyxy": self.bbox_xyxy.copy(),
            "bbox_xywh": self.bbox_xywh.copy(),
            "confidence": self.confidence,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "source": "track",
        }


class TrackManager:
    """
    Track manager for ProTracker.

    This module maintains vessel trajectories during inference. It assigns
    track IDs to detections, updates matched tracks, creates new tracks, and
    removes long-lost tracks.
    """

    def __init__(
        self,
        association_threshold: float = 0.5,
        max_lost_frames: int = 30,
        min_track_length: int = 3,
    ) -> None:
        """
        Args:
            association_threshold: Matching threshold.
            max_lost_frames: Maximum number of frames a track can be lost.
            min_track_length: Minimum track length used for valid output.
        """
        self.association_threshold = association_threshold
        self.max_lost_frames = max_lost_frames
        self.min_track_length = min_track_length

        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1

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
    def get_detection_box(detection: Dict[str, Any]) -> np.ndarray:
        """
        Get xyxy box from detection.

        Args:
            detection: Detection dictionary.

        Returns:
            Bounding box in xyxy format.
        """
        if "bbox_xyxy" in detection:
            return np.asarray(detection["bbox_xyxy"], dtype=np.float32)

        if "bbox_xywh" in detection:
            x, y, w, h = detection["bbox_xywh"]
            return np.array([x, y, x + w, y + h], dtype=np.float32)

        raise KeyError("Detection must contain bbox_xyxy or bbox_xywh.")

    def get_active_tracks(self) -> List[Track]:
        """
        Get active tracks.

        Returns:
            Active track list.
        """
        return [
            track
            for track in self.tracks.values()
            if track.is_active
        ]

    def get_histories(self) -> Dict[int, List[Dict[str, Any]]]:
        """
        Get histories of all active tracks.

        Returns:
            Dictionary mapping track_id to track history.
        """
        histories = {}

        for track_id, track in self.tracks.items():
            if track.is_active:
                histories[track_id] = track.history

        return histories

    def create_track(
        self,
        detection: Dict[str, Any],
        frame_id: int,
    ) -> Track:
        """
        Create a new track.

        Args:
            detection: Detection dictionary.
            frame_id: Current frame index.

        Returns:
            Created Track object.
        """
        track = Track(
            track_id=self.next_track_id,
            detection=detection,
            frame_id=frame_id,
        )

        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

        return track

    def match_by_iou(
        self,
        detections: List[Dict[str, Any]],
    ) -> Dict[int, int]:
        """
        Match detections to active tracks using IoU.

        Args:
            detections: Detection list.

        Returns:
            Dictionary mapping detection index to track_id.
        """
        active_tracks = self.get_active_tracks()

        if len(active_tracks) == 0 or len(detections) == 0:
            return {}

        matched = {}
        used_track_ids = set()

        for det_idx, detection in enumerate(detections):
            det_box = self.get_detection_box(detection)

            best_track_id = None
            best_score = -1.0

            for track in active_tracks:
                if track.track_id in used_track_ids:
                    continue

                pred_box = track.predict_next_box()
                score = self.box_iou(det_box, pred_box)

                if score > best_score:
                    best_score = score
                    best_track_id = track.track_id

            if best_track_id is not None and best_score >= self.association_threshold:
                matched[det_idx] = best_track_id
                used_track_ids.add(best_track_id)

        return matched

    def assign_track_ids(
        self,
        detections: List[Dict[str, Any]],
        edge_output: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Assign track IDs to detections.

        Args:
            detections: Detection list.
            edge_output: Optional graph output from GraphPromptOptimizer.

        Returns:
            Detection list with track_id.
        """
        if len(detections) == 0:
            return []

        # At this stage, IoU matching is used as a stable default.
        # The graph edge scores can be further integrated in association.py.
        matched = self.match_by_iou(detections)

        assigned_detections = []

        for det_idx, detection in enumerate(detections):
            new_det = dict(detection)

            if det_idx in matched:
                track_id = matched[det_idx]
            else:
                track_id = self.next_track_id

            new_det["track_id"] = int(track_id)
            assigned_detections.append(new_det)

        return assigned_detections

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_id: int,
        edge_output: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Update track manager with current detections.

        Args:
            detections: Current detection list.
            frame_id: Current frame index.
            edge_output: Optional graph output.

        Returns:
            Detection list with assigned track IDs.
        """
        assigned_detections = self.assign_track_ids(
            detections=detections,
            edge_output=edge_output,
        )

        updated_track_ids = set()

        for detection in assigned_detections:
            track_id = int(detection["track_id"])

            if track_id in self.tracks:
                self.tracks[track_id].update(detection, frame_id)
            else:
                track = Track(track_id, detection, frame_id)
                self.tracks[track_id] = track

                if track_id >= self.next_track_id:
                    self.next_track_id = track_id + 1

            updated_track_ids.add(track_id)

        for track_id, track in list(self.tracks.items()):
            if track_id not in updated_track_ids:
                track.mark_missed()

                if track.time_since_update > self.max_lost_frames:
                    track.deactivate()

        return assigned_detections

    def get_valid_tracks(self) -> List[Track]:
        """
        Get valid tracks for output.

        Returns:
            Track list whose length is larger than min_track_length.
        """
        valid_tracks = []

        for track in self.tracks.values():
            if len(track.history) >= self.min_track_length:
                valid_tracks.append(track)

        return valid_tracks

    def get_current_results(self) -> List[Dict[str, Any]]:
        """
        Get current active tracking results.

        Returns:
            Detection-like tracking results.
        """
        results = []

        for track in self.get_active_tracks():
            if track.time_since_update == 0:
                results.append(track.to_detection())

        return results

    def reset(self) -> None:
        """
        Reset all tracks.
        """
        self.tracks.clear()
        self.next_track_id = 1