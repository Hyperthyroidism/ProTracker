from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from networks.motmpnet import MOTMPNet


class GraphPromptOptimizer(nn.Module):
    """
    Graph-based Cascaded Prompt Optimizer for ProTracker.

    This module converts vessel detections or tracklets into a graph structure,
    applies a message-passing graph neural network to refine association
    relationships, and generates optimized vessel prompts for SAM2.

    In the graph:

        nodes represent vessel detections or tracklets
        edges represent possible temporal association hypotheses
        node features describe vessel appearance, position, and confidence
        edge features describe spatial, temporal, motion, and similarity cues
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Args:
            config: Configuration dictionary loaded from yaml files.
        """
        super().__init__()

        self.config = config

        gpo_cfg = config.get("graph_prompt_optimizer", {})
        graph_cfg = gpo_cfg.get("graph", {})

        self.node_feature_dim = graph_cfg.get("node_feature_dim", 256)
        self.edge_feature_dim = graph_cfg.get("edge_feature_dim", 32)
        self.hidden_dim = graph_cfg.get("hidden_dim", 128)

        self.max_temporal_distance = graph_cfg.get("max_temporal_distance", 30)
        self.max_neighbors = graph_cfg.get("max_neighbors", 10)

        self.use_motion_feature = graph_cfg.get("use_motion_feature", True)
        self.use_position_feature = graph_cfg.get("use_position_feature", True)
        self.use_appearance_feature = graph_cfg.get("use_appearance_feature", True)
        self.use_reid_feature = graph_cfg.get("use_reid_feature", False)

        self.association_threshold = config.get("tracking", {}).get(
            "association_threshold",
            0.5,
        )

        self.mpn = MOTMPNet(config)

    @staticmethod
    def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
        """
        Convert box from xyxy format to xywh format.

        Args:
            box: Bounding box in [x1, y1, x2, y2].

        Returns:
            Bounding box in [x, y, w, h].
        """
        x1, y1, x2, y2 = box
        return np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

    @staticmethod
    def xywh_to_xyxy(box: np.ndarray) -> np.ndarray:
        """
        Convert box from xywh format to xyxy format.

        Args:
            box: Bounding box in [x, y, w, h].

        Returns:
            Bounding box in [x1, y1, x2, y2].
        """
        x, y, w, h = box
        return np.array([x, y, x + w, y + h], dtype=np.float32)

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
    def box_center(box_xyxy: np.ndarray) -> Tuple[float, float]:
        """
        Calculate box center.

        Args:
            box_xyxy: Bounding box in xyxy format.

        Returns:
            Center point.
        """
        x1, y1, x2, y2 = box_xyxy
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

    def encode_node_feature(
        self,
        detection: Dict[str, Any],
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        Encode one vessel detection into a fixed-length node feature.

        Args:
            detection: Detection dictionary from YOLODetector.
            image_shape: Image shape in (height, width) format.

        Returns:
            Node feature vector with shape [node_feature_dim].
        """
        feature = np.zeros((self.node_feature_dim,), dtype=np.float32)

        if "bbox_xyxy" in detection:
            box_xyxy = np.asarray(detection["bbox_xyxy"], dtype=np.float32)
        elif "bbox_xywh" in detection:
            box_xyxy = self.xywh_to_xyxy(
                np.asarray(detection["bbox_xywh"], dtype=np.float32)
            )
        else:
            return feature

        x1, y1, x2, y2 = box_xyxy
        cx, cy = self.box_center(box_xyxy)
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        area = w * h

        if image_shape is not None:
            img_h, img_w = image_shape
            img_w = max(1, img_w)
            img_h = max(1, img_h)

            normalized_values = [
                x1 / img_w,
                y1 / img_h,
                x2 / img_w,
                y2 / img_h,
                cx / img_w,
                cy / img_h,
                w / img_w,
                h / img_h,
                area / (img_w * img_h),
            ]
        else:
            normalized_values = [
                x1,
                y1,
                x2,
                y2,
                cx,
                cy,
                w,
                h,
                area,
            ]

        confidence = float(detection.get("confidence", 1.0))
        class_id = float(detection.get("class_id", 0))

        base_values = normalized_values + [confidence, class_id]
        base_values = np.asarray(base_values, dtype=np.float32)

        feature[: len(base_values)] = base_values

        appearance = detection.get("appearance_feature", None)

        if appearance is not None and self.use_appearance_feature:
            appearance = np.asarray(appearance, dtype=np.float32)
            length = min(len(appearance), self.node_feature_dim - len(base_values))
            feature[len(base_values): len(base_values) + length] = appearance[:length]

        return feature

    def encode_edge_feature(
        self,
        det_i: Dict[str, Any],
        det_j: Dict[str, Any],
        frame_id: int = 0,
    ) -> np.ndarray:
        """
        Encode the relationship between two vessel candidates.

        Args:
            det_i: First detection dictionary.
            det_j: Second detection dictionary.
            frame_id: Current frame index.

        Returns:
            Edge feature vector with shape [edge_feature_dim].
        """
        feature = np.zeros((self.edge_feature_dim,), dtype=np.float32)

        if "bbox_xyxy" in det_i:
            box_i = np.asarray(det_i["bbox_xyxy"], dtype=np.float32)
        else:
            box_i = self.xywh_to_xyxy(np.asarray(det_i["bbox_xywh"], dtype=np.float32))

        if "bbox_xyxy" in det_j:
            box_j = np.asarray(det_j["bbox_xyxy"], dtype=np.float32)
        else:
            box_j = self.xywh_to_xyxy(np.asarray(det_j["bbox_xywh"], dtype=np.float32))

        cx_i, cy_i = self.box_center(box_i)
        cx_j, cy_j = self.box_center(box_j)

        w_i = max(1.0, box_i[2] - box_i[0])
        h_i = max(1.0, box_i[3] - box_i[1])
        w_j = max(1.0, box_j[2] - box_j[0])
        h_j = max(1.0, box_j[3] - box_j[1])

        dx = cx_j - cx_i
        dy = cy_j - cy_i
        distance = np.sqrt(dx * dx + dy * dy)

        iou = self.box_iou(box_i, box_j)
        area_i = w_i * h_i
        area_j = w_j * h_j
        area_ratio = area_j / max(area_i, 1.0)

        frame_i = int(det_i.get("frame_id", frame_id))
        frame_j = int(det_j.get("frame_id", frame_id))
        temporal_distance = abs(frame_j - frame_i)

        confidence_i = float(det_i.get("confidence", 1.0))
        confidence_j = float(det_j.get("confidence", 1.0))

        values = [
            dx,
            dy,
            distance,
            iou,
            w_i,
            h_i,
            w_j,
            h_j,
            area_ratio,
            temporal_distance,
            confidence_i,
            confidence_j,
        ]

        values = np.asarray(values, dtype=np.float32)
        feature[: len(values)] = values[: self.edge_feature_dim]

        return feature

    def build_graph(
        self,
        detections: List[Dict[str, Any]],
        frame_id: int = 0,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Build a graph from current detections.

        Args:
            detections: Detection list.
            frame_id: Current frame index.
            image_shape: Image shape in (height, width) format.

        Returns:
            Graph dictionary containing x, edge_index, edge_attr.
        """
        num_nodes = len(detections)

        if num_nodes == 0:
            return None

        node_features = [
            self.encode_node_feature(det, image_shape=image_shape)
            for det in detections
        ]

        node_features = np.stack(node_features, axis=0)

        edge_indices = []
        edge_features = []

        for i in range(num_nodes):
            candidate_edges = []

            for j in range(num_nodes):
                if i == j:
                    continue

                edge_feature = self.encode_edge_feature(
                    detections[i],
                    detections[j],
                    frame_id=frame_id,
                )

                distance = edge_feature[2]
                candidate_edges.append((distance, i, j, edge_feature))

            candidate_edges = sorted(candidate_edges, key=lambda x: x[0])
            candidate_edges = candidate_edges[: self.max_neighbors]

            for _, src, dst, edge_feature in candidate_edges:
                edge_indices.append([src, dst])
                edge_features.append(edge_feature)

        if len(edge_indices) == 0:
            edge_indices = [[0, 0]]
            edge_features = [np.zeros((self.edge_feature_dim,), dtype=np.float32)]

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.stack(edge_features, axis=0), dtype=torch.float32)
        x = torch.tensor(node_features, dtype=torch.float32)

        graph = {
            "x": x,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
        }

        return graph

    def forward(
        self,
        graph: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Forward propagation.

        Args:
            graph: Graph dictionary containing x, edge_index, edge_attr.

        Returns:
            Output dictionary from MOTMPNet.
        """
        device = next(self.parameters()).device

        graph = {
            key: value.to(device)
            for key, value in graph.items()
        }

        output = self.mpn(graph)

        return output

    def assign_track_ids(
        self,
        detections: List[Dict[str, Any]],
        edge_output: Optional[Dict[str, Any]] = None,
        track_manager: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """
        Assign track IDs to detections.

        Args:
            detections: Detection list.
            edge_output: Output of MOTMPNet.
            track_manager: Optional track manager.

        Returns:
            Detection list with track_id field.
        """
        if len(detections) == 0:
            return detections

        if track_manager is not None:
            if hasattr(track_manager, "assign_track_ids"):
                return track_manager.assign_track_ids(
                    detections=detections,
                    edge_output=edge_output,
                )

            if hasattr(track_manager, "update"):
                return track_manager.update(
                    detections=detections,
                    edge_output=edge_output,
                )

        assigned = []

        for idx, det in enumerate(detections):
            new_det = dict(det)
            new_det["track_id"] = int(det.get("track_id", idx + 1))
            assigned.append(new_det)

        return assigned

    def detections_to_prompts(
        self,
        detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Convert detections with track IDs into SAM2 prompts.

        Args:
            detections: Detection list.

        Returns:
            Prompt list.
        """
        prompts = []

        for idx, det in enumerate(detections):
            if "bbox_xyxy" in det:
                box_xyxy = np.asarray(det["bbox_xyxy"], dtype=np.float32)
            elif "bbox_xywh" in det:
                box_xyxy = self.xywh_to_xyxy(
                    np.asarray(det["bbox_xywh"], dtype=np.float32)
                )
            else:
                continue

            box_xywh = self.xyxy_to_xywh(box_xyxy)

            prompt = {
                "frame_id": int(det.get("frame_id", 0)),
                "track_id": int(det.get("track_id", idx + 1)),
                "bbox_xyxy": box_xyxy,
                "bbox_xywh": box_xywh,
                "confidence": float(det.get("confidence", 1.0)),
                "class_id": int(det.get("class_id", 0)),
                "class_name": det.get("class_name", "vessel"),
                "source": "graph_prompt_optimizer",
            }

            prompts.append(prompt)

        return prompts

    def generate_prompts(
        self,
        detections: List[Dict[str, Any]],
        frame_id: int = 0,
        track_manager: Optional[Any] = None,
        image_shape: Optional[Tuple[int, int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate optimized prompts from detection results.

        Args:
            detections: Vessel detection list.
            frame_id: Current frame index.
            track_manager: Optional track manager.
            image_shape: Image shape in (height, width) format.

        Returns:
            Optimized prompt list for SAM2.
        """
        if len(detections) == 0:
            return []

        graph = self.build_graph(
            detections=detections,
            frame_id=frame_id,
            image_shape=image_shape,
        )

        edge_output = None

        if graph is not None:
            edge_output = self.forward(graph)

        assigned_detections = self.assign_track_ids(
            detections=detections,
            edge_output=edge_output,
            track_manager=track_manager,
        )

        prompts = self.detections_to_prompts(assigned_detections)

        return prompts

    def compute_loss(
        self,
        output: Dict[str, Any],
        target_edges: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute graph edge classification loss.

        Args:
            output: Output dictionary from MOTMPNet.
            target_edges: Ground-truth edge labels with shape [num_edges, 1].

        Returns:
            Loss dictionary.
        """
        edge_logits = output["edge_logits"]

        target_edges = target_edges.to(edge_logits.device).float()

        if target_edges.dim() == 1:
            target_edges = target_edges.unsqueeze(-1)

        loss_fn = nn.BCEWithLogitsLoss()
        edge_loss = loss_fn(edge_logits, target_edges)

        loss_dict = {
            "total_loss": edge_loss,
            "edge_loss": edge_loss,
        }

        return loss_dict