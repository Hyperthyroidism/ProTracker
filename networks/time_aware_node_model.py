from typing import Callable

import torch
from torch import nn


class TimeAwareNodeModel(nn.Module):
    """
    Time-aware node update model for ProTracker.

    In multi-object tracking, graph edges usually represent temporal association
    hypotheses between vessel detections or tracklets. Therefore, messages from
    past frames and future frames should be treated differently.

    This module separates graph messages into two directions:

        1. flow_in:  messages from future or later nodes to current nodes
        2. flow_out: messages from past or earlier nodes to current nodes

    Then the two directional messages are aggregated and used to update node
    representations.
    """

    def __init__(
        self,
        flow_in_mlp: nn.Module,
        flow_out_mlp: nn.Module,
        node_mlp: nn.Module,
        node_agg_fn: Callable,
    ) -> None:
        """
        Args:
            flow_in_mlp: MLP used to process incoming temporal messages.
            flow_out_mlp: MLP used to process outgoing temporal messages.
            node_mlp: MLP used to update node features after aggregation.
            node_agg_fn: Aggregation function, such as scatter_mean,
                         scatter_max, or scatter_add.
        """
        super().__init__()

        self.flow_in_mlp = flow_in_mlp
        self.flow_out_mlp = flow_out_mlp
        self.node_mlp = node_mlp
        self.node_agg_fn = node_agg_fn

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Update node features with time-aware message passing.

        Args:
            x: Node feature matrix with shape [num_nodes, node_dim].
            edge_index: Edge index with shape [2, num_edges].
            edge_attr: Edge feature matrix with shape [num_edges, edge_dim].

        Returns:
            Updated node feature matrix with shape [num_nodes, node_dim].
        """
        row, col = edge_index

        num_nodes = x.size(0)
        device = x.device

        flow_out = self._compute_flow_out(
            x=x,
            row=row,
            col=col,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
        )

        flow_in = self._compute_flow_in(
            x=x,
            row=row,
            col=col,
            edge_attr=edge_attr,
            num_nodes=num_nodes,
        )

        if flow_out is None:
            flow_out = torch.zeros_like(flow_in)

        if flow_in is None:
            flow_in = torch.zeros_like(flow_out)

        if flow_in is None and flow_out is None:
            hidden_dim = self._infer_hidden_dim(x, device)
            flow_in = torch.zeros(
                (num_nodes, hidden_dim),
                dtype=x.dtype,
                device=device,
            )
            flow_out = torch.zeros(
                (num_nodes, hidden_dim),
                dtype=x.dtype,
                device=device,
            )

        flow = torch.cat([flow_in, flow_out], dim=1)
        updated_x = self.node_mlp(flow)

        return updated_x

    def _compute_flow_out(
        self,
        x: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Compute outgoing temporal messages.

        In the original time-aware graph formulation, row < col is used as a
        simple temporal direction assumption. It means messages are propagated
        from earlier nodes to later nodes.

        Args:
            x: Node feature matrix.
            row: Source node indices.
            col: Target node indices.
            edge_attr: Edge feature matrix.
            num_nodes: Number of graph nodes.

        Returns:
            Aggregated outgoing messages.
        """
        flow_out_mask = row < col

        if torch.sum(flow_out_mask) == 0:
            return None

        flow_out_row = row[flow_out_mask]
        flow_out_col = col[flow_out_mask]

        flow_out_input = torch.cat(
            [
                x[flow_out_col],
                edge_attr[flow_out_mask],
            ],
            dim=1,
        )

        flow_out = self.flow_out_mlp(flow_out_input)
        flow_out = self.node_agg_fn(
            flow_out,
            flow_out_row,
            num_nodes,
        )

        return flow_out

    def _compute_flow_in(
        self,
        x: torch.Tensor,
        row: torch.Tensor,
        col: torch.Tensor,
        edge_attr: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Compute incoming temporal messages.

        In the original time-aware graph formulation, row > col is used as the
        opposite temporal direction. It helps the model aggregate messages from
        the other temporal side.

        Args:
            x: Node feature matrix.
            row: Source node indices.
            col: Target node indices.
            edge_attr: Edge feature matrix.
            num_nodes: Number of graph nodes.

        Returns:
            Aggregated incoming messages.
        """
        flow_in_mask = row > col

        if torch.sum(flow_in_mask) == 0:
            return None

        flow_in_row = row[flow_in_mask]
        flow_in_col = col[flow_in_mask]

        flow_in_input = torch.cat(
            [
                x[flow_in_col],
                edge_attr[flow_in_mask],
            ],
            dim=1,
        )

        flow_in = self.flow_in_mlp(flow_in_input)
        flow_in = self.node_agg_fn(
            flow_in,
            flow_in_row,
            num_nodes,
        )

        return flow_in

    def _infer_hidden_dim(
        self,
        x: torch.Tensor,
        device: torch.device,
    ) -> int:
        """
        Infer hidden dimension when the graph has no valid temporal edges.

        Args:
            x: Node feature matrix.
            device: Current device.

        Returns:
            Hidden dimension.
        """
        dummy_input_dim = x.size(1) * 2

        dummy_input = torch.zeros(
            (1, dummy_input_dim),
            dtype=x.dtype,
            device=device,
        )

        try:
            dummy_output = self.flow_in_mlp(dummy_input)
            hidden_dim = dummy_output.size(1)
        except Exception:
            hidden_dim = x.size(1)

        return hidden_dim