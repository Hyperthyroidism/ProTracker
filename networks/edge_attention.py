from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F


class EdgeAttentionModule(nn.Module):
    """
    Edge attention module for ProTracker.

    This module calculates an attention weight for each edge according to
    the features of its source node and target node. The attention weight
    is used to dynamically modulate edge features in the graph-based
    cascaded prompt optimizer.

    In multi-vessel tracking, different candidate associations should not
    contribute equally. Edges connecting visually or temporally consistent
    vessel candidates should have higher weights, while noisy or unreliable
    associations should be weakened.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        negative_slope: float = 0.2,
        activation: str = "sigmoid",
    ) -> None:
        """
        Args:
            in_channels: Dimension of node features.
            hidden_channels: Hidden dimension of the attention network.
                             If None, a single linear layer is used.
            negative_slope: Negative slope used in LeakyReLU.
            activation: Output activation function. Supported values:
                        "sigmoid", "softmax", and "none".
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.negative_slope = negative_slope
        self.activation = activation

        if hidden_channels is None:
            self.attn_fc = nn.Linear(2 * in_channels, 1)
        else:
            self.attn_fc = nn.Sequential(
                nn.Linear(2 * in_channels, hidden_channels),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_channels, 1),
            )

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate edge attention weights.

        Args:
            source: Source node features with shape [num_edges, node_dim].
            target: Target node features with shape [num_edges, node_dim].
            edge_index: Optional edge index. It is only used when softmax
                        normalization over graph edges is needed.

        Returns:
            Edge attention weights with shape [num_edges].
        """
        edge_input = torch.cat([source, target], dim=-1)

        edge_attention = self.attn_fc(edge_input)
        edge_attention = torch.squeeze(edge_attention, dim=-1)

        edge_attention = F.leaky_relu(
            edge_attention,
            negative_slope=self.negative_slope,
        )

        if self.activation == "sigmoid":
            edge_attention = torch.sigmoid(edge_attention)

        elif self.activation == "softmax":
            if edge_index is None:
                edge_attention = torch.softmax(edge_attention, dim=0)
            else:
                edge_attention = self.edge_softmax(edge_attention, edge_index)

        elif self.activation == "none":
            pass

        else:
            raise ValueError(f"Unsupported activation type: {self.activation}")

        return edge_attention

    @staticmethod
    def edge_softmax(
        edge_scores: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply softmax normalization for edges with the same source node.

        Args:
            edge_scores: Raw edge scores with shape [num_edges].
            edge_index: Graph edge index with shape [2, num_edges].

        Returns:
            Normalized edge scores with shape [num_edges].
        """
        row = edge_index[0]
        normalized_scores = torch.zeros_like(edge_scores)

        unique_nodes = torch.unique(row)

        for node_id in unique_nodes:
            mask = row == node_id
            normalized_scores[mask] = torch.softmax(edge_scores[mask], dim=0)

        return normalized_scores

    def apply_attention(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply edge attention to edge features.

        Args:
            source: Source node features with shape [num_edges, node_dim].
            target: Target node features with shape [num_edges, node_dim].
            edge_attr: Edge feature tensor with shape [num_edges, edge_dim].
            edge_index: Optional edge index.

        Returns:
            Attention-weighted edge features with shape [num_edges, edge_dim].
        """
        attention = self.forward(
            source=source,
            target=target,
            edge_index=edge_index,
        )

        refined_edge_attr = edge_attr * attention.unsqueeze(-1)

        return refined_edge_attr

    def extra_repr(self) -> str:
        """
        Extra representation for printing the module.

        Returns:
            Description string.
        """
        return (
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"negative_slope={self.negative_slope}, "
            f"activation={self.activation}"
        )