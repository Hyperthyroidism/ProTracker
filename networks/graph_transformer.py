from typing import Optional

import torch
from torch import nn

try:
    from torch_geometric.nn import TransformerConv
    from torch_geometric.nn.dense.linear import Linear
except ImportError as e:
    raise ImportError(
        "Failed to import torch_geometric. "
        "Please install PyTorch Geometric and its dependencies."
    ) from e


class TransformerEncoderLayer(nn.Module):
    """
    Graph Transformer Encoder Layer for ProTracker.

    This layer is used to enhance node representations in the graph-based
    cascaded prompt optimizer. Different from the standard Transformer encoder
    used for sequence data, this module applies TransformerConv on graph
    structures, where nodes represent vessel detections or tracklets and edges
    represent possible temporal associations.

    The layer contains two main parts:

        1. Graph self-attention block based on TransformerConv
        2. Feed-forward network block

    It supports both pre-normalization and post-normalization forms.
    """

    def __init__(
        self,
        d_model: int,
        heads: int = 1,
        dropout: float = 0.0,
        norm_first: bool = False,
    ) -> None:
        """
        Args:
            d_model: Dimension of node features.
            heads: Number of attention heads.
            dropout: Dropout probability.
            norm_first: Whether to use pre-normalization.
        """
        super().__init__()

        self.d_model = d_model
        self.heads = heads
        self.dropout_p = dropout
        self.norm_first = norm_first

        self.head_channels = self.d_model // self.heads

        assert self.head_channels * self.heads == self.d_model, (
            "d_model must be divisible by heads. "
            f"Got d_model={self.d_model}, heads={self.heads}."
        )

        self.self_attn = TransformerConv(
            in_channels=self.d_model,
            out_channels=self.head_channels,
            heads=self.heads,
            dropout=self.dropout_p,
        )

        self.lin1 = Linear(self.d_model, self.d_model)
        self.lin2 = Linear(self.d_model, self.d_model)

        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout_p)

        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)

        self.dropout1 = nn.Dropout(self.dropout_p)
        self.dropout2 = nn.Dropout(self.dropout_p)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            x: Node feature tensor with shape [num_nodes, d_model].
            edge_index: Graph edge index with shape [2, num_edges].
            edge_attr: Optional edge feature tensor.

        Returns:
            Updated node feature tensor with shape [num_nodes, d_model].
        """
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), edge_index, edge_attr)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, edge_index, edge_attr))
            x = self.norm2(x + self._ff_block(x))

        return x

    def _sa_block(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Graph self-attention block.

        Args:
            x: Node feature tensor.
            edge_index: Graph edge index.
            edge_attr: Optional edge feature tensor.

        Returns:
            Attention-enhanced node feature tensor.
        """
        if edge_attr is None:
            x = self.self_attn(x, edge_index)
        else:
            try:
                x = self.self_attn(x, edge_index, edge_attr=edge_attr)
            except TypeError:
                x = self.self_attn(x, edge_index)

        return self.dropout1(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed-forward block.

        Args:
            x: Node feature tensor.

        Returns:
            Updated node feature tensor.
        """
        x = self.lin1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.lin2(x)
        x = self.dropout2(x)

        return x

    def extra_repr(self) -> str:
        """
        Extra representation for printing the module.

        Returns:
            Description string.
        """
        return (
            f"d_model={self.d_model}, "
            f"heads={self.heads}, "
            f"dropout={self.dropout_p}, "
            f"norm_first={self.norm_first}"
        )