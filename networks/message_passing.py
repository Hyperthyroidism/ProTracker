from typing import Optional, Sequence, Tuple

import torch
from torch import nn

from networks.edge_attention import EdgeAttentionModule


class MLP(nn.Module):
    """
    Multi-layer perceptron used in ProTracker graph networks.

    This module is used for node encoding, edge encoding, node update,
    edge update, and edge classification.
    """

    def __init__(
        self,
        input_dim: int,
        fc_dims: Sequence[int],
        dropout_p: Optional[float] = 0.4,
        use_batchnorm: bool = False,
    ) -> None:
        """
        Args:
            input_dim: Input feature dimension.
            fc_dims: Output dimensions of fully connected layers.
            dropout_p: Dropout probability. If None, dropout is disabled.
            use_batchnorm: Whether to use BatchNorm1d after linear layers.
        """
        super().__init__()

        assert isinstance(fc_dims, (list, tuple)), (
            f"fc_dims must be a list or tuple, but got {type(fc_dims)}"
        )

        layers = []

        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))

            if use_batchnorm and dim != 1:
                layers.append(nn.BatchNorm1d(dim))

            if dim != 1:
                layers.append(nn.ReLU(inplace=True))

            if dropout_p is not None and dim != 1:
                layers.append(nn.Dropout(p=dropout_p))

            input_dim = dim

        self.fc_layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation.

        Args:
            x: Input feature tensor.

        Returns:
            Output feature tensor.
        """
        return self.fc_layers(x)


class MetaLayer(nn.Module):
    """
    Core message passing layer for graph-based tracking.

    This layer performs one round of graph update:

        1. Edge update
        2. Node update

    It follows the general message passing idea used in graph neural networks.
    """

    def __init__(
        self,
        edge_model: Optional[nn.Module] = None,
        node_model: Optional[nn.Module] = None,
    ) -> None:
        """
        Args:
            edge_model: Edge update model.
            node_model: Node update model.
        """
        super().__init__()

        self.edge_model = edge_model
        self.node_model = node_model

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Reset parameters of edge and node models when possible.
        """
        for module in [self.edge_model, self.node_model]:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform one message passing step.

        Args:
            x: Node feature matrix with shape [num_nodes, node_dim].
            edge_index: Edge index with shape [2, num_edges].
            edge_attr: Edge feature matrix with shape [num_edges, edge_dim].

        Returns:
            Updated node features and edge features.
        """
        row, col = edge_index

        if self.edge_model is not None:
            edge_attr = self.edge_model(
                source=x[row],
                target=x[col],
                edge_attr=edge_attr,
                edge_index=edge_index,
            )

        if self.node_model is not None:
            x = self.node_model(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

        return x, edge_attr

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"(edge_model={self.edge_model}, node_model={self.node_model})"
        )


class EdgeModel(nn.Module):
    """
    Edge update model with edge attention.

    This module updates edge features according to:

        source node feature
        target node feature
        current edge feature
        dynamic edge attention weight

    In ProTracker, this module corresponds to the dynamic edge refinement
    mechanism in the graph-based cascaded prompt optimizer.
    """

    def __init__(
        self,
        edge_mlp: nn.Module,
        node_feat_dim: int,
        use_edge_attention: bool = True,
        attention_hidden_dim: Optional[int] = None,
        attention_activation: str = "sigmoid",
    ) -> None:
        """
        Args:
            edge_mlp: MLP used to update edge features.
            node_feat_dim: Dimension of node features.
            use_edge_attention: Whether to use edge attention.
            attention_hidden_dim: Hidden dimension of edge attention module.
            attention_activation: Activation type for edge attention.
        """
        super().__init__()

        self.edge_mlp = edge_mlp
        self.use_edge_attention = use_edge_attention

        if use_edge_attention:
            self.edge_attention = EdgeAttentionModule(
                in_channels=node_feat_dim,
                hidden_channels=attention_hidden_dim,
                activation=attention_activation,
            )
        else:
            self.edge_attention = None

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update edge features.

        Args:
            source: Source node features with shape [num_edges, node_dim].
            target: Target node features with shape [num_edges, node_dim].
            edge_attr: Edge features with shape [num_edges, edge_dim].
            edge_index: Optional edge index.

        Returns:
            Updated edge features.
        """
        if self.edge_attention is not None:
            edge_attr = self.edge_attention.apply_attention(
                source=source,
                target=target,
                edge_attr=edge_attr,
                edge_index=edge_index,
            )

        edge_input = torch.cat([source, target, edge_attr], dim=1)
        updated_edge_attr = self.edge_mlp(edge_input)

        return updated_edge_attr


class MLPGraphIndependent(nn.Module):
    """
    Independent MLP encoders for node and edge features.

    This module applies one MLP to node features and another MLP to edge features.
    It is commonly used before and after message passing:

        before message passing: encode raw node and edge features
        after message passing: classify refined edge features
    """

    def __init__(
        self,
        edge_in_dim: Optional[int] = None,
        node_in_dim: Optional[int] = None,
        edge_out_dim: Optional[int] = None,
        node_out_dim: Optional[int] = None,
        node_fc_dims: Optional[Sequence[int]] = None,
        edge_fc_dims: Optional[Sequence[int]] = None,
        dropout_p: Optional[float] = None,
        use_batchnorm: bool = False,
    ) -> None:
        """
        Args:
            edge_in_dim: Input dimension of edge features.
            node_in_dim: Input dimension of node features.
            edge_out_dim: Output dimension of edge features.
            node_out_dim: Output dimension of node features.
            node_fc_dims: Hidden dimensions of node MLP.
            edge_fc_dims: Hidden dimensions of edge MLP.
            dropout_p: Dropout probability.
            use_batchnorm: Whether to use batch normalization.
        """
        super().__init__()

        node_fc_dims = list(node_fc_dims) if node_fc_dims is not None else []
        edge_fc_dims = list(edge_fc_dims) if edge_fc_dims is not None else []

        if node_in_dim is not None and node_out_dim is not None:
            self.node_mlp = MLP(
                input_dim=node_in_dim,
                fc_dims=node_fc_dims + [node_out_dim],
                dropout_p=dropout_p,
                use_batchnorm=use_batchnorm,
            )
        else:
            self.node_mlp = None

        if edge_in_dim is not None and edge_out_dim is not None:
            self.edge_mlp = MLP(
                input_dim=edge_in_dim,
                fc_dims=edge_fc_dims + [edge_out_dim],
                dropout_p=dropout_p,
                use_batchnorm=use_batchnorm,
            )
        else:
            self.edge_mlp = None

    def forward(
        self,
        edge_feats: Optional[torch.Tensor] = None,
        node_feats: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode edge and node features independently.

        Args:
            edge_feats: Edge feature tensor.
            node_feats: Node feature tensor.

        Returns:
            Encoded edge features and node features.
        """
        if self.edge_mlp is not None and edge_feats is not None:
            out_edge_feats = self.edge_mlp(edge_feats)
        else:
            out_edge_feats = edge_feats

        if self.node_mlp is not None and node_feats is not None:
            out_node_feats = self.node_mlp(node_feats)
        else:
            out_node_feats = node_feats

        return out_edge_feats, out_node_feats