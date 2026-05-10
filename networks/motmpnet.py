from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

try:
    from torch_scatter import scatter_mean, scatter_max, scatter_add
except ImportError as e:
    raise ImportError(
        "Failed to import torch_scatter. "
        "Please install torch-scatter according to your PyTorch and CUDA version."
    ) from e

from networks.graph_transformer import TransformerEncoderLayer
from networks.message_passing import MLP, MLPGraphIndependent, MetaLayer, EdgeModel
from networks.time_aware_node_model import TimeAwareNodeModel


class HiclFeatsEncoder(nn.Module):
    """
    Hierarchical feature encoder.

    This module is reserved for hierarchical graph features. In ProTracker,
    the graph-based cascaded prompt optimizer can progressively merge short
    vessel tracklets into longer tracklets. This encoder provides an interface
    for encoding and merging hierarchical node features.
    """

    def __init__(
        self,
        node_dim: int,
        detach_hicl_grad: bool = False,
        merge_method: str = "cat",
        skip_conn: bool = False,
        ignore_mpn_out: bool = False,
        use_layerwise: bool = False,
    ) -> None:
        """
        Args:
            node_dim: Dimension of node features.
            detach_hicl_grad: Whether to detach gradients of message-passing outputs.
            merge_method: Feature merging method. Supported values: "cat" and "sum".
            skip_conn: Whether to use skip connection when merging features.
            ignore_mpn_out: Whether to ignore message-passing output.
            use_layerwise: Whether to use layer-wise merging.
        """
        super().__init__()

        assert merge_method in ("cat", "sum"), "merge_method must be 'cat' or 'sum'."

        self.node_dim = node_dim
        self.detach_hicl_grad = detach_hicl_grad
        self.merge_method = merge_method
        self.skip_conn = skip_conn
        self.ignore_mpn_out = ignore_mpn_out
        self.use_layerwise = use_layerwise

        self.encoder_hicl_feats = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, node_dim),
            nn.ReLU(inplace=True),
        )

        self.encoder_hicl_feats_post_mpn = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(node_dim, node_dim),
            nn.ReLU(inplace=True),
        )

        self.merge_skip_conn = nn.Sequential(
            nn.Linear(2 * node_dim, 2 * node_dim),
            nn.ReLU(inplace=True),
            nn.Linear(2 * node_dim, node_dim),
            nn.ReLU(inplace=True),
        )

        if merge_method == "cat":
            self.merge_hicl_feats = nn.Sequential(
                nn.Linear(2 * node_dim, 2 * node_dim),
                nn.ReLU(inplace=True),
                nn.Linear(2 * node_dim, node_dim),
                nn.ReLU(inplace=True),
            )
        else:
            self.merge_hicl_feats = None

        if use_layerwise:
            self.layerwise_merge = nn.Linear(2 * node_dim, node_dim)
        else:
            self.layerwise_merge = None

    def pool_node_feats(
        self,
        node_feats: torch.Tensor,
        labels: Union[List[int], torch.Tensor],
    ) -> torch.Tensor:
        """
        Pool node features according to hierarchical labels.

        Args:
            node_feats: Node feature tensor with shape [num_nodes, node_dim].
            labels: Cluster labels or tracklet labels.

        Returns:
            Pooled node features.
        """
        labels = torch.as_tensor(labels, device=node_feats.device).long()
        return scatter_mean(node_feats, labels, dim=0)

    def forward(
        self,
        latent_node_feats: torch.Tensor,
        hicl_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode hierarchical node features.

        Args:
            latent_node_feats: Current node features.
            hicl_feats: Hierarchical features.

        Returns:
            Encoded hierarchical features.
        """
        hicl_feats = self.encoder_hicl_feats(hicl_feats)

        if self.use_layerwise and self.layerwise_merge is not None:
            hicl_feats = torch.cat([hicl_feats, latent_node_feats], dim=1)
            hicl_feats = self.layerwise_merge(hicl_feats)

        return hicl_feats

    def post_mpn_encode_node_feats(
        self,
        latent_node_feats: torch.Tensor,
        initial_hicl_feats: Optional[torch.Tensor],
        initial_node_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode node features after message passing.

        Args:
            latent_node_feats: Output node features from message passing.
            initial_hicl_feats: Initial hierarchical features.
            initial_node_feats: Initial node features.

        Returns:
            Encoded node features.
        """
        if initial_hicl_feats is None:
            initial_hicl_feats = initial_node_feats

        if self.ignore_mpn_out:
            return initial_hicl_feats

        if self.detach_hicl_grad:
            latent_node_feats = latent_node_feats.detach()

        latent_node_feats = self.encoder_hicl_feats_post_mpn(latent_node_feats)

        if self.skip_conn and initial_hicl_feats is not None:
            latent_node_feats = torch.cat(
                [latent_node_feats, initial_hicl_feats],
                dim=1,
            )
            latent_node_feats = self.merge_skip_conn(latent_node_feats)

        return latent_node_feats


class MOTMPNet(nn.Module):
    """
    Message Passing Network for multi-vessel tracking.

    This module is the core graph neural network used in the graph-based
    cascaded prompt optimizer. It encodes node and edge features, performs
    iterative message passing, and predicts edge association scores.

    In ProTracker, graph nodes represent vessel detections or tracklets,
    while graph edges represent possible temporal association hypotheses.
    """

    def __init__(
        self,
        model_params: Dict[str, Any],
        bb_encoder: Optional[nn.Module] = None,
        motion: bool = False,
        pos_feats: bool = True,
        reid: bool = False,
    ) -> None:
        """
        Args:
            model_params: Model configuration dictionary.
            bb_encoder: Optional CNN encoder for image patch features.
            motion: Whether motion features are used.
            pos_feats: Whether position features are used.
            reid: Whether ReID appearance features are used.
        """
        super().__init__()

        self.model_params = self._parse_model_params(model_params)
        self.node_cnn = bb_encoder

        encoder_feats_dict = deepcopy(self.model_params["encoder_feats_dict"])

        if motion:
            encoder_feats_dict["edge_in_dim"] += 1

        if not reid and self.model_params.get("remove_reid_dim", False):
            encoder_feats_dict["edge_in_dim"] -= 1

        if not pos_feats and self.model_params.get("remove_pos_dim", False):
            encoder_feats_dict["edge_in_dim"] -= 4

        self.encoder_feats_dict = encoder_feats_dict
        self.classifier_feats_dict = self.model_params["classifier_feats_dict"]

        self.transformer_encoder = TransformerEncoderLayer(
            d_model=encoder_feats_dict["node_in_dim"],
            heads=self.model_params.get("transformer_heads", 8),
            dropout=self.model_params.get("transformer_dropout", 0.1),
            norm_first=self.model_params.get("transformer_norm_first", False),
        )

        self.encoder = MLPGraphIndependent(**encoder_feats_dict)
        self.classifier = MLPGraphIndependent(**self.classifier_feats_dict)

        self.MPNet = self._build_core_mpnet(
            model_params=self.model_params,
            encoder_feats_dict=encoder_feats_dict,
        )

        self.num_enc_steps = self.model_params.get("num_enc_steps", 6)
        self.num_class_steps = self.model_params.get("num_class_steps", 6)

        if self.model_params.get("do_hicl_feats", False):
            self.hicl_feats_encoder = HiclFeatsEncoder(
                node_dim=encoder_feats_dict["node_out_dim"],
                **self.model_params.get("hicl_feats_encoder", {}),
            )
        else:
            self.hicl_feats_encoder = None

    @staticmethod
    def _parse_model_params(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse model parameters from either a full ProTracker config or a compact
        MOTMPNet config.

        Args:
            config: Configuration dictionary.

        Returns:
            Parsed model parameter dictionary.
        """
        if "encoder_feats_dict" in config:
            return config

        gpo_cfg = config.get("graph_prompt_optimizer", {})
        graph_cfg = gpo_cfg.get("graph", {})
        mp_cfg = gpo_cfg.get("message_passing", {})
        trans_cfg = gpo_cfg.get("transformer_encoder", {})

        node_in_dim = graph_cfg.get("node_feature_dim", 256)
        edge_in_dim = graph_cfg.get("edge_feature_dim", 32)
        hidden_dim = graph_cfg.get("hidden_dim", 128)

        model_params = {
            "encoder_feats_dict": {
                "edge_in_dim": edge_in_dim,
                "node_in_dim": node_in_dim,
                "edge_out_dim": hidden_dim,
                "node_out_dim": hidden_dim,
                "node_fc_dims": [hidden_dim],
                "edge_fc_dims": [hidden_dim],
                "dropout_p": mp_cfg.get("dropout", 0.1),
                "use_batchnorm": mp_cfg.get("use_batchnorm", False),
            },
            "classifier_feats_dict": {
                "edge_in_dim": hidden_dim,
                "node_in_dim": None,
                "edge_out_dim": 1,
                "node_out_dim": None,
                "node_fc_dims": [],
                "edge_fc_dims": [
                    gpo_cfg.get("classifier", {}).get(
                        "edge_classifier_hidden_dim",
                        hidden_dim,
                    )
                ],
                "dropout_p": 0.0,
                "use_batchnorm": False,
            },
            "edge_model_feats_dict": {
                "fc_dims": [hidden_dim, hidden_dim],
                "dropout_p": mp_cfg.get("dropout", 0.1),
                "use_batchnorm": mp_cfg.get("use_batchnorm", False),
            },
            "node_model_feats_dict": {
                "fc_dims": [hidden_dim],
                "dropout_p": mp_cfg.get("dropout", 0.1),
                "use_batchnorm": mp_cfg.get("use_batchnorm", False),
            },
            "node_agg_fn": mp_cfg.get("node_agg_fn", "mean"),
            "reattach_initial_nodes": mp_cfg.get("reattach_initial_nodes", True),
            "reattach_initial_edges": mp_cfg.get("reattach_initial_edges", True),
            "num_enc_steps": mp_cfg.get("num_steps", 6),
            "num_class_steps": mp_cfg.get("num_steps", 6),
            "transformer_heads": trans_cfg.get("heads", 8),
            "transformer_dropout": trans_cfg.get("dropout", 0.1),
            "transformer_norm_first": trans_cfg.get("norm_first", False),
            "use_edge_attention": gpo_cfg.get("edge_attention", {}).get("enabled", True),
            "attention_activation": gpo_cfg.get("edge_attention", {}).get(
                "activation",
                "sigmoid",
            ),
            "do_hicl_feats": False,
            "remove_reid_dim": False,
            "remove_pos_dim": False,
        }

        return model_params

    def _build_core_mpnet(
        self,
        model_params: Dict[str, Any],
        encoder_feats_dict: Dict[str, Any],
    ) -> MetaLayer:
        """
        Build the core message passing network.

        Args:
            model_params: Model parameter dictionary.
            encoder_feats_dict: Encoder feature configuration.

        Returns:
            MetaLayer object.
        """
        node_agg_fn = model_params.get("node_agg_fn", "mean").lower()

        if node_agg_fn == "mean":
            node_agg_fn = lambda out, row, x_size: scatter_mean(
                out,
                row,
                dim=0,
                dim_size=x_size,
            )
        elif node_agg_fn == "max":
            node_agg_fn = lambda out, row, x_size: scatter_max(
                out,
                row,
                dim=0,
                dim_size=x_size,
            )[0]
        elif node_agg_fn == "sum":
            node_agg_fn = lambda out, row, x_size: scatter_add(
                out,
                row,
                dim=0,
                dim_size=x_size,
            )
        else:
            raise ValueError("node_agg_fn must be one of: mean, max, sum.")

        self.reattach_initial_nodes = model_params.get("reattach_initial_nodes", True)
        self.reattach_initial_edges = model_params.get("reattach_initial_edges", True)

        edge_factor = 2 if self.reattach_initial_edges else 1
        node_factor = 2 if self.reattach_initial_nodes else 1

        node_out_dim = encoder_feats_dict["node_out_dim"]
        edge_out_dim = encoder_feats_dict["edge_out_dim"]

        edge_model_in_dim = (
            node_factor * 2 * node_out_dim
            + edge_factor * edge_out_dim
        )

        node_model_in_dim = (
            node_factor * node_out_dim
            + edge_out_dim
        )

        edge_model_feats_dict = model_params["edge_model_feats_dict"]
        node_model_feats_dict = model_params["node_model_feats_dict"]

        edge_mlp = MLP(
            input_dim=edge_model_in_dim,
            fc_dims=edge_model_feats_dict["fc_dims"],
            dropout_p=edge_model_feats_dict.get("dropout_p", 0.1),
            use_batchnorm=edge_model_feats_dict.get("use_batchnorm", False),
        )

        flow_in_mlp = MLP(
            input_dim=node_model_in_dim,
            fc_dims=node_model_feats_dict["fc_dims"],
            dropout_p=node_model_feats_dict.get("dropout_p", 0.1),
            use_batchnorm=node_model_feats_dict.get("use_batchnorm", False),
        )

        flow_out_mlp = MLP(
            input_dim=node_model_in_dim,
            fc_dims=node_model_feats_dict["fc_dims"],
            dropout_p=node_model_feats_dict.get("dropout_p", 0.1),
            use_batchnorm=node_model_feats_dict.get("use_batchnorm", False),
        )

        node_hidden_dim = node_model_feats_dict["fc_dims"][-1]

        node_mlp = nn.Sequential(
            nn.Linear(2 * node_hidden_dim, node_out_dim),
            nn.ReLU(inplace=True),
        )

        edge_model = EdgeModel(
            edge_mlp=edge_mlp,
            node_feat_dim=node_factor * node_out_dim,
            use_edge_attention=model_params.get("use_edge_attention", True),
            attention_activation=model_params.get("attention_activation", "sigmoid"),
        )

        node_model = TimeAwareNodeModel(
            flow_in_mlp=flow_in_mlp,
            flow_out_mlp=flow_out_mlp,
            node_mlp=node_mlp,
            node_agg_fn=node_agg_fn,
        )

        return MetaLayer(
            edge_model=edge_model,
            node_model=node_model,
        )

    @staticmethod
    def _get_data_attr(data: Any, name: str) -> torch.Tensor:
        """
        Read attribute from either a PyG data object or a dictionary.

        Args:
            data: Graph data.
            name: Attribute name.

        Returns:
            Attribute tensor.
        """
        if isinstance(data, dict):
            return data[name]

        return getattr(data, name)

    def _prepare_input_features(
        self,
        data: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prepare node and edge features.

        Args:
            data: Graph data object or dictionary. It should contain:
                  x, edge_index, edge_attr.

        Returns:
            x, edge_index, edge_attr.
        """
        x = self._get_data_attr(data, "x")
        edge_index = self._get_data_attr(data, "edge_index")
        edge_attr = self._get_data_attr(data, "edge_attr")

        x_is_img = len(x.shape) == 4

        if self.node_cnn is not None and x_is_img:
            x = self.node_cnn(x)

            emb_dists = nn.functional.pairwise_distance(
                x[edge_index[0]],
                x[edge_index[1]],
            ).view(-1, 1)

            edge_attr = torch.cat([edge_attr, emb_dists], dim=1)

        return x, edge_index, edge_attr

    def forward(
        self,
        data: Any,
        edge_level_embed: Optional[torch.Tensor] = None,
        node_level_embed: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Forward propagation of MOTMPNet.

        Args:
            data: Graph data object or dictionary containing:
                  x, edge_index, edge_attr.
            edge_level_embed: Optional edge-level embedding.
            node_level_embed: Optional node-level embedding.

        Returns:
            Dictionary containing edge scores and latent features.
        """
        x, edge_index, edge_attr = self._prepare_input_features(data)

        if node_level_embed is not None:
            x = x + node_level_embed

        if edge_level_embed is not None:
            edge_attr = edge_attr + edge_level_embed

        x = self.transformer_encoder(
            x=x,
            edge_index=edge_index,
        )

        latent_edge_feats, latent_node_feats = self.encoder(
            edge_feats=edge_attr,
            node_feats=x,
        )

        initial_edge_feats = latent_edge_feats
        initial_node_feats = latent_node_feats
        initial_hicl_feats = None

        classified_edges = []

        total_steps = self.num_enc_steps

        for step in range(total_steps):
            if self.reattach_initial_nodes:
                mpn_node_feats = torch.cat(
                    [latent_node_feats, initial_node_feats],
                    dim=1,
                )
            else:
                mpn_node_feats = latent_node_feats

            if self.reattach_initial_edges:
                mpn_edge_feats = torch.cat(
                    [latent_edge_feats, initial_edge_feats],
                    dim=1,
                )
            else:
                mpn_edge_feats = latent_edge_feats

            latent_node_feats, latent_edge_feats = self.MPNet(
                x=mpn_node_feats,
                edge_index=edge_index,
                edge_attr=mpn_edge_feats,
            )

            if self.hicl_feats_encoder is not None:
                initial_hicl_feats = self.hicl_feats_encoder.post_mpn_encode_node_feats(
                    latent_node_feats=latent_node_feats,
                    initial_hicl_feats=initial_hicl_feats,
                    initial_node_feats=initial_node_feats,
                )

            should_classify = step >= total_steps - self.num_class_steps

            if should_classify:
                classified_edge_feats, _ = self.classifier(
                    edge_feats=latent_edge_feats,
                    node_feats=None,
                )
                classified_edges.append(classified_edge_feats)

        if len(classified_edges) > 0:
            edge_logits = classified_edges[-1]
            edge_scores = torch.sigmoid(edge_logits)
        else:
            edge_logits = torch.empty(
                (edge_attr.size(0), 1),
                dtype=edge_attr.dtype,
                device=edge_attr.device,
            )
            edge_scores = torch.empty_like(edge_logits)

        output = {
            "edge_logits": edge_logits,
            "edge_scores": edge_scores,
            "classified_edges": classified_edges,
            "latent_node_feats": latent_node_feats,
            "latent_edge_feats": latent_edge_feats,
            "edge_index": edge_index,
        }

        return output