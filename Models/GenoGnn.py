"""
REGNN: Relation-Enhanced Graph Neural Network for protein sequence modelling.

Architecture
------------
REGCNConv  →  stacked relation-aware GCN layers
REGNN      →  multi-layer REGNN backbone + graph pooling + ``Sequential`` MLP head

Data fields expected per graph (from H1N1Dataset)
--------------------------------------------------
  x             : float [N, 8]        VHSE node features
  edge_index    : long  [2, E]        linear-chain adjacency (bidirectional)
  edge_attr     : long  [E]           edge relation id = src_aa * 20 + dst_aa
  residue_type  : long  [N]           amino-acid category index (0-19)
  y             : float [1]           HI regression target
  batch         : long  [N]           batch assignment (added by DataLoader)
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, ModuleList, Parameter, ReLU, Sequential, init

from torch_geometric.nn import MessagePassing
from torch_geometric.nn import (
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from torch_geometric.utils import softmax as pyg_softmax


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _weighted_degree(
    index: Tensor,
    edge_weight: Tensor,
    num_nodes: int,
    dtype: torch.dtype,
) -> Tensor:
    """Per-node weighted in-degree: ``deg[v] = Σ edge_weight[e]`` for edges ``e → v``."""
    out = torch.zeros(num_nodes, dtype=dtype, device=index.device)
    return out.scatter_add_(0, index, edge_weight.to(dtype))


# ---------------------------------------------------------------------------
# REGCNConv
# ---------------------------------------------------------------------------

class REGCNConv(MessagePassing):
    """Relation-Enhanced GCN Convolution layer.

    Each edge carries a scalar weight derived from a learnable
    ``relation_weight`` vector, indexed by a one-hot encoding of the
    relation type.  Self-loops are added internally and their type is
    the node's amino-acid category offset by ``num_edge_types``.

    Args:
        in_channels (int): Input node-feature dimension.
        out_channels (int): Output node-feature dimension.
        num_node_types (int): Number of distinct node types (20 amino acids).
        num_edge_types (int): Number of distinct edge types (20 × 20 pairs).
        scaling_factor (float): Multiplier applied to ``relation_weight``
            before ``leaky_relu``.  Keeps initial weight magnitudes small.
        use_softmax (bool): If ``True``, normalise edge weights per target node
            with sparse softmax; otherwise use weighted-degree GCN normalisation.
        residual (bool): Add a separate linear projection of the centre node
            as a residual connection.
        use_norm (str | None): ``'bn'`` → BatchNorm1d, ``'ln'`` → LayerNorm,
            ``None`` → no normalisation.
        no_re (bool): If ``True``, ``relation_weight`` is frozen (not learned).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_node_types: int,
        num_edge_types: int,
        scaling_factor: float = 100.0,
        use_softmax: bool = False,
        residual: bool = True,
        use_norm: Optional[str] = None,
        no_re: bool = False,
    ):
        super().__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.scaling_factor = scaling_factor
        self.use_softmax = use_softmax
        self.residual = residual
        self.use_norm = use_norm

        # Neighbour linear transform
        self.weight = Parameter(torch.empty(in_channels, out_channels))
        # Centre-node (residual) linear transform — independent weight matrix
        if self.residual:
            self.weight_root = Parameter(torch.empty(in_channels, out_channels))
        self.bias = Parameter(torch.empty(out_channels))

        # One learnable scalar per relation dimension
        rw_dim = num_edge_types + num_node_types
        self.relation_weight = Parameter(
            torch.empty(rw_dim), requires_grad=not no_re
        )

        if use_norm == 'bn':
            self.norm = torch.nn.BatchNorm1d(out_channels)
        elif use_norm == 'ln':
            self.norm = torch.nn.LayerNorm(out_channels)
        else:
            self.norm = None

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.residual:
            init.xavier_uniform_(self.weight_root)
        init.zeros_(self.bias)
        init.constant_(self.relation_weight, 1.0 / self.scaling_factor)
        if self.norm is not None:
            self.norm.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_type: Tensor,
        target_node_type: Tensor,
        return_weights: bool = False,
    ):
        """
        Args:
            x (Tensor): Node features ``[N, in_channels]``.
            edge_index (LongTensor): Graph connectivity ``[2, E]``.
            edge_type (LongTensor): Edge relation id ``[E]``.
            target_node_type (LongTensor): Amino-acid category per node ``[N]``.
            return_weights (bool): If ``True``, also return ``(ew, relation_weight)``.

        Returns:
            Tensor: Updated node features ``[N, out_channels]``.
            (optional) ew (Tensor): Normalised edge weights ``[E + N]``.
            (optional) relation_weight (Tensor): Activated relation weights.
        """
        num_nodes = target_node_type.size(0)

        # --- self-loops: type = node_type + num_edge_types -------------------
        loop_idx = (
            torch.arange(num_nodes, dtype=torch.long, device=edge_index.device)
            .unsqueeze(0)
            .expand(2, -1)
        )
        edge_index = torch.cat([edge_index, loop_idx], dim=1)
        edge_type = torch.cat(
            [edge_type, target_node_type + self.num_edge_types], dim=0
        )

        # --- relation one-hot → per-edge scalar weight -----------------------
        rw_dim = self.num_edge_types + self.num_node_types
        e_feat = torch.zeros(
            edge_type.size(0), rw_dim, device=edge_type.device
        ).scatter_(1, edge_type.unsqueeze(1), 1.0)

        relation_weight = F.leaky_relu(self.relation_weight * self.scaling_factor)
        edge_weight = e_feat.mv(relation_weight)  # [E + N]

        # --- normalisation ---------------------------------------------------
        col = edge_index[1]
        if self.use_softmax:
            ew = pyg_softmax(edge_weight, col, num_nodes=num_nodes)
        else:
            deg = _weighted_degree(col, edge_weight, num_nodes, x.dtype)
            ew = edge_weight * deg.pow(-1.0)[col]

        # --- linear projections ----------------------------------------------
        x_src = x @ self.weight
        x_target = x @ (self.weight_root if self.residual else self.weight)

        # --- message passing -------------------------------------------------
        out = self.propagate(edge_index, x=(x_src, x_target), ew=ew)

        if self.residual:
            out = out + x_target

        if self.norm is not None:
            out = self.norm(out)

        if return_weights:
            return out, ew, relation_weight
        return out

    def message(self, x_j: Tensor, ew: Tensor) -> Tensor:
        return ew.unsqueeze(-1) * x_j

    def update(self, aggr_out: Tensor) -> Tensor:
        return aggr_out + self.bias


# ---------------------------------------------------------------------------
# REGNN
# ---------------------------------------------------------------------------

class REGNN(torch.nn.Module):

    _VALID_POOLING = ('sum', 'mean', 'max', 'readout', 'set2set')

    def __init__(
        self,
        in_channel: int,
        hidden_channel: int,
        out_channel: int,
        num_gnn_layers: int,
        dropout: float = 0.0,
        graph_pooling: str = 'sum',
        norm: Optional[str] = None,
        scaling_factor: float = 1.0,
        no_re: bool = False,
    ):
        super().__init__()

        if graph_pooling not in self._VALID_POOLING:
            raise ValueError(
                f"graph_pooling must be one of {self._VALID_POOLING}, "
                f"got '{graph_pooling}'."
            )

        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers
        self.graph_pooling = graph_pooling
        self.hidden_channel = hidden_channel

        # GNN backbone
        self.convs = ModuleList([
            REGCNConv(
                in_channels=in_channel if i == 0 else hidden_channel,
                out_channels=hidden_channel,
                num_node_types=20,
                num_edge_types=20 * 20,
                scaling_factor=scaling_factor,
                use_norm=norm,
                no_re=no_re,
            )
            for i in range(num_gnn_layers)
        ])

        # Graph pooling operator
        if graph_pooling == 'sum':
            self.pool = global_add_pool
        elif graph_pooling == 'mean':
            self.pool = global_mean_pool
        elif graph_pooling == 'max':
            self.pool = global_max_pool
        elif graph_pooling == 'set2set':
            self.pool = Set2Set(hidden_channel, processing_steps=2)
        else:  # 'readout' — mean + max concat, handled inline
            self.pool = None

        # MLP head
        mlp_in = hidden_channel * 2 if graph_pooling == 'readout' else hidden_channel
        self.mlp = Sequential(
            Linear(mlp_in, hidden_channel),
            ReLU(),
            Linear(hidden_channel, out_channel),
        )

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for m in self.mlp.modules():
            if hasattr(m, 'reset_parameters'):
                m.reset_parameters()

    def forward(self, data, return_emb: bool = False):

        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        residue_type = data.residue_type
        batch = data.batch

        graph_repr = None
        for conv in self.convs:
            x = F.relu(conv(x, edge_index, edge_attr, residue_type))
            x = F.dropout(x, p=self.dropout, training=self.training)

            if self.graph_pooling == 'readout':
                layer_pool = torch.cat(
                    [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
                )
            else:
                layer_pool = self.pool(x, batch)

            graph_repr = layer_pool if graph_repr is None else graph_repr + layer_pool

        graph_repr = F.dropout(graph_repr, p=self.dropout, training=self.training)


        return self.mlp(graph_repr)
