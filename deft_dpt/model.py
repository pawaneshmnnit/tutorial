"""
DEFT-DPT — Graph Construction and Model
=======================================
  dynamic_percentile_graph : DPT — percentile-thresholded similarity graph
  GATModel                 : 2-layer GAT matching the released checkpoints
  infer_arch               : recover architecture from a saved state_dict
  load_checkpoint          : build + load in one call

STATUS: verified against all four SavedModels/*.pth. Architecture is inferred
from the checkpoint, so no --input_dim / --num_classes flags are needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_PERCENTILE = 95.0
HIDDEN_DIM = 128
HEADS = 2


def dynamic_percentile_graph(X, percentile=DEFAULT_PERCENTILE, device="cpu"):
    """
    Dynamic Percentile Thresholding.

    Keeps pairs whose cosine similarity is at or above the given percentile of
    the full similarity matrix, so sparsity adapts per clip instead of fixing K.

    Args:
        X          : (N, D) frame features
        percentile : threshold percentile (95 in the paper)
    Returns:
        (edge_index (2,E), edge_count, total_possible_edges)

    NOTE: builds a full N x N similarity matrix, so memory is O(N^2). For long
    videos, window the clip or subsample before calling.
    """
    X = X.to(device)
    N = X.shape[0]
    Xn = F.normalize(X, dim=1)
    sim = Xn @ Xn.t()
    threshold = torch.quantile(sim, percentile / 100.0)
    adj = (sim >= threshold).float()
    adj.fill_diagonal_(0)
    rows, cols = torch.nonzero(adj, as_tuple=True)
    edge_index = torch.stack([rows, cols], dim=0)
    return edge_index, edge_index.shape[1], N * (N - 1)


class GATModel(nn.Module):
    """Two GATConv layers with ELU between. Matches DEFT_DPT_GAT.ipynb."""

    def __init__(self, input_dim, hidden_dim=HIDDEN_DIM, output_dim=None,
                 heads=HEADS):
        super().__init__()
        from torch_geometric.nn import GATConv
        if output_dim is None:
            raise ValueError("output_dim (num classes) is required")
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads)
        self.conv2 = GATConv(hidden_dim * heads, output_dim, heads=1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class MultiTaskLoss(nn.Module):
    """CE plus a temporal-smoothness term on consecutive frame features."""

    def __init__(self, alpha=0.8):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.alpha = alpha

    def forward(self, outputs, targets, features):
        ce = self.ce_loss(outputs, targets)
        temporal = torch.mean(torch.abs(features[1:] - features[:-1]))
        return self.alpha * ce + (1 - self.alpha) * temporal


def infer_arch(sd):
    """
    Recover (input_dim, hidden_dim, output_dim, heads) from a state_dict.
      conv1.lin.weight : (hidden*heads, input)
      conv1.att_src    : (1, heads, hidden)
      conv2.lin.weight : (output, hidden*heads)
    """
    need = ["conv1.lin.weight", "conv1.att_src", "conv2.lin.weight"]
    missing = [k for k in need if k not in sd]
    if missing:
        raise KeyError(f"Checkpoint missing {missing}. Present: {list(sd.keys())}")
    heads = sd["conv1.att_src"].shape[1]
    hidden = sd["conv1.att_src"].shape[2]
    in_dim = sd["conv1.lin.weight"].shape[1]
    out_dim = sd["conv2.lin.weight"].shape[0]
    if sd["conv1.lin.weight"].shape[0] != hidden * heads:
        raise ValueError("Inconsistent conv1 shapes; not a plain GATModel.")
    return in_dim, hidden, out_dim, heads


def load_checkpoint(path, device="cpu", verbose=True):
    """Load a raw state_dict or a {'model_state': ...} wrapper. Returns
    (model, input_dim, output_dim)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    sd = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    sd = {k: v for k, v in sd.items()
          if "total_ops" not in k and "total_params" not in k}
    in_dim, hidden, out_dim, heads = infer_arch(sd)
    if verbose:
        print(f"[INFO] Inferred architecture: in={in_dim} hidden={hidden} "
              f"heads={heads} classes={out_dim}")
    model = GATModel(in_dim, hidden, out_dim, heads).to(device)
    model.load_state_dict(sd)
    model.eval()
    return model, in_dim, out_dim
