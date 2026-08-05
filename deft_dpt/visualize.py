"""
DEFT-DPT — Visualisation
========================
    python visualize.py deft  --frames Input_Data/RGB/P01_04 --n 4
    python visualize.py graph --features Features/Feature_P01_04.csv --meta_cols 4

`deft` needs only the committed JPGs: no features, no checkpoint, no GPU,
no torch_geometric. Run it first to confirm the install works.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from deft import DEFTModule, DEFAULT_SIGMA, report_theta
from model import dynamic_percentile_graph, DEFAULT_PERCENTILE

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denorm(t):
    return (t.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN) \
        .clamp(0, 1).permute(1, 2, 0).numpy()


def save_fig(fig, out, dpi=150):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"[INFO] Saved -> {out}")


def cmd_deft(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torchvision.transforms as T
    from PIL import Image

    torch.manual_seed(args.seed)
    paths = sorted(Path(args.frames).glob("*.jpg"))[::args.stride][:args.n]
    if not paths:
        raise FileNotFoundError(f"No .jpg frames in {args.frames}")
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])
    batch = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])
    print(f"[INFO] {len(paths)} frames from {args.frames}")

    deft = DEFTModule(3, sigma=args.sigma, input_hw=(224, 224)).eval()
    if args.deft_ckpt:
        sd = torch.load(args.deft_ckpt, map_location="cpu", weights_only=False)
        deft.load_state_dict(sd.get("model_state", sd))
        print(f"[INFO] Loaded DEFT weights from {args.deft_ckpt}")

    with torch.no_grad():
        out, theta, warped, wmap = deft(batch, return_parts=True)
    report_theta([p.name for p in paths], theta, wmap)

    cols = ["Input frame", "After affine warp", "DEFT weight map", "DEFT output"]
    fig, axes = plt.subplots(len(paths), 4, figsize=(12, 3 * len(paths)))
    if len(paths) == 1:
        axes = axes[None, :]
    for r in range(len(paths)):
        axes[r, 0].imshow(denorm(batch[r]))
        axes[r, 1].imshow(denorm(warped[r]))
        im = axes[r, 2].imshow(wmap[r].numpy(), cmap="inferno")
        axes[r, 3].imshow(denorm(out[r]))
        axes[r, 0].set_ylabel(paths[r].name, fontsize=9)
        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(cols[c], fontsize=12, pad=8)
    fig.colorbar(im, ax=axes[:, 2].tolist(), fraction=0.035, pad=0.02,
                 label="weight")
    fig.suptitle("DEFT: affine transform and radial weighting"
                 + ("" if args.deft_ckpt else "  (identity_init)"),
                 fontsize=14, y=0.995)
    save_fig(fig, args.out, args.dpi)


def cmd_graph(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import networkx as nx
    from dataset import load_feature_csv, to_tensors

    _, eval_df, _, _ = load_feature_csv(args.features, "all", args.seed,
                                        args.meta_cols)
    if args.subset and args.subset < len(eval_df):
        eval_df = eval_df.sample(n=args.subset, random_state=args.seed)
    X, y = to_tensors(eval_df, args.meta_cols)
    print(f"[INFO] {len(eval_df)} frames, feat_dim={X.shape[1]}")

    ps = np.arange(50, 100, 2.5)
    keeps = [100 * dynamic_percentile_graph(X, float(p))[1] /
             dynamic_percentile_graph(X, float(p))[2] for p in ps]
    _, n_e, n_p = dynamic_percentile_graph(X, args.percentile)
    print(f"[INFO] p{args.percentile:g}: {n_e}/{n_p} edges "
          f"({100 * n_e / n_p:.2f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(ps, keeps, marker="o", lw=2)
    axes[0].axvline(args.percentile, ls="--", c="crimson",
                    label=f"operating point p{args.percentile:g}")
    axes[0].set_xlabel("DPT percentile"); axes[0].set_ylabel("Edges retained (%)")
    axes[0].set_title("Graph sparsity adapts to the clip")
    axes[0].grid(alpha=0.3); axes[0].legend()

    n_show = min(args.max_nodes, X.shape[0])
    ei, _, _ = dynamic_percentile_graph(X[:n_show], args.percentile)
    G = nx.DiGraph(); G.add_nodes_from(range(n_show))
    G.add_edges_from(ei.t().tolist())
    pos = nx.circular_layout(G)
    nx.draw_networkx_edges(G, pos, ax=axes[1], edge_color="gray",
                           alpha=0.35, arrows=False)
    nx.draw_networkx_nodes(G, pos, ax=axes[1], node_size=90,
                           node_color=y[:n_show].numpy(), cmap="tab20",
                           edgecolors="black", linewidths=0.5)
    axes[1].set_title(f"Sparse Video Similarity Graph "
                      f"(first {n_show} frames, coloured by action)")
    axes[1].axis("off")
    fig.suptitle("Dynamic Percentile Thresholding", fontsize=14)
    save_fig(fig, args.out, args.dpi)


def main():
    ap = argparse.ArgumentParser(description="DEFT-DPT visualisation")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("deft")
    p.add_argument("--frames", required=True)
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--deft_ckpt", default=None)
    p.add_argument("--out", default="Results/deft_demo.png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_deft)

    p = sub.add_parser("graph")
    p.add_argument("--features", required=True)
    p.add_argument("--meta_cols", type=int, default=4)
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--max_nodes", type=int, default=60)
    p.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    p.add_argument("--out", default="Results/dpt_graph.png")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_graph)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
