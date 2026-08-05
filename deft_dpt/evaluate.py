"""
DEFT-DPT — Evaluation
=====================
Loads a pretrained GAT checkpoint and scores a feature CSV.

Usage:
    python evaluate.py --features Features/Feature_P01_04_EpicKitchen.csv \
        --model_path SavedModels/best_model.pth --meta_cols 4

    # fast reproducible live run
    python evaluate.py --features <csv> --model_path <pth> \
        --meta_cols 4 --subset 200 --seed 42

STATUS: architecture inference verified against all four released checkpoints.
Not run end-to-end against a real feature CSV — if the feature dim mismatches,
adjust --meta_cols first.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import load_feature_csv, to_tensors, N_META_COLS
from model import dynamic_percentile_graph, load_checkpoint, DEFAULT_PERCENTILE


def main():
    p = argparse.ArgumentParser(description="DEFT-DPT Evaluation")
    p.add_argument("--features", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test", "all"])
    p.add_argument("--subset", type=int, default=None,
                   help="Sample N rows for a fast live run")
    p.add_argument("--meta_cols", type=int, default=N_META_COLS)
    p.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    p.add_argument("--confusion", action="store_true")
    p.add_argument("--save_json", default=None)
    args = p.parse_args()

    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_score, recall_score)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    model, in_dim, out_dim = load_checkpoint(args.model_path, device)
    _, eval_df, classes, _ = load_feature_csv(args.features, args.split,
                                              args.seed, args.meta_cols)

    if args.subset and args.subset < len(eval_df):
        eval_df = eval_df.sample(n=args.subset, random_state=args.seed)
        print(f"[INFO] SUBSET MODE: {args.subset} rows (seed={args.seed}). "
              "Not comparable to paper numbers.")

    X, y = to_tensors(eval_df, args.meta_cols)
    print(f"[INFO] split='{args.split}' rows={len(eval_df)} "
          f"feat_dim={X.shape[1]} labels_in_csv={len(classes)}")

    if X.shape[1] != in_dim:
        raise ValueError(
            f"Feature dim mismatch: CSV gives {X.shape[1]}, checkpoint wants "
            f"{in_dim}. Wrong feature file, or --meta_cols ({args.meta_cols}) "
            "is off.")
    if len(classes) != out_dim:
        print(f"[WARN] CSV has {len(classes)} labels but checkpoint has "
              f"{out_dim} outputs; label mapping may not match training.")

    t0 = time.time()
    with torch.no_grad():
        ei, n_edges, n_poss = dynamic_percentile_graph(X, args.percentile, device)
        probs = torch.softmax(model(X.to(device), ei.to(device)), dim=1)
    elapsed = time.time() - t0

    y_true = y.numpy()
    y_pred = probs.argmax(dim=1).cpu().numpy()
    top5 = torch.topk(probs, min(5, out_dim), dim=1).indices.cpu().numpy()

    m = {
        "top1": float(accuracy_score(y_true, y_pred)),
        "top5": float(sum(t in r for t, r in zip(y_true, top5)) / len(y_true)),
        "precision": float(precision_score(y_true, y_pred, average="weighted",
                                           zero_division=1)),
        "recall": float(recall_score(y_true, y_pred, average="weighted",
                                     zero_division=1)),
        "f1": float(f1_score(y_true, y_pred, average="weighted",
                             zero_division=1)),
        "n_samples": int(len(y_true)),
        "edges_retained": int(n_edges), "edges_possible": int(n_poss),
        "sparsity_pct": float(100 * n_edges / n_poss) if n_poss else 0.0,
        "eval_seconds": round(elapsed, 3),
        "split": args.split, "subset": args.subset,
        "percentile": args.percentile,
        "checkpoint": str(args.model_path), "features": str(args.features),
    }

    print("\n" + "=" * 60)
    print("  DEFT-DPT Evaluation Results")
    print("=" * 60)
    print(f"  Top-1 Accuracy  : {100 * m['top1']:.2f}%")
    print(f"  Top-5 Accuracy  : {100 * m['top5']:.2f}%")
    print(f"  Precision (wtd) : {100 * m['precision']:.2f}%")
    print(f"  Recall    (wtd) : {100 * m['recall']:.2f}%")
    print(f"  F1-Score  (wtd) : {100 * m['f1']:.2f}%")
    print(f"  DPT graph       : {n_edges}/{n_poss} edges "
          f"({m['sparsity_pct']:.2f}% @ p{args.percentile:g})")
    print(f"  Wall time       : {elapsed:.2f}s on {len(y_true)} samples")
    print("=" * 60)

    if args.confusion:
        print("\nConfusion matrix:")
        print(confusion_matrix(y_true, y_pred))
    if args.save_json:
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.save_json).write_text(json.dumps(m, indent=2))
        print(f"\n[INFO] Metrics -> {args.save_json}")


if __name__ == "__main__":
    main()
