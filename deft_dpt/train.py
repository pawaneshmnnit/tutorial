"""
DEFT-DPT — Training
===================
Trains the GAT classifier on PRE-EXTRACTED features. Reproduces the loop in
Model Training/GAT/DEFT_DPT_GAT.ipynb, plus checkpoint saving (which the
notebook did not do for the GAT variant).

    python train.py --features Features/Feature_P01_04_EpicKitchen.csv \
        --meta_cols 4 --epochs 100 --out SavedModels/best_model.pth

SCOPE — important
-----------------
This trains the GAT ONLY. DEFT is upstream of feature extraction here, so its
parameters receive no gradient from this loop; the features are already fixed
on disk. Joint DEFT+GAT training (frame -> DEFT -> backbone -> DPT -> GAT ->
CE, backpropagating into theta) is NOT implemented yet — it requires running
the backbone inside the training loop rather than reading cached features.
See deft.py for why a standalone DEFT objective is not a substitute.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from dataset import load_feature_csv, to_tensors, N_META_COLS
from model import (GATModel, MultiTaskLoss, dynamic_percentile_graph,
                   DEFAULT_PERCENTILE, HIDDEN_DIM, HEADS)


def main():
    p = argparse.ArgumentParser(description="DEFT-DPT GAT training")
    p.add_argument("--features", required=True)
    p.add_argument("--out", default="SavedModels/best_model.pth")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_dim", type=int, default=HIDDEN_DIM)
    p.add_argument("--heads", type=int, default=HEADS)
    p.add_argument("--alpha", type=float, default=0.8,
                   help="CE vs temporal-smoothness weight")
    p.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    p.add_argument("--meta_cols", type=int, default=N_META_COLS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    from sklearn.metrics import f1_score, precision_score, recall_score

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Device: {device}")

    df, _, classes, _ = load_feature_csv(args.features, "all", args.seed,
                                         args.meta_cols)
    _, tr_df, _, _ = load_feature_csv(args.features, "train", args.seed,
                                      args.meta_cols)
    _, va_df, _, _ = load_feature_csv(args.features, "val", args.seed,
                                      args.meta_cols)
    X_tr, y_tr = to_tensors(tr_df, args.meta_cols)
    X_va, y_va = to_tensors(va_df, args.meta_cols)
    print(f"[INFO] train={len(X_tr)} val={len(X_va)} "
          f"feat_dim={X_tr.shape[1]} classes={len(classes)}")

    model = GATModel(X_tr.shape[1], args.hidden_dim, len(classes),
                     args.heads).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    criterion = MultiTaskLoss(alpha=args.alpha)

    best_top1 = -1.0
    for epoch in range(args.epochs):
        model.train()
        ei, n_e, n_p = dynamic_percentile_graph(X_tr, args.percentile, device)
        optimizer.zero_grad()
        out = model(X_tr.to(device), ei.to(device))
        loss = criterion(out, y_tr.to(device), X_tr.to(device))
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            ei_v, _, _ = dynamic_percentile_graph(X_va, args.percentile, device)
            vout = model(X_va.to(device), ei_v.to(device))
            vloss = criterion(vout, y_va.to(device), X_va.to(device))
            probs = torch.softmax(vout, dim=1)
            preds = probs.argmax(dim=1).cpu().numpy()
            yv = y_va.numpy()
            top1 = (preds == yv).mean()
            k = min(5, probs.shape[1])
            top5idx = torch.topk(probs, k, dim=1).indices.cpu().numpy()
            top5 = np.mean([t in r for t, r in zip(yv, top5idx)])
            f1 = f1_score(yv, preds, average="weighted", zero_division=1)

        if top1 > best_top1:
            best_top1 = top1
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), args.out)

        print(f"Epoch {epoch+1}/{args.epochs} - loss {loss.item():.4f} - "
              f"val_loss {vloss.item():.4f} - top1 {top1:.4f} - "
              f"top5 {top5:.4f} - f1 {f1:.4f} - "
              f"edges {n_e}/{n_p} ({100*n_e/n_p:.2f}%)")

    print(f"\n[INFO] Best val Top-1 {best_top1:.4f}; checkpoint -> {args.out}")
    print("[INFO] Saved as a raw state_dict, matching the released checkpoints.")


if __name__ == "__main__":
    main()
