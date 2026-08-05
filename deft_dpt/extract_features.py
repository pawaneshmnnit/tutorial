"""
DEFT-DPT — Step 1: Feature Extraction
=====================================
Runs DEFT + a CNN backbone over RGB (and optionally flow u/v) frames and
writes a feature CSV with 4 metadata columns:
    Frame, Verb_class, Noun_class, ActionLabel, feat_0 ... feat_N

    python extract_features.py --rgb_root $DATA_ROOT/EPIC_Kitchen/RGB/P01_04 \
        --flow_u_root .../u --flow_v_root .../v \
        --labels .../P01_04.csv \
        --out Features/Feature_P01_04_EpicKitchen.csv

Dimensions: ResNet-50 gives 2048 per stream. RGB+u+v = 6144, matching
SavedModels/best_model.pth. RGB only = 2048, matching best_model_meccano.pth.

STATUS: DEFT runs with identity_init by default, reproducing the behaviour of
the existing feature-extraction notebooks. Pass --deft_ckpt once a trained
DEFT checkpoint exists; features (and downstream numbers) will change.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from deft import DEFTModule, DEFAULT_SIGMA

LABEL_COL = "ActionLabel"
BACKBONES = {"resnet50": 2048, "efficientnet_b0": 1280,
             "vgg16": 4096, "alexnet": 4096}


def build_backbone(name, device):
    import torchvision.models as tvm
    if name == "resnet50":
        m = tvm.resnet50(weights="IMAGENET1K_V1")
        m = nn.Sequential(*list(m.children())[:-1])
    elif name == "efficientnet_b0":
        m = tvm.efficientnet_b0(weights="IMAGENET1K_V1")
        m = nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1))
    elif name == "vgg16":
        m = tvm.vgg16(weights="IMAGENET1K_V1")
        m.classifier = nn.Sequential(*list(m.classifier.children())[:-3])
    elif name == "alexnet":
        m = tvm.alexnet(weights="IMAGENET1K_V1")
        m.classifier = nn.Sequential(*list(m.classifier.children())[:-2])
    else:
        raise ValueError(f"Unknown backbone {name}")
    return m.eval().to(device)


def main():
    p = argparse.ArgumentParser(description="DEFT + backbone feature extraction")
    p.add_argument("--rgb_root", required=True)
    p.add_argument("--flow_u_root", default=None)
    p.add_argument("--flow_v_root", default=None)
    p.add_argument("--labels", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backbone", default="resnet50", choices=list(BACKBONES))
    p.add_argument("--stride", type=int, default=5, help="Sample every Nth frame")
    p.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    p.add_argument("--deft_ckpt", default=None)
    p.add_argument("--no_deft", action="store_true",
                   help="Ablation: backbone only, DEFT bypassed")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    import torchvision.transforms as T
    from PIL import Image
    from tqdm import tqdm

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    backbone = build_backbone(args.backbone, device)

    deft = None
    if not args.no_deft:
        deft = DEFTModule(3, sigma=args.sigma, input_hw=(224, 224)).eval().to(device)
        if args.deft_ckpt:
            sd = torch.load(args.deft_ckpt, map_location=device,
                            weights_only=False)
            deft.load_state_dict(sd.get("model_state", sd))
            print(f"[INFO] Loaded DEFT weights from {args.deft_ckpt}")
        else:
            print("[INFO] DEFT at identity_init: the localization branch is "
                  "inactive; only the radial weighting applies.")

    tf = T.Compose([
        T.Resize((224, 224)), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def feat(path):
        try:
            x = tf(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                if deft is not None:
                    x = deft(x)
                return backbone(x).squeeze().cpu().numpy().ravel()
        except Exception as e:
            print(f"[SKIP] {path}: {e}")
            return None

    labels_df = pd.read_csv(args.labels)
    for col in ["StartFrame", "EndFrame", "Verb_class", "Noun_class", LABEL_COL]:
        if col not in labels_df.columns:
            sys.exit(f"[ERROR] Label CSV missing column '{col}'. "
                     f"Found: {list(labels_df.columns)}")

    frames = sorted(Path(args.rgb_root).glob("*.jpg"))[::args.stride]
    if not frames:
        sys.exit(f"[ERROR] No .jpg frames in {args.rgb_root}")
    print(f"[INFO] {len(frames)} frames (stride {args.stride}), "
          f"backbone={args.backbone}")

    rows, skipped = [], 0
    for fp in tqdm(frames, desc="Extracting"):
        parts = [feat(fp)]
        if args.flow_u_root:
            parts.append(feat(Path(args.flow_u_root) / fp.name))
        if args.flow_v_root:
            parts.append(feat(Path(args.flow_v_root) / fp.name))
        if any(x is None for x in parts):
            skipped += 1
            continue

        try:
            n = int(fp.stem.split("_")[-1])
        except ValueError:
            skipped += 1
            continue
        lab = labels_df[(labels_df["StartFrame"] <= n) &
                        (labels_df["EndFrame"] >= n)]
        if lab.empty:
            skipped += 1
            continue
        r = lab.iloc[0]
        rows.append([fp.name, int(r["Verb_class"]), int(r["Noun_class"]),
                     int(r[LABEL_COL])] + list(np.concatenate(parts)))

    if not rows:
        sys.exit("[ERROR] No frames matched a label interval. Check --labels "
                 "and the frame-numbering convention.")

    d = len(rows[0]) - 4
    cols = ["Frame", "Verb_class", "Noun_class", LABEL_COL] + \
           [f"feat_{i}" for i in range(d)]
    out_df = pd.DataFrame(rows, columns=cols)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"[INFO] Wrote {len(out_df)} rows x {d} features -> {args.out}")
    print(f"[INFO] Skipped {skipped} frames (unreadable or unlabelled)")
    print("[INFO] This file has 4 metadata columns: pass --meta_cols 4 "
          "to train.py / evaluate.py")


if __name__ == "__main__":
    main()
