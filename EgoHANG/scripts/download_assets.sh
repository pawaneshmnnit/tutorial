#!/usr/bin/env bash
# Fetch EgoHAnG pretrained weights and fused feature CSVs.
set -euo pipefail
ASSET_URL="${ASSET_URL:-https://github.com/pawanesh-mnnit/EgoHANG/releases/download/v1.0}"

mkdir -p checkpoints EPIC-Kitchens/Features EGTEA/Features

fetch () {
  if [ -f "$1/$2" ]; then echo "  [skip] $1/$2"; else
    echo "  [get ] $2"; curl -fL --retry 3 -o "$1/$2" "$ASSET_URL/$2"; fi
}

echo "== Checkpoints =="
fetch checkpoints P01_04_Fused_model.pth
fetch checkpoints OP01-R01_Fused_model.pth

echo "== Fused features (PCA 2048 -> 512) =="
fetch EPIC-Kitchens/Features P01_04_fused_features_PCA.csv
fetch EGTEA/Features        OP01-R01_fused_features_PCA.csv

echo
echo "Verify with:"
echo "  python evaluate.py --dataset epic_kitchens \\"
echo "    --fused_csv EPIC-Kitchens/Features/P01_04_fused_features_PCA.csv \\"
echo "    --label_csv EPIC-Kitchens/Labels/P01_04.csv \\"
echo "    --model_path checkpoints/P01_04_Fused_model.pth --subset 32"
