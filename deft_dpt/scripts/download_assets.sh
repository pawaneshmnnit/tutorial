#!/usr/bin/env bash
# Fetch pretrained weights and precomputed features for the tutorial demo.
set -euo pipefail

ASSET_URL="${ASSET_URL:-https://github.com/pawanesh-mnnit/deft_dpt/releases/download/v1.0}"

mkdir -p SavedModels Features

fetch () {  # fetch <dest_dir> <filename>
  if [ -f "$1/$2" ]; then
    echo "  [skip] $1/$2 already present"
  else
    echo "  [get ] $2"
    curl -fL --retry 3 -o "$1/$2" "$ASSET_URL/$2"
  fi
}

echo "== Checkpoints =="
fetch SavedModels best_model.pth
fetch SavedModels best_model_VNA.pth
fetch SavedModels best_model_gtea.pth
fetch SavedModels best_model_meccano.pth

echo "== Features =="
fetch Features Feature_P01_04_EpicKitchen.csv
fetch Features Feature_S2_Cheese_C1_GTEA.csv
fetch Features Feature_0005_RGB_Meccano_Sampled.csv

echo
echo "Done. Verify with:"
echo "  python deft_dpt.py evaluate --features Features/Feature_P01_04_EpicKitchen.csv \\"
echo "      --model_path SavedModels/best_model.pth --meta_cols 4 --subset 200"
