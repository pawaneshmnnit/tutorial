#!/usr/bin/env bash
# Reorganise deft_dpt to mirror EgoHAnG. Uses `git mv` so history follows.
# Run from the repo root. Review `git status` before committing.
set -euo pipefail

echo "== Renaming notebook folders (removing spaces from paths) =="
mkdir -p notebooks/01_feature_extraction notebooks/02_training notebooks/03_analysis

[ -d "Feature Extractions" ] && git mv "Feature Extractions" notebooks/01_feature_extraction/tmp && \
  git mv notebooks/01_feature_extraction/tmp/* notebooks/01_feature_extraction/ && \
  rmdir notebooks/01_feature_extraction/tmp

if [ -d "Model Training" ]; then
  git mv "Model Training/GAT"  notebooks/02_training/GAT  2>/dev/null || true
  git mv "Model Training/GCN"  notebooks/02_training/GCN  2>/dev/null || true
  for f in "Model Training"/*.ipynb; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    # strip leading spaces, lowercase, replace spaces/& with underscores
    clean=$(echo "$base" | sed 's/^ *//; s/&/and/g; s/ /_/g' | tr 'A-Z' 'a-z')
    case "$clean" in
      *analysis*|*visualization*|*curve*|*scalability*)
        git mv "$f" "notebooks/03_analysis/$clean" ;;
      *)
        git mv "$f" "notebooks/02_training/$clean" ;;
    esac
  done
  rmdir "Model Training" 2>/dev/null || true
fi

echo "== Untracking large binaries (files stay on disk) =="
git rm --cached SavedModels/*.pth 2>/dev/null || true

cat <<'EOF'

Done. Remaining manual steps:

  1. Fix hardcoded paths in the notebooks:
       D:/Datasets/Datasets/...        -> ${DATA_ROOT}/... or configs/*.yaml
       "..Feature_with_FrameLevel/..." -> "../Features/..."   (missing slash)
       "..Feature_LocalisationLevel/..."-> "../Features/..."  (missing slash)
       "..Features/Feature_P01_04_tsne.csv" -> "../Features/..."

  2. Replace the inline DEFTModule / GATModel / dynamic_percentile_graph
     definitions in the notebooks with:
         from deft_dpt import DEFTModule, dynamic_percentile_graph, build_gat
     DIFF THEM FIRST. If any DEFT_Level notebook has a non-zero fc2 init,
     that is the trained variant and must not be overwritten.

  3. Upload SavedModels/*.pth to a GitHub Release, then fill in ASSET_URL
     in scripts/download_assets.sh.

  4. Note: `git rm --cached` does NOT shrink the clone; the blobs remain in
     history. To actually reclaim the ~239 MB you need git-filter-repo, which
     invalidates existing clones. Cheapest to do now, while forks are few.

EOF
