# DEFT-DPT

**Dynamic Egocentric Feature Transformation (DEFT)** + **Dynamic Percentile Thresholding (DPT)**
Egocentric multimodal action recognition (RGB, Optical Flow, Depth) with graph-based learning.

> Companion repo: [EgoHAnG](https://github.com/pawanesh-mnnit/EgoHANG) — action *anticipation* on the same features.

---

## Results

<!-- FIXME: fill from the paper before release -->

| Dataset | Split | Top-1 | Top-5 | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| EPIC-Kitchens (P01_04) | test | – | – | – | – | – |
| GTEA (S2_Cheese_C1) | test | – | – | – | – | – |
| MECCANO (0005) | test | – | – | – | – | – |
| ADL (P_16) | test | – | – | – | – | – |

**Evaluation protocol.** <!-- FIXME --> Numbers above use a random 80/20 split over
*individual frames* (`random_state=42`), so frames from the same action instance may
appear in both partitions. This is an interpolation-within-clip estimate, not a
segment-disjoint generalisation estimate. State clearly which one you report.

---

## Install

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# torch/torchvision for your CUDA:  https://pytorch.org/
# torch-geometric:                  https://pytorch-geometric.readthedocs.io/
```

## Reproduce

```bash
bash scripts/download_assets.sh          # weights + precomputed features

python deft_dpt.py evaluate \
    --features Features/Feature_P01_04_EpicKitchen.csv \
    --model_path SavedModels/best_model.pth \
    --meta_cols 4
```

## Everything else is one script

```bash
python deft_dpt.py deft     --frames Input_Data/RGB/P01_04 --n 4   # DEFT visualisation
python deft_dpt.py graph    --features <csv>                       # DPT sparsity + SVSG
python deft_dpt.py demo     --features <csv> --model_path <pth>    # one sample, end to end
python deft_dpt.py evaluate --features <csv> --model_path <pth>    # metrics
python deft_dpt.py extract  --rgb_root <dir> --labels <csv> --out <csv>
```

Add `--subset N --seed 42` to any evaluation for a fast, reproducible live run.

## Pipeline

1. Input frame
2. **DEFT** — affine warp (localization net) + radial weighting
3. Backbone features (ResNet-50 / EfficientNet-B0 / VGG16 / AlexNet)
4. **DPT** — percentile-thresholded Sparse Video Similarity Graph
5. GAT / GCN over the graph
6. Evaluation

## Layout

```
deft_dpt.py       reference implementation + CLI (DEFT, DPT, model, eval, extract)
configs/          per-dataset paths and hyperparameters
notebooks/        01_feature_extraction, 02_training, 03_analysis
Input_Data/       sample RGB + flow frames (P01_04) for the demo
Features/         generated — see scripts/download_assets.sh
SavedModels/      pretrained checkpoints — see scripts/download_assets.sh
Results/          paper figures
```

## Checkpoints

| file | input_dim | classes | notes |
|---|---|---|---|
| `best_model.pth` | 6144 | 16 | RGB + flow_u + flow_v, 3x2048 |
| `best_model_VNA.pth` | 6147 | 16 | verb/noun/action variant |
| `best_model_gtea.pth` | 2049 | 10 | **FIXME:** GAT-shaped tensors but saved by the GCN notebook |
| `best_model_meccano.pth` | 2048 | 30 | RGB only |

All are raw `state_dict`s of the 2-layer GAT. `deft_dpt.py` infers the architecture
from the checkpoint, so no `--input_dim` flag is needed.

## Datasets

[EPIC-Kitchens](https://github.com/epic-kitchens) ·
[MECCANO](https://iplab.dmi.unict.it/MECCANO/) ·
[GTEA](https://cbs.ic.gatech.edu/fpv/) ·
[ADL](https://web.cs.ucdavis.edu/~hpirsiav/papers/ADLdataset/)

Set `export DATA_ROOT=/path/to/datasets` and the configs resolve against it.

## Citation

See `CITATION.cff`.

## License

<!-- FIXME: add a LICENSE file. MIT or Apache-2.0 are the usual choices. -->
