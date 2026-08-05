# File status

## Ready to use

| file | condition |
|---|---|
| `deft.py` | Complete. `identity_init=True` by default, reproducing current behaviour. Pass `identity_init=False` for joint training. `input_hw` fixes the lazy-`fc1` checkpoint bug. |
| `model.py` | Complete. Architecture inference verified against all four released checkpoints. `dynamic_percentile_graph` is O(N²) memory — window long videos. |
| `dataset.py` | Complete. Default `meta_cols=4` matches `extract_features.py`; the old notebooks used 3. |
| `evaluate.py` | Complete. Compiles; not run against a real feature CSV (none exist in the repo). |
| `visualize.py` | `deft` runs on the committed JPGs with no features/GPU/PyG. Run this first. |
| `LICENSE` | MIT template — **replace `FIXME_AUTHOR_NAMES`**. |

## Ready, with a scope limit

| file | condition |
|---|---|
| `extract_features.py` | Works, but DEFT runs at `identity_init` unless `--deft_ckpt` is given, and no such checkpoint exists yet. Features are therefore backbone + radial weighting only. |
| `train.py` | Trains the **GAT only**, on cached features. DEFT gets no gradient from this loop. Adds checkpoint saving, which the GAT notebook lacked. |

## Not created — deferred

| file | why |
|---|---|
| joint DEFT+GAT trainer | Needs the backbone inside the training loop, GPU time, and a decision on the paper. Deferred per your call. |
| `Features/*.csv` | Requires the raw datasets. Generate with `extract_features.py`. |
| trained DEFT checkpoint | Blocked on the above. |
| `Results/full_eval.json` | Generate with `evaluate.py --save_json` once features exist. |

## Known issues these files do NOT fix

1. DEFT's localization branch is inactive at `identity_init`; `Localization_DEFT.ipynb`'s reconstruction loss `MSE(warped, original)` has the identity as its optimum, so it is not a fix either.
2. The frame-level random split is not segment-disjoint. Documented in `dataset.py`; the numbers it produces are interpolation-within-clip.
3. `best_model_gtea.pth` holds GAT tensors but is saved by the GCN notebook. Unresolved.
4. Hardcoded `D:/Datasets/...` paths and missing-slash paths remain in the notebooks. See `scripts/restructure.sh`.
