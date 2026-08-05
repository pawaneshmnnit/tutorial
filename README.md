# tutorial

# How to apply this scaffold

This is an **overlay**, not a replacement. Nothing here deletes your work.
Copy files in, run the restructure script, then fix the FIXMEs.

---

## 1. deft_dpt

```bash
cd /path/to/deft_dpt
cp -r /path/to/scaffold/deft_dpt/. .        # note the trailing /. — copies dotfiles too
git add .gitignore requirements.txt CITATION.cff deft_dpt.py configs scripts
```

Smoke-test immediately — this needs no features and no GPU:

```bash
python deft_dpt.py deft --frames Input_Data/RGB/P01_04 --n 4 --out Results/deft_demo.png
```

Read the printed `theta` block. If it says IDENTITY for every frame, the
localization network is untrained and only the radial weighting is active.
Settle that before building slides around DEFT.

Then reorganise:

```bash
bash scripts/restructure.sh
git status          # review before committing
```

### Manual fixes the script cannot do

| where | problem |
|---|---|
| all `Feature Extractions/*.ipynb` | `D:/Datasets/Datasets/...` hardcoded Windows paths |
| `DEFT_DPT_GAT.ipynb`, `GraphSAGE_DEFT-DPT.ipynb` | `"..Feature_with_FrameLevel/..."` — missing slash |
| `DEFT_DPT_Verb_Noun_Action.ipynb` | `"..Feature_LocalisationLevel/..."` — missing slash |
| `Feature_L_W_Level_Resnet50.ipynb` | `"..Features/Feature_P01_04_tsne.csv"` — missing slash |
| README | claims a `requirements.txt` is included (now true) |
| `best_model_gtea.pth` | GAT tensors, saved by the GCN notebook — resolve the naming |

Before replacing inline `DEFTModule` definitions with `from deft_dpt import ...`,
**diff all eight copies** in `DEFT_Level/`. If one has a non-zero `fc2` init, that
is a trained variant and must not be overwritten.

---

## 2. EgoHANG

```bash
cd /path/to/EgoHANG
cp /path/to/scaffold/egohang/evaluate.py .
cp /path/to/scaffold/egohang/.gitignore .
cp -r /path/to/scaffold/egohang/scripts .
mkdir -p checkpoints && touch checkpoints/.gitkeep
```

`evaluate.py` changes:
- fixes the `NameError` when the checkpoint is a raw `state_dict`
  (the original only assigned `state_dict`/`feat_dim`/`num_classes` inside
  `if "model_state" in ckpt:`)
- falls back to `DATASET_CFG` num_classes if the heads cannot be inferred
- adds `--subset N` and `--seed` to match deft_dpt

Still to do by hand:
- `EPIC-Kitchens/Features/1.csv` and `EGTEA/Features/1.csv` are 1-byte empty
  placeholders — replace or delete them
- no checkpoints exist yet; the docstring already points at `checkpoints/`
- document the anticipation protocol in the README: horizons
  (2.0 → 0.25 s), `T_OBS = 90`, `VAL_SPLIT = 0.2`, and the exact metric

---

## 3. Shared

Both repos now expect the same env. Verify they agree:

```bash
python -c "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"
```

Upload binaries to a GitHub Release in each repo, then set `ASSET_URL` at the
top of each `scripts/download_assets.sh`.

`git rm --cached` does not shrink the clone — the ~239 MB stays in history.
Use `git-filter-repo` if you want it back, and do it now while forks are few.

---

## 4. Order of operations

| when | what |
|---|---|
| today | copy files, run `deft_dpt.py deft`, settle the theta question |
| today | locate a 6144-d feature CSV to pair with `best_model.pth` |
| this week | `restructure.sh`, fix paths, notebooks import from `deft_dpt.py` |
| this week | EgoHANG features + checkpoints uploaded and fetchable |
| before the talk | fill the README results tables; rehearse offline twice |
