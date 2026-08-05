"""
DEFT-DPT — Feature CSV Loading
==============================
CSV layout: <meta_cols> metadata columns, then feature columns, plus a column
named ActionLabel. extract_features.py writes 4 metadata columns
(Frame, Verb_class, Noun_class, ActionLabel); the original notebooks assumed 3.
Always pass meta_cols explicitly.

SPLIT CAVEAT
------------
load_feature_csv reproduces the notebooks' random 80/20 split over individual
FRAMES (random_state=42). Frames from the same action instance land on both
sides, and adjacent frames are near-duplicates, so `test` measures
interpolation within seen clips, not generalisation to unseen actions.
For a generalisation number, hold out whole videos and use split="all" on the
held-out file. State in the README which one you report.
"""

import pandas as pd
import torch

LABEL_COL = "ActionLabel"
N_META_COLS = 4


def load_feature_csv(csv_path, split="all", seed=42, meta_cols=N_META_COLS):
    """Returns (full_df, eval_df, classes, mapping)."""
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    if LABEL_COL not in df.columns:
        raise KeyError(f"'{LABEL_COL}' not in {csv_path}. "
                       f"Columns: {list(df.columns)[:8]}...")

    classes = sorted(df[LABEL_COL].unique())
    mapping = {c: i for i, c in enumerate(classes)}
    df["action_class_mapped"] = df[LABEL_COL].map(mapping)

    if split == "all":
        eval_df = df
    else:
        tr, te = train_test_split(df, test_size=0.2, random_state=seed)
        tr, va = train_test_split(tr, test_size=0.2, random_state=seed)
        eval_df = {"train": tr, "val": va, "test": te}[split]
    return df, eval_df, classes, mapping


def to_tensors(df, meta_cols=N_META_COLS):
    """Feature columns are [meta_cols : -1]; the last column is the mapped label."""
    X = torch.tensor(df.iloc[:, meta_cols:-1].values, dtype=torch.float32)
    y = torch.tensor(df["action_class_mapped"].values, dtype=torch.long)
    return X, y
