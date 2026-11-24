'''
Helper for every model!
Will load data, split by season, preprocess and score
Ensures all models are using the same splits and metrics
'''
import json
import sys
from imghdr import test_rgb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, log_loss

# Paths
REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / 'data' / 'match_dataset'
RESULTS_DIR = REPO / 'model' / 'results'
PREDICTIONS_DIR = REPO / 'predictions'

# Reuse train-only preprocessing pipeline from analysis/
sys.path.insert(0, str(REPO / "analysis"))
from missing_preprocessing import make_numeric_preprocessor

# Labels and season splits
LABEL = "label_90min_wdl"
CLASSES = ["H", "D", "A"] # home win, draw, away win
GOAL_LABELS = ["label_90min_home_goals", "label_90min_away_goals"]

TRAIN_SEASONS = ("2014-2015", "2023-24")
VALID_SEASON = "2024-25"
TEST_SEASON = "2025-26"

def load_data(kind="compact"):
    # return a DataFrame and list of feature column names
    df = pd.read_csv(DATA_DIR / f"matches_{kind}.csv", encoding="utf-8-sig")
    features = json.loads((DATA_DIR / f"{kind}_feature_columns.json").read_text(encoding="utf-8"))
    df = df.dropna(subset=[LABEL]).reset_index(drop=True)

    return df, features

def split(df):
    '''Split chronologically:
    Train on older seasons, validate on season before most recent, test on most recent'''
    train = df[df.season.between(*TRAIN_SEASONS)].copy()
    valid = df[df.season.between(*VALID_SEASON)].copy()
    test = df[df.season.between(*TEST_SEASON)].copy()

    return train, valid, test

def preprocess(train, others, features):
    '''Fill in any missing data
    Returns numpy arrays'''
    pre = make_numeric_preprocessor(features)
    X_train = pre.fit_transform(train[features])
    return X_train, [pre.transform(d[features]) for d in others]

# Metrics for Scoring Model Predictions
def rps(y_true, probs):
    """Ranked probability score (lower is better; 0 = perfect).

    Treats outcomes as ordered H < D < A, so predicting a draw when the home
    team won is penalised less than predicting an away win.
    """
    onehot = pd.get_dummies(pd.Categorical(y_true, categories=CLASSES)).values.astype(float)
    csum_p = np.cumsum(probs, axis=1)[:, :-1]
    csum_o = np.cumsum(onehot, axis=1)[:, :-1]
    return float(np.mean(np.sum((csum_p - csum_o) ** 2, axis=1) / (len(CLASSES) - 1)))

def evaluate(y_true, probs):
    # Score probability predictions
    probs = np.asarray(probs, dtype=float)
    y_pred = np.array(CLASSES)[probs.argmax(axis=1)]

    # Reorder H/D/A columns before calling it bc sklearn log_loss wants alphabetical order
    alpha = sorted(CLASSES)
    probs_alpha = probs[:, [CLASSES.index(c) for c in alpha]]
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=CLASSES, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, probs_alpha, labels=alpha),
        "rps": rps(y_true, probs),
    }

# Code to save results
def save_results(model_name, split_name, data, probs):
    """Append one row to results/scores.csv and save per-match probabilities."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    scores = {"model": model_name, "split": split_name, **evaluate(data[LABEL], probs)}
    path = RESULTS_DIR / "scores.csv"
    table = pd.read_csv(path) if path.exists() else pd.DataFrame()
    if not table.empty:  # rerunning a model replaces its old row
        table = table[~((table.model == model_name) & (table.split == split_name))]
    table = pd.concat([table, pd.DataFrame([scores])], ignore_index=True)
    table.to_csv(path, index=False)

    keep = [c for c in ["season", "stage", "home_team", "away_team", LABEL] if c in data.columns]
    out = data[keep].copy()
    out[[f"p_{c}" for c in CLASSES]] = probs
    out.to_csv(PREDICTIONS_DIR / f"{model_name}_{split_name}.csv", index=False)

    print(f"{model_name:<25} {split_name:<6} " +
          "  ".join(f"{k}={v:.3f}" for k, v in scores.items() if isinstance(v, float)))
    return scores
