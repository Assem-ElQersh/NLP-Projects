from __future__ import annotations

import os
from typing import Tuple, Optional

import pandas as pd


def load_dataset(csv_path: str) -> pd.DataFrame:
    """
    Load the resume-job matching dataset CSV.

    Expected columns: 'job_description', 'resume', 'match_score'.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"job_description", "resume", "match_score"}
    missing = sorted(list(required - set(df.columns)))
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    return df


def train_val_test_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train/val/test by proportions.
    """
    if not (0.0 < train_size < 1.0) or not (0.0 < val_size < 1.0):
        raise ValueError("train_size and val_size must be in (0, 1)")

    if train_size + val_size >= 1.0:
        raise ValueError("train_size + val_size must be < 1.0")

    df_shuffled = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n = len(df_shuffled)
    n_train = int(n * train_size)
    n_val = int(n * val_size)

    train_df = df_shuffled.iloc[:n_train].reset_index(drop=True)
    val_df = df_shuffled.iloc[n_train:n_train + n_val].reset_index(drop=True)
    test_df = df_shuffled.iloc[n_train + n_val:].reset_index(drop=True)
    return train_df, val_df, test_df


def coerce_score_and_clip(df: pd.DataFrame, score_col: str = "match_score") -> pd.DataFrame:
    """
    Ensure numeric scores in [1, 5]. Returns a copy.
    """
    out = df.copy()
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce").fillna(1.0)
    out[score_col] = out[score_col].clip(lower=1.0, upper=5.0)
    return out


