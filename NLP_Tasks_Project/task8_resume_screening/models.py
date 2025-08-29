from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error


class TextSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key: str):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class NumericSelector(BaseEstimator, TransformerMixin):
    def __init__(self, key: str):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[[self.key]]


@dataclass
class TfidfRidgeConfig:
    max_features: int = 50000
    ngram_range: Tuple[int, int] = (1, 2)
    alpha: float = 5.0


def build_tfidf_ridge_regressor(config: TfidfRidgeConfig) -> Tuple[Any, Dict[str, Any]]:
    text_features = ColumnTransformer(
        transformers=[
            (
                "jd_tfidf",
                TfidfVectorizer(max_features=config.max_features, ngram_range=config.ngram_range, sublinear_tf=True),
                "job_description",
            ),
            (
                "cv_tfidf",
                TfidfVectorizer(max_features=config.max_features, ngram_range=config.ngram_range, sublinear_tf=True),
                "resume",
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    full = ColumnTransformer(
        transformers=[
            ("text", text_features, ["job_description", "resume"]),
            ("numeric", StandardScaler(with_mean=False), ["skill_overlap"]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = Ridge(alpha=config.alpha, random_state=42)

    pipe = (
        full,
        model,
    )

    # We can't create a sklearn Pipeline with ColumnTransformer + estimator without importing Pipeline.
    # To keep code minimal, return components and let caller fit manually.
    return pipe, {"alpha": config.alpha}


def fit_tfidf_ridge(pipe, train_df: pd.DataFrame, y: np.ndarray):
    transformer, estimator = pipe
    X = train_df[["job_description", "resume", "skill_overlap"]]
    X_trans = transformer.fit_transform(X)
    estimator.fit(X_trans, y)
    return transformer, estimator


def predict(pipe, df: pd.DataFrame) -> np.ndarray:
    transformer, estimator = pipe
    X = df[["job_description", "resume", "skill_overlap"]]
    X_trans = transformer.transform(X)
    preds = estimator.predict(X_trans)
    return preds


