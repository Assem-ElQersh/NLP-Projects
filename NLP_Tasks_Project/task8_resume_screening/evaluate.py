from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)  # Manual RMSE calculation for compatibility
    r2 = r2_score(y_true, y_pred)
    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }


def to_class(y_continuous, bins=(1, 2, 3, 4, 5)):
    # Round to nearest valid class in [1..5]
    y = np.rint(np.clip(y_continuous, 1, 5)).astype(int)
    return y


