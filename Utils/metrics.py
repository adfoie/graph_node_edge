"""
Regression evaluation metrics for H1N1 HI prediction.

All functions accept flat numpy arrays or torch Tensors.
"""

import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().ravel()
    return np.asarray(x).ravel()


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred) -> float:
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def pearson_r(y_true, y_pred) -> float:
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    if y_true.std() < 1e-8 or y_pred.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def spearman_r(y_true, y_pred) -> float:
    from scipy.stats import spearmanr
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    corr, _ = spearmanr(y_true, y_pred)
    return float(corr) if not np.isnan(corr) else 0.0


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = _to_numpy(y_true), _to_numpy(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / (ss_tot + 1e-8))


def compute_all(y_true, y_pred) -> dict:
    """Return a dict with all metrics keyed by name."""
    return {
        'MAE':      mae(y_true, y_pred),
        'RMSE':     rmse(y_true, y_pred),
        'R2':       r2_score(y_true, y_pred),
        'Pearson':  pearson_r(y_true, y_pred),
        'Spearman': spearman_r(y_true, y_pred),
    }
