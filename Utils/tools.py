"""
Utils/tools.py
==============
Shared helper utilities used across the framework.

Responsibilities
----------------
  set_seed          – global random-seed control for reproducibility
  resolve_device    – auto-select CPU / CUDA / MPS
  setup_logging     – dual-sink logger (stdout + file)
  count_parameters  – trainable parameter count
  save_checkpoint   – atomic checkpoint write
  load_checkpoint   – checkpoint restore (model + optional optimizer)
  save_history      – persist per-epoch metrics to CSV
  format_metrics    – pretty-print a metrics dict
  split_indices     – deterministic train / val / test index split
"""

from __future__ import annotations

import csv
import logging
import random
import sys
from pathlib import Path
from typing import Optional
import json
import numpy as np
import torch
from torch_geometric.loader import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Fix all relevant RNG states for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(args,log_dir: Path, experiment_name: str = 'experiment') -> logging.Logger:
    """
    Create (or retrieve) a named logger that writes to both stdout and a
    rotating log file at ``log_dir/<experiment_name>.log``.

    Safe to call multiple times with the same ``experiment_name`` — handlers
    are not duplicated.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f'{experiment_name}.log'

    fmt     = '%(asctime)s | %(levelname)-8s | %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        for handler in (
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode='a', encoding='utf-8'),
        ):
            handler.setFormatter(formatter)
            logger.addHandler(handler)


    logger.info('=' * 65)
    logger.info('  H1N1 HI Regression — REGNN Framework')
    logger.info('=' * 65)
    logger.info(f'  Device     : {args.device}')
    logger.info(f'  Seed       : {args.seed}')
    logger.info(f'  Run dir    : {args.log_dir}')
    logger.info(f'  Experiment : {args.experiment_name}')
    logger.info('  Parameters:\n' + json.dumps(vars(args), indent=2, default=str))
    return logger




# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint I/O
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: Path) -> None:
    """
    Serialise *state* to *path* with ``torch.save``.

    Parent directories are created automatically.  Typical *state* dict::

        {
            'epoch':           <int>,
            'model_state':     model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'val_metrics':     {'MAE': ..., 'RMSE': ..., ...},
            'config':          cfg.to_dict(),
        }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """
    Load a checkpoint and restore *model* (and optionally *optimizer*) states.

    Returns the full checkpoint dict so callers can inspect metadata
    (epoch, val_metrics, config, …).
    """
    ckpt = torch.load(Path(path), map_location=device or torch.device('cpu'))
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# History persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_history(history: list[dict], path: Path) -> None:
    """
    Write the per-epoch training history to a CSV file.

    *history* is a list of dicts with identical keys, e.g.::

        [{'epoch': 1, 'train_mse': 0.42, 'val_MAE': 0.31, ...}, ...]
    """
    if not history:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


# ─────────────────────────────────────────────────────────────────────────────
# Metric formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics(metrics: dict, prefix: str = '') -> str:
    """
    Render a metrics dict as a compact human-readable string.

    Example::
        format_metrics({'MAE': 0.12, 'R2': 0.95}, prefix='val_')
        # → 'val_MAE=0.1200  val_R2=0.9500'
    """
    return '  '.join(f'{prefix}{k}={v:.4f}' for k, v in metrics.items())


# ─────────────────────────────────────────────────────────────────────────────
# Dataset splitting
# ─────────────────────────────────────────────────────────────────────────────

def split_indices(
    n: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Produce reproducible, non-overlapping train / val / test index lists.

    Args:
        n:           Total number of samples.
        train_ratio: Fraction of data for training.
        val_ratio:   Fraction of data for validation.
        seed:        RNG seed for determinism.

    Returns:
        (train_idx, val_idx, test_idx) — plain Python lists of int.
    """
    if train_ratio + val_ratio >= 1.0:
        raise ValueError(
            f'train_ratio + val_ratio must be < 1.0, '
            f'got {train_ratio} + {val_ratio} = {train_ratio + val_ratio:.3f}'
        )

    rng     = np.random.default_rng(seed)
    idx     = rng.permutation(n)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    return (
        idx[:n_train].tolist(),
        idx[n_train: n_train + n_val].tolist(),
        idx[n_train + n_val:].tolist(),
    )



def make_loaders(dataset, args, device):
    train_idx, val_idx, test_idx = split_indices(
        len(dataset), args.train_ratio, args.val_ratio, args.seed)

    train_loader = DataLoader(dataset[torch.tensor(train_idx)], shuffle=True,  batch_size=args.batch_size)
    val_loader   = DataLoader(dataset[torch.tensor(val_idx)],   shuffle=False, batch_size=args.batch_size)
    test_loader  = DataLoader(dataset[torch.tensor(test_idx)],  shuffle=False, batch_size=args.batch_size)

    return train_loader, val_loader, test_loader

