

from __future__ import annotations
import os
import copy
import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from Utils.metrics import compute_all
from Utils.tools import format_metrics, load_checkpoint, save_checkpoint, save_history


class Trainer:
    """
    General-purpose trainer for graph-based regression tasks.

    Args:
        model (torch.nn.Module):
            The model to train.  Must accept a PyG ``Data`` / ``Batch``
            object and return ``Tensor[B, out_channel]``.
        optimizer (torch.optim.Optimizer):
            Gradient-descent optimiser (e.g. Adam).
        scheduler:
            Learning-rate scheduler with a ``step(metric)`` interface
            (e.g. ``ReduceLROnPlateau``).
        config:
            Nested ``dict`` loaded from YAML (see ``Config/default.yaml``).
            Uses keys ``train`` (``epochs``, ``patience``, …) and ``run_dir``.
        device (torch.device):
            Target computation device.
        logger (logging.Logger):
            Pre-configured logger (dual stdout + file sink).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        config: dict[str, Any],
        run_dir: Path,
        device: torch.device,
        logger: logging.Logger,
    ) -> None:
        self.model     = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config    = config
        self.device    = device
        self.logger    = logger

        self.run_dir   = run_dir
        self.best_ckpt = os.path.join(self.run_dir, 'best_model.pt')
        self.history: list[dict] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Core steps
    # ─────────────────────────────────────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> float:
        """
        Execute one full pass over *loader* in training mode.

        Steps per batch:
          1. Move batch to device.
          2. Zero gradients → forward → MSE loss → backward → step.

        Returns:
            Mean MSE loss over the epoch (weighted by batch size).
        """
        self.model.train()
        total_loss   = 0.0
        total_graphs = 0

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            pred = self.model(batch).squeeze(-1)          # [B]
            loss = F.mse_loss(pred, batch.y.squeeze(-1))  # scalar

            loss.backward()
            self.optimizer.step()

            total_loss   += loss.item() * batch.num_graphs
            total_graphs += batch.num_graphs

        return total_loss / total_graphs

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        """
        Evaluate the model on *loader* without computing gradients.

        Collects all predictions and labels, then computes the full
        regression metric suite via ``Utils.metrics.compute_all``.

        Returns:
            dict with keys: MAE, RMSE, R2, Pearson, Spearman.
        """
        self.model.eval()
        preds_all: list[torch.Tensor]  = []
        labels_all: list[torch.Tensor] = []

        for batch in loader:
            batch = batch.to(self.device)
            pred  = self.model(batch).squeeze(-1)
            preds_all.append(pred.cpu())
            labels_all.append(batch.y.squeeze(-1).cpu())

        preds  = torch.cat(preds_all)
        labels = torch.cat(labels_all)
        return compute_all(labels, preds)

    # ─────────────────────────────────────────────────────────────────────────
    # High-level interface
    # ─────────────────────────────────────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> list[dict]:
        """
        Full training loop with validation, early stopping, and checkpointing.

        For each epoch:
          • ``train_epoch`` → update weights
          • ``evaluate`` on val set → structured metrics
          • ``scheduler.step(val_MAE)`` → adjust LR
          • Save best checkpoint when val MAE improves
          • Increment patience counter; stop early if threshold reached

        After training, history is persisted to ``run_dir/history.csv``.

        Args:
            train_loader: DataLoader for the training split.
            val_loader:   DataLoader for the validation split.

        Returns:
            List of per-epoch dicts (same rows as history.csv).
        """
        epochs = int(self.config.epochs)
        patience = int(self.config.patience)
        self._log_header()

        best_val_mae = float('inf')
        patience_ctr = 0

        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()

            # ── one epoch ────────────────────────────────────────────────
            train_loss  = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            val_mae     = val_metrics['MAE']

            self.scheduler.step(val_mae)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            elapsed    = time.perf_counter() - t0

            # ── record ───────────────────────────────────────────────────
            row = {
                'epoch':     epoch,
                'train_mse': round(train_loss, 6),
                **{f'val_{k}': round(v, 6) for k, v in val_metrics.items()},
                'lr':        current_lr,
            }
            self.history.append(row)
            self._log_epoch(epoch, train_loss, val_metrics, current_lr, elapsed)

            # ── checkpoint on improvement ─────────────────────────────────
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                patience_ctr = 0
                self._save_best(epoch, val_metrics)
                self.logger.info(
                    f'  [ckpt] Saved best model  '
                    f'(epoch={epoch}  val_MAE={best_val_mae:.4f})'
                )
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    self.logger.info(
                        f'Early stopping at epoch {epoch}  '
                        f'(no val MAE improvement for {patience} epochs).'
                    )
                    break

        # ── persist history ───────────────────────────────────────────────
        history_path = self.run_dir / 'history.csv'
        save_history(self.history, history_path)
        self.logger.info(f'Training history → {history_path}')

        return self.history

    def test(
        self,
        test_loader: DataLoader,
        checkpoint_path: Optional[Path] = None,
    ) -> dict:
        """
        Load a checkpoint (best by default) and evaluate on the test set.

        Args:
            test_loader:     DataLoader for the test split.
            checkpoint_path: Explicit path to a ``.pt`` checkpoint.
                             Falls back to ``run_dir/best_model.pt``.

        Returns:
            Metric dict: MAE, RMSE, R2, Pearson, Spearman.
        """
        ckpt_path = Path(checkpoint_path) if checkpoint_path else self.best_ckpt

        if ckpt_path.exists():
            ckpt = load_checkpoint(ckpt_path, self.model, device=self.device)
            epoch       = ckpt.get('epoch', '?')
            best_val    = ckpt.get('val_metrics', {}).get('MAE', '?')
            self.logger.info(
                f'Loaded checkpoint: {ckpt_path}  '
                f'(epoch={epoch}  val_MAE={best_val})'
            )
        else:
            self.logger.warning(
                f'Checkpoint not found at {ckpt_path}; '
                'evaluating with current model weights.'
            )

        test_metrics = self.evaluate(test_loader)

        self.logger.info('─── Test Results ───')
        self.logger.info('  ' + format_metrics(test_metrics))
        self.logger.info('────────────────────')

        return test_metrics

    # ─────────────────────────────────────────────────────────────────────────
    # Private logging helpers
    # ─────────────────────────────────────────────────────────────────────────

    # Column widths kept consistent between header and row formatters.
    _HEADER = (
        f'{"Epoch":>6}  {"TrainMSE":>9}  {"ValMAE":>7}  '
        f'{"ValRMSE":>8}  {"ValR2":>7}  {"Pearson":>8}  '
        f'{"Spearman":>9}  {"LR":>9}  {"t(s)":>6}'
    )

    def _log_header(self) -> None:
        self.logger.info('─── Training ───')
        self.logger.info(self._HEADER)
        self.logger.info('─' * len(self._HEADER))

    def _log_epoch(
        self,
        epoch: int,
        train_loss: float,
        val_metrics: dict,
        lr: float,
        elapsed: float,
    ) -> None:
        self.logger.info(
            f'{epoch:>6d}  '
            f'{train_loss:>9.4f}  '
            f'{val_metrics["MAE"]:>7.4f}  '
            f'{val_metrics["RMSE"]:>8.4f}  '
            f'{val_metrics["R2"]:>7.4f}  '
            f'{val_metrics["Pearson"]:>8.4f}  '
            f'{val_metrics["Spearman"]:>9.4f}  '
            f'{lr:>9.2e}  '
            f'{elapsed:>6.2f}'
        )

    def _save_best(self, epoch: int, val_metrics: dict) -> None:
        save_checkpoint(
            {
                'epoch':           epoch,
                'model_state':     self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'val_metrics':     val_metrics,
                'config':          copy.deepcopy(self.config),
            },
            self.best_ckpt,
        )
