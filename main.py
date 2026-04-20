
from __future__ import annotations
import os
import argparse
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from Datasets.get_dataset import H1N1Dataset
from Models.GenoGnn import REGNN
from Utils.tools import (
    set_seed,
    setup_logging,
    make_loaders,
)
from Utils.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(
        description='H1N1 HI Regression — REGNN Framework',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument('--csv_path',      default='Datasets/ori_data/a_h1n1_hi_folds.csv')
    parser.add_argument('--data_root',     default='Datasets/')
    parser.add_argument('--dataset_name',  default='H1N1')
    parser.add_argument('--train_ratio',   type=float, default=0.6)
    parser.add_argument('--val_ratio',     type=float, default=0.2)
    parser.add_argument('--batch_size',    type=int,   default=64)

    # ── Model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--model_name',      default='REGNN')
    parser.add_argument('--hidden_channel',  type=int,   default=128)
    parser.add_argument('--out_channel',     type=int,   default=1)
    parser.add_argument('--num_gnn_layers',  type=int,   default=3)
    parser.add_argument('--dropout',         type=float, default=0.2)
    parser.add_argument('--graph_pooling',   default='sum',
                        choices=['sum', 'mean', 'max', 'readout', 'set2set'])
    parser.add_argument('--norm',            default=None, choices=[None, 'bn', 'ln'])
    parser.add_argument('--scaling_factor',  type=float, default=100.0)
    parser.add_argument('--no_re',           action='store_true', default=False)

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',          type=int,   default=300)
    parser.add_argument('--learning_rate',   type=float, default=1e-3)
    parser.add_argument('--weight_decay',    type=float, default=0.0)
    parser.add_argument('--patience',        type=int,   default=50)
    parser.add_argument('--lr_patience',     type=int,   default=10)
    parser.add_argument('--lr_factor',       type=float, default=0.5)
    parser.add_argument('--min_lr',          type=float, default=1e-6)
    parser.add_argument('--scheduler_name',  default='ReduceLROnPlateau',
                        choices=['ReduceLROnPlateau', 'null'])

    # ── Misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--seed',             type=int, default=42)
    parser.add_argument('--device',           default='cuda')
    parser.add_argument('--save_root',        default='Results')
    parser.add_argument('--log_dir',          default='Results/Logs')
    parser.add_argument('--experiment_name',  default='h1n1_regnn')
    parser.add_argument('--test_only',        action='store_true', default=False)

    args = parser.parse_args()
    args.device = (
        torch.device('cuda') if torch.cuda.is_available() and args.device == 'cuda'
        else torch.device('cpu')
    )
    return args


def build_model(args: argparse.Namespace, in_channel: int):
    if args.model_name == 'REGNN':
        return REGNN(
            in_channel=in_channel,
            hidden_channel=args.hidden_channel,
            out_channel=args.out_channel,
            num_gnn_layers=args.num_gnn_layers,
            dropout=args.dropout,
            graph_pooling=args.graph_pooling,
            norm=args.norm,
            scaling_factor=args.scaling_factor,
            no_re=args.no_re,
        ).to(device=args.device)
    else:
        raise ValueError(f'Sorry, the model name {args.model_name} is not supported')


def build_optimizer_and_scheduler(model: REGNN, args: argparse.Namespace):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if args.scheduler_name == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    else:
        scheduler = None
    return optimizer, scheduler


if __name__ == '__main__':

    args = parse_args()
    set_seed(args.seed)

    ########### 设置结果保存路径 ###########
    ckpt_root = os.path.join(args.save_root, args.experiment_name)
    os.makedirs(ckpt_root, exist_ok=True)

    ############################### 加载数据集 ###############################
    logger = setup_logging(args, Path(args.log_dir), str(args.experiment_name))

    logger.info('─── Loading Dataset ───')
    dataset = H1N1Dataset(
        root=str(args.data_root),
        name=args.dataset_name,
        csv_path=str(args.csv_path),
    )

    logger.info(
        f'  Total graphs   : {len(dataset)}'
        f'  |  Node features : {dataset.num_node_features}'
    )

    train_loader, val_loader, test_loader = make_loaders(dataset, args, args.device)
    logger.info(
        f'  Split          : '
        f'train={len(train_loader.dataset)}  '
        f'val={len(val_loader.dataset)}  '
        f'test={len(test_loader.dataset)}'
    )

    ############################### 构建模型 ###############################
    logger.info('─── Building Model ───')
    model = build_model(args, in_channel=dataset.num_node_features)

    optimizer, scheduler = build_optimizer_and_scheduler(model, args)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        config=args,
        run_dir=ckpt_root,
        device=args.device,
        logger=logger,
    )

    ############################### 训练/测试模型 ###############################
    if os.path.exists(trainer.best_ckpt) and args.test_only:
        trainer.test(test_loader, checkpoint_path=trainer.best_ckpt)
    else:
        trainer.fit(train_loader, val_loader)
        trainer.test(test_loader, checkpoint_path=trainer.best_ckpt)

    logger.info('─── Final  Evaluation ───')
    logger.info('=' * 65)
    logger.info('  Done.')
    logger.info('=' * 65)
