"""
统计并可视化 HI 标签（ground truth）的范围与分布。

数据来源：``H1N1Dataset``（与训练管线一致）。
图表保存目录：``Results/figures/hi_distribution.png``

用法::
    python data_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from Datasets.get_dataset import H1N1Dataset


def labels_from_dataset(dataset: H1N1Dataset) -> np.ndarray:
    """从 InMemoryDataset 的聚合存储中取全部 ``y``（每条样本一个标量）。"""
    y = dataset.data.y
    arr = y.view(-1).float().cpu().numpy()
    assert arr.shape[0] == len(dataset), '样本数与标签数不一致'
    return arr


def plot_distribution(arr: np.ndarray, save_path: Path) -> None:
    """绘制直方图 + 箱线图 + ECDF，保存为一张图。"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    mean_v, med_v = arr.mean(), np.median(arr)
    qs = np.quantile(arr, [0.025, 0.25, 0.75, 0.975])

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.1, 0.45, 1.0], hspace=0.35, wspace=0.25)

    # Histogram
    ax0 = fig.add_subplot(gs[0, :])
    ax0.hist(arr, bins='auto', color='steelblue', edgecolor='white', linewidth=0.6, alpha=0.85)
    ax0.axvline(mean_v, color='darkorange', linestyle='--', linewidth=1.5, label=f'mean={mean_v:.4g}')
    ax0.axvline(med_v, color='forestgreen', linestyle='--', linewidth=1.5, label=f'median={med_v:.4g}')
    ax0.set_xlabel('HI (ground truth)')
    ax0.set_ylabel('count')
    ax0.set_title('HI distribution (histogram)')
    ax0.legend(loc='upper right', fontsize=9)

    # Box plot (full width under hist first row — use gs[1,:])
    ax1 = fig.add_subplot(gs[1, :])
    ax1.boxplot(
        arr,
        vert=False,
        widths=0.35,
        patch_artist=True,
        boxprops=dict(facecolor='lightsteelblue', edgecolor='steelblue'),
        medianprops=dict(color='darkred', linewidth=2),
        whiskerprops=dict(color='steelblue'),
        capprops=dict(color='steelblue'),
    )
    ax1.set_xlabel('HI (ground truth)')
    ax1.set_yticks([])
    ax1.set_title('Box plot')

    # ECDF
    ax2 = fig.add_subplot(gs[2, :])
    x = np.sort(arr)
    y = np.arange(1, len(x) + 1, dtype=np.float64) / len(x)
    ax2.plot(x, y, color='steelblue', linewidth=1.2)
    for q, lab in zip(qs, ['2.5%', '25%', '75%', '97.5%']):
        ax2.axvline(q, color='gray', linestyle=':', alpha=0.7)
    ax2.set_xlabel('HI (ground truth)')
    ax2.set_ylabel('cumulative probability')
    ax2.set_title('Empirical CDF')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f'H1N1 HI labels  (n={len(arr)}, min={arr.min():.4g}, max={arr.max():.4g}, std={arr.std():.4g})',
        fontsize=11,
        y=1.02,
    )
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    dataset = H1N1Dataset(
        root=str(ROOT / 'Datasets'),
        name='H1N1',
        csv_path=str(ROOT / 'Datasets/ori_data/a_h1n1_hi_folds.csv'),
    )
    print(f'H1N1Dataset 样本数: {len(dataset)}\n')

    arr = labels_from_dataset(dataset)

    print('=== HI (ground truth) 数值范围 ===')
    print(f'  min     : {arr.min():.6g}')
    print(f'  max     : {arr.max():.6g}')
    print(f'  mean    : {arr.mean():.6g}')
    print(f'  std     : {arr.std(ddof=0):.6g}')
    print(f'  median  : {np.median(arr):.6g}')
    qs = np.quantile(arr, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    print(f'  q0/q5/q25/q50/q75/q95/q100 : {qs}')

    uniq = np.unique(arr)
    print(f'\n=== 离散度 ===')
    print(f'  唯一取值个数 : {len(uniq)}')
    if len(uniq) <= 20:
        for v in uniq:
            c = (arr == v).sum()
            print(f'    hi={v:g}  ->  {int(c)}')

    print('\n=== 分箱频数 (10 等宽区间 [min, max]) ===')
    counts, edges = np.histogram(arr, bins=10)
    mx = counts.max() if counts.max() > 0 else 1
    for i in range(len(counts)):
        lo, hi = edges[i], edges[i + 1]
        bar = '#' * max(1, int(60 * counts[i] / mx))
        print(f'  [{lo:10.4g}, {hi:10.4g})  {counts[i]:6d}  {bar}')

    print('\n=== 二元占比 (hi==0 vs hi!=0，若适用) ===')
    print(f'  hi == 0  : {(arr == 0).sum()} ({100 * (arr == 0).mean():.2f}%)')
    print(f'  hi != 0  : {(arr != 0).sum()} ({100 * (arr != 0).mean():.2f}%)')

    out = ROOT / 'Figures' / 'hi_distribution.png'
    plot_distribution(arr, out)
    print(f'\n图表已保存: {out}')


if __name__ == '__main__':
    main()
