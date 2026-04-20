import os
import os.path as osp
import shutil

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset

from tqdm import tqdm


def VHSE_featurize(seqs):
    """
    将蛋白质序列转换为 VHSE 特征嵌入和类别 ID。
    """
    vhse = {}
    vhse_id = {}
    # 标准 20 种氨基酸的 VHSE 描述符 (8维)
    vhselis = [
        'A,0.15,-1.11,-1.35,-0.92,0.02,-0.91,0.36,-0.48',
        'R,-1.47,1.45,1.24,1.27,1.55,1.47,1.3,0.83',
        'N,-0.99,0,-0.37,0.69,-0.55,0.85,0.73,-0.8',
        'D,-1.15,0.67,-0.41,-0.01,-2.68,1.31,0.03,0.56',
        'C,0.18,-1.67,-0.46,-0.21,0,1.2,-1.61,-0.19',
        'Q,-0.96,0.12,0.18,0.16,0.09,0.42,-0.2,-0.41',
        'E,-1.18,0.4,0.1,0.36,-2.16,-0.17,0.91,0.02',
        'G,-0.2,-1.53,-2.63,2.28,-0.53,-1.18,2.01,-1.34',
        'H,-0.43,-0.25,0.37,0.19,0.51,1.28,0.93,0.65',
        'I,1.27,-0.14,0.3,-1.8,0.3,-1.61,-0.16,-0.13',
        'L,1.36,0.07,0.26,-0.8,0.22,-1.37,0.08,-0.62',
        'K,-1.17,0.7,0.7,0.8,1.64,0.67,1.63,0.13',
        'M,1.01,-0.53,0.43,0,0.23,0.1,-0.86,-0.68',
        'F,1.52,0.61,0.96,-0.16,0.25,0.28,-1.33,-0.2',
        'P,0.22,-0.17,-0.5,0.05,-0.01,-1.34,-0.19,3.56',
        'S,-0.67,-0.86,-1.07,-0.41,-0.32,0.27,-0.64,0.11',
        'T,-0.34,-0.51,-0.55,-1.06,-0.06,-0.01,-0.79,0.39',
        'W,1.5,2.06,1.79,0.75,0.75,-0.13,-1.01,-0.85',
        'Y,0.61,1.6,1.17,0.73,0.53,0.25,-0.96,-0.52',
        'V,0.76,-0.92,-0.17,-1.91,0.22,-1.4,-0.24,-0.03'
    ]

    cnt = 0
    for i in vhselis:
        lis = i.split(',')
        vhse[lis[0]] = np.array([float(k) for k in lis[1:]]).reshape(1, -1)
        vhse_id[lis[0]] = cnt
        cnt += 1

    all_embeds = []
    all_node_types = []

    print("Featurizing sequences...")
    for seq in tqdm(seqs):
        embed = []
        node_type = []
        for char in seq:

            embed.append(vhse[char].tolist())
            node_type.append(vhse_id[char])

        all_embeds.append(np.array(embed, dtype=np.float32).reshape(-1, 8))
        all_node_types.append(np.array(node_type, dtype=np.int64))

    return all_embeds, all_node_types


def _npy_paths(raw_dir, name):
    """返回 (embeds, labels, residue types) 的缓存路径；兼容旧版 h1n1_*.npy 命名。"""
    prefix = osp.join(raw_dir, f'{name}_')
    legacy_prefix = osp.join(raw_dir, 'h1n1_')
    embeds_path = prefix + 'embeds.npy'
    bds_path = prefix + 'bind.npy'
    node_types_path = prefix + 'node_types.npy'
    if os.path.exists(bds_path):
        return embeds_path, bds_path, node_types_path
    le, lb, ln = legacy_prefix + 'embeds.npy', legacy_prefix + 'bind.npy', legacy_prefix + 'node_types.npy'
    if os.path.exists(lb):
        return le, lb, ln
    return embeds_path, bds_path, node_types_path


def get_data(raw_dir, csv_path, name='H1N1'):
    """
    从 CSV 加载数据，合并序列并构建图结构，同时在 raw_dir 下缓存中间 npy（与 PyG raw 约定一致）。
    每个样本为单个 ``Data``：``x``、``edge_index``、``edge_attr``、``y``、``residue_type``。
    """
    embeds_path, bds_path, node_types_path = _npy_paths(raw_dir, name)

    if os.path.exists(bds_path):
        print(f"Loading cached numpy arrays from {raw_dir}...")
        embeds = np.load(embeds_path, allow_pickle=True)
        labels = np.load(bds_path, allow_pickle=True)
        node_types = np.load(node_types_path, allow_pickle=True)
    else:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cannot find data at {csv_path}")

        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        orig_len = len(df)

        df = df[~df['virus_seq'].str.contains('X', na=False)]
        df = df[~df['reference_seq'].str.contains('X', na=False)]

        new_len = len(df)
        print(f"Filtered out {orig_len - new_len} rows containing 'X'. Remaining: {new_len}")

        seqs = (df['virus_seq'] + df['reference_seq']).tolist()
        labels = np.array(df['hi'].astype(float).tolist())

        embeds, node_types = VHSE_featurize(seqs)

        print(f"Saving numpy arrays to {raw_dir}...")
        os.makedirs(raw_dir, exist_ok=True)
        np.save(embeds_path, np.array(embeds, dtype=object), allow_pickle=True)
        np.save(bds_path, labels, allow_pickle=True)
        np.save(node_types_path, np.array(node_types, dtype=object), allow_pickle=True)

    data_list = []
    print("Building graphs...")
    for i in tqdm(range(len(embeds))):
        x_np = embeds[i]
        res_np = node_types[i]
        n_nodes = x_np.shape[0]

        row = list(range(0, n_nodes - 1))
        col = list(range(1, n_nodes))
        edge_index_row = row + col
        edge_index_col = col + row
        edge_index = torch.tensor([edge_index_row, edge_index_col], dtype=torch.long)

        x = torch.as_tensor(x_np, dtype=torch.float)
        residue_type = torch.as_tensor(res_np, dtype=torch.long)

        et_row = residue_type[edge_index_row]
        et_col = residue_type[edge_index_col]
        edge_attr = et_row * 20 + et_col
        edge_attr = edge_attr.to(torch.long)

        y = torch.tensor([labels[i]], dtype=torch.float)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=y,
            residue_type=residue_type,
        )
        data_list.append(data)

    return data_list


class H1N1Dataset(InMemoryDataset):
    """
    目录结构::

        root/
          H1N1/
            raw/          # 放置 CSV（或通过 csv_path 在 download 阶段复制至此）
            processed/
              data.pt     # ``collate`` 后的缓存

    每个 ``Data`` 字段：
        - ``x``: 节点 VHSE 特征 ``[num_nodes, 8]``
        - ``edge_index``: ``long``，形状 ``[2, num_edges]``，序列线性相邻边（无向展开）
        - ``edge_attr``: 边类型编码 ``long``
        - ``y``: 标量回归标签 HI
        - ``residue_type``: 氨基酸类别索引 ``long[num_nodes]``（非语义上的 3D ``pos``）
    """

    def __init__(
        self,
        root,
        name='H1N1',
        csv_path=None,
        csv_name='a_h1n1_hi_folds.csv',
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.name = name
        self.csv_name = csv_name
        self._csv_src = osp.abspath(csv_path) if csv_path else None
        super().__init__(root, transform, pre_transform, pre_filter)
        try:
            out = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:
            out = torch.load(self.processed_paths[0])
        self.data, self.slices = out

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        return [self.csv_name]

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        if self._csv_src is None:
            return
        os.makedirs(self.raw_dir, exist_ok=True)
        dst = self.raw_paths[0]
        src = self._csv_src
        if not osp.isfile(src):
            return
        if not osp.isfile(dst) or osp.getmtime(src) > osp.getmtime(dst):
            shutil.copy2(src, dst)

    def process(self):
        data_list = get_data(self.raw_dir, self.raw_paths[0], name=self.name)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])






if __name__ == "__main__":
    CSV_PATH = 'Datasets/ori_data/a_h1n1_hi_folds.csv'
    ROOT_DIR = 'Datasets/'

    dataset = H1N1Dataset(root=ROOT_DIR, name='H1N1', csv_path=CSV_PATH)
    print(f"Dataset size: {len(dataset)}")
    print(f"First data sample: {dataset[0]}")
