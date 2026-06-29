"""
Parquet 文件格式检查脚本

在正式训练前运行，验证：
1. 文件可以正常读取
2. feature_config.yaml 中声明的字段都存在
3. 数据格式符合模型输入要求（缺失率、类型）

用法：
    python check_parquet.py --data_dir ./data/parquet/train
    python check_parquet.py --data_dir ./data/parquet/train --feature_config feature_config.yaml
"""

import argparse
import glob
import os

import pandas as pd
from torch.utils.data import DataLoader

from data import (
    NS_NUM_COLS, SEQ_CONFIGS, LABEL_COLS,
    ParquetRecDataset, collate_fn, load_feature_config,
)


def check(data_dir: str, feature_config: str = None):
    files = glob.glob(os.path.join(data_dir, "*.parquet"))
    print(f"找到 {len(files)} 个 Parquet 文件")
    assert len(files) > 0, "没有找到文件！"

    # ── 加载特征配置 ──
    if feature_config:
        ns_specs, seq_configs, label_cols = load_feature_config(feature_config)
        print(f"使用配置文件: {feature_config}")
        print(f"  NS 特征: {len(ns_specs)} 个，序列: {len(seq_configs)} 组，标签: {label_cols}")
    else:
        ns_specs    = NS_NUM_COLS
        seq_configs = SEQ_CONFIGS
        label_cols  = LABEL_COLS
        print("未指定 feature_config，使用 data.py 内默认字段配置")

    # ── 检查第一个文件的字段 ──
    df = pd.read_parquet(files[0])
    print(f"\n第一个文件: {files[0]}")
    print(f"行数: {len(df)}, 列数: {len(df.columns)}")
    print(f"列名: {df.columns.tolist()}")
    print(f"\n前2行预览:")
    print(df.head(2).to_string())

    # ── 检查必要字段 ──
    ns_names = [s["name"] if isinstance(s, dict) else s[0] for s in ns_specs]
    # seq_configs 为 dict 格式：{"name", "timestamps", "max_len", "fields": [...]}
    seq_field_cols = [f["col"] for cfg in seq_configs for f in cfg["fields"]]
    seq_ts         = [cfg["timestamps"] for cfg in seq_configs if cfg.get("timestamps")]
    required       = ns_names + seq_field_cols + seq_ts + label_cols

    print(f"\n── 字段存在性检查 ──")
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"缺少字段: {missing}")
        print("请检查 feature_config.yaml 中的字段名是否与 Hive 表一致")
        return False
    print("所有必要字段存在 ✓")

    # ── 缺失率检查 ──
    print(f"\n── 字段缺失率 ──")
    warned = False
    for col in required:
        miss = df[col].isna().mean()
        flag = " ⚠️  过高" if miss > 0.5 else ""
        print(f"  {col:<35} {miss:.1%}{flag}")
        if miss > 0.5:
            warned = True
    if warned:
        print("存在缺失率过高的字段，请确认数据是否正确")

    # ── 类型检查（discrete 特征应为整数）──
    print(f"\n── 离散特征类型检查 ──")
    for spec in ns_specs:
        if not isinstance(spec, dict):
            continue
        if spec["type"] in ("discrete_id", "discrete_str"):
            col = spec["name"]
            if col in df.columns:
                sample = df[col].dropna().head(5).tolist()
                try:
                    [int(float(x)) for x in sample]
                    print(f"  {col:<35} 整数 ✓  样例: {sample[:3]}")
                except (ValueError, TypeError):
                    print(f"  {col:<35} ⚠️  无法转为整数，样例: {sample[:3]}")
                    print(f"    → 请在 Hive 中用 dense_rank() 预处理为整数")

    # ── 尝试加载一个 batch ──
    print("\n── 尝试加载一个 batch ──")
    dataset = ParquetRecDataset(
        data_dir, ns_cols=ns_specs, seq_configs=seq_configs,
        label_cols=label_cols, shuffle=False,
    )
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=0)
    batch  = next(iter(loader))

    print(f"ns_inputs keys    : {list(batch['ns_inputs'].keys())[:5]} ...")
    for i, seq in enumerate(batch["sequences"]):
        print(f"sequences[{i}] shape: {seq.shape}")
    for i, ts in enumerate(batch["timestamps"]):
        print(f"timestamps[{i}] shape: {ts.shape}  range=[{ts.min():.3f}, {ts.max():.3f}]")
    for i, mask in enumerate(batch["seq_masks"]):
        valid_ratio = mask.float().mean().item()
        print(f"seq_masks[{i}] shape: {mask.shape}  有效率={valid_ratio:.1%}")
    for col in label_cols:
        print(f"{col:<20} shape={batch[col].shape}, 正样本率={batch[col].mean():.3f}")

    print(f"\nNS 特征字段数:   {len(ns_specs)}")
    print(f"NS 向量总维度:   {dataset.ns_dim}")
    print(f"序列 event 维度: {dataset.seq_dims}")
    print(f"序列最大长度:    {dataset.seq_lengths}")
    print(f"\n格式检查全部通过，可以开始训练 ✓")
    return True


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",       type=str, required=True)
    p.add_argument("--feature_config", type=str, default=None,
                   help="YAML 特征配置文件（不指定则用 data.py 内默认配置）")
    args = p.parse_args()
    check(args.data_dir, args.feature_config)
