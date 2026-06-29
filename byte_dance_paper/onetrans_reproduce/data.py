"""
OneTrans 数据模块

支持两种数据来源，通过 mode 参数切换：
  mode="mock"    → 随机生成模拟数据（无需任何文件，用于快速验证模型）
  mode="parquet" → 从 Hive 导出的 Parquet 文件流式加载（正式训练用）

特征配置支持两种方式（优先级：yaml 文件 > 代码内默认值）：
  1. 通过 feature_config.yaml 外部配置（推荐，无需改代码）
  2. 直接修改本文件底部的 NS_NUM_COLS / SEQ_CONFIGS / LABEL_COLS 默认值

Hive 导出参考：
    SET mapreduce.job.reduces=100;
    INSERT OVERWRITE DIRECTORY '/user/yourname/rec_train/'
    STORED AS PARQUET
    SELECT ... FROM rec_training_data WHERE dt='20240101';

    hdfs dfs -get /user/yourname/rec_train/ ./data/parquet/train/
"""

import glob
import math
import os
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


# ---------------------------------------------------------------------------
# YAML 配置加载
# ---------------------------------------------------------------------------

def load_feature_config(config_path: str) -> Tuple[List, List, List]:
    """
    从 YAML 文件加载特征配置，返回 (ns_feature_specs, seq_configs, label_cols)。

    ns_feature_specs: List[dict]，每条包含完整的特征描述：
        continuous:   {"name", "type", "scale"}
        discrete_id:  {"name", "type", "vocab_size", "emb_dim"}
        discrete_str: {"name", "type", "vocab_size", "emb_dim"}
        multihot:     {"name", "type", "vocab_size", "emb_dim", "max_len", "pooling"}

    seq_configs: List[dict]，每条结构：
        {"name": str, "timestamps": str|None, "max_len": int,
         "fields": [{"col": str, "scale": float}, ...]}
    label_cols:  List[str]
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("加载 YAML 配置需要安装 PyYAML：pip install pyyaml")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 解析 ns_features（保留完整 spec，供 data 层和 model 层各自使用）
    ns_feature_specs = []
    for item in cfg.get("ns_features", []):
        feat_type = item.get("type", "continuous")
        spec = {"name": item["name"], "type": feat_type}
        if feat_type == "continuous":
            spec["scale"] = float(item.get("scale", 1.0))
        elif feat_type in ("discrete_id", "discrete_str"):
            spec["vocab_size"] = int(item["vocab_size"])
            spec["emb_dim"]    = int(item.get("emb_dim", 16))
        elif feat_type == "multihot":
            spec["vocab_size"] = int(item["vocab_size"])
            spec["emb_dim"]    = int(item.get("emb_dim", 16))
            spec["max_len"]    = int(item.get("max_len", 10))
            spec["pooling"]    = item.get("pooling", "mean")
        else:
            raise ValueError(f"未知特征类型 type={feat_type!r}，支持: continuous/discrete_id/discrete_str/multihot")
        ns_feature_specs.append(spec)

    # 解析 sequences
    # 新格式：每条序列用 fields 列表声明子字段，支持任意字段组合
    # seq_config 结构：{"name", "timestamps", "max_len", "fields": [{"col", "scale"}, ...]}
    seq_configs = []
    for seq in cfg.get("sequences", []):
        fields = []
        for f in seq.get("fields", []):
            fields.append({
                "col":   f["col"],
                "scale": float(f.get("scale", 1.0)),
            })
        seq_configs.append({
            "name":       seq.get("name", ""),
            "timestamps": seq.get("timestamps"),
            "max_len":    int(seq.get("max_len", 50)),
            "fields":     fields,
        })

    # 解析 labels
    label_cols = cfg.get("labels", ["ctr_label", "cvr_label"])

    return ns_feature_specs, seq_configs, label_cols


def ns_specs_to_total_dim(ns_feature_specs: List) -> int:
    """
    根据 ns_feature_specs 计算拼接后的总向量维度。
    - continuous:   贡献 1 维
    - discrete_id / discrete_str / multihot: 贡献 emb_dim 维
    """
    total = 0
    for spec in ns_feature_specs:
        if isinstance(spec, dict):
            if spec["type"] == "continuous":
                total += 1
            else:
                total += spec["emb_dim"]
        else:
            # 兼容旧版 (name, scale) 元组格式
            total += 1
    return total


# ---------------------------------------------------------------------------
# 代码内默认字段配置（当不使用 YAML 时生效）
# ---------------------------------------------------------------------------

# 非序列特征：(Hive字段名, 归一化分母)，scale=-1 表示 log(x+1) 归一化
NS_NUM_COLS: List[Tuple[str, float]] = [
    ("user_age_bucket", 10.0),
    ("user_city",       1000.0),
    ("user_7d_clicks",  100.0),
    ("item_cat",        20.0),
    ("item_price",      1000.0),
    ("item_ctr_7d",     1.0),
    ("hour",            24.0),
    ("day_of_week",     7.0),
]

# 行为序列默认配置（dict 格式，与 YAML 解析结果保持一致）
SEQ_CONFIGS: List[Dict] = [
    {"name": "click",    "timestamps": "click_seq_ts", "max_len": 50, "fields": [
        {"col": "click_seq_ids",    "scale": 100000.0},
        {"col": "click_seq_cats",   "scale": 100.0},
        {"col": "click_seq_prices", "scale": 1000.0},
    ]},
    {"name": "purchase", "timestamps": "buy_seq_ts",   "max_len": 10, "fields": [
        {"col": "buy_seq_ids",      "scale": 100000.0},
    ]},
    {"name": "cart",     "timestamps": "cart_seq_ts",  "max_len": 20, "fields": [
        {"col": "cart_seq_ids",     "scale": 100000.0},
        {"col": "cart_seq_cats",    "scale": 100.0},
    ]},
]

# 标签字段
LABEL_COLS: List[str] = ["ctr_label", "cvr_label"]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def collate_fn(batch: List[Dict]) -> Dict:
    """通用 collate，兼容模拟数据和 Parquet 数据"""
    n_seqs = len(batch[0]["sequences"])
    result = {
        "sequences":  [torch.stack([b["sequences"][i] for b in batch]) for i in range(n_seqs)],
        "timestamps": [torch.stack([b["timestamps"][i] for b in batch]) for i in range(n_seqs)],
        "seq_masks":  [torch.stack([b["seq_masks"][i]  for b in batch]) for i in range(n_seqs)],
    }
    # ns_inputs 是 dict of tensors，按 key 分别 stack
    ns_keys = batch[0]["ns_inputs"].keys()
    result["ns_inputs"] = {k: torch.stack([b["ns_inputs"][k] for b in batch]) for k in ns_keys}

    for key in batch[0]:
        if key not in ("ns_inputs", "sequences", "timestamps", "seq_masks"):
            result[key] = torch.stack([b[key] for b in batch])
    return result


# ---------------------------------------------------------------------------
# 模式一：模拟数据集（mock）
# ---------------------------------------------------------------------------

class MockRecDataset(Dataset):
    """
    模拟酒店推荐场景的 mock 数据集，特征分布和标签逻辑尽量贴近真实业务。

    特征设计（对应 seq_dims=[3,3,2]，seq_lengths=[50,10,20]）：
      NS 特征（非序列）：
        feat_0  用户活跃度分（归一化，越高越容易点击）
        feat_1  候选酒店价格分（归一化，中等价格 CTR 最高）
        feat_2  用户-酒店城市匹配（0/1，匹配则 CTR 更高）
        feat_3  候选酒店评分（归一化，越高越好）
        feat_4  候选酒店距离分（归一化，越近越好）
        feat_5  用户历史消费等级（归一化）
        feat_6  展示时间段（归一化，晚上高峰期 CTR 更高）
        feat_7  候选酒店库存紧张度（归一化，紧张则 CVR 更高）
        ...其余特征为噪声

      序列特征（3 种行为序列）：
        click_seq  (L=50, dim=3): [hotel_id_emb, price_norm, star_norm]  点击序列
        buy_seq    (L=10, dim=3): [hotel_id_emb, price_norm, star_norm]  购买序列
        cart_seq   (L=20, dim=2): [hotel_id_emb, price_norm]             加购序列

      标签：
        ctr_label: 点击，由用户活跃度 + 价格匹配 + 城市匹配 + 历史偏好共同决定
        cvr_label: 下单，在点击基础上还受库存紧张度、价格影响

    ns_inputs 格式：{"feat_0": tensor(B,), "feat_1": tensor(B,), ...}
    """

    # 语义化特征名，方便理解
    NS_FEAT_NAMES = [
        "user_active_score",    # 用户活跃度
        "hotel_price_score",    # 酒店价格匹配分（中等最优）
        "city_match",           # 城市匹配
        "hotel_star_score",     # 酒店星级评分
        "hotel_dist_score",     # 距离分（越近越好）
        "user_consume_level",   # 用户消费等级
        "time_peak_score",      # 展示时段（晚上高峰）
        "hotel_urgency",        # 库存紧张度
    ]

    def __init__(
        self,
        n_samples: int,
        ns_dim: int,
        seq_lengths: List[int],
        seq_dims: List[int],
        label_cols: List[str] = None,
        ctr_pos_ratio: float = 0.15,   # 真实 CTR 约 5~20%
        cvr_pos_ratio: float = 0.08,   # 真实 CVR 约 3~10%
        seed: int = 42,
    ):
        self.seq_lengths = seq_lengths
        self.seq_dims    = seq_dims
        self.label_cols  = label_cols or ["ctr_label", "cvr_label"]

        rng = np.random.default_rng(seed)
        N   = n_samples

        # ── 构造语义化 NS 特征 ──────────────────────────────────────────────
        # 前 8 个有语义，其余补噪声，总维度 = ns_dim
        n_semantic = len(self.NS_FEAT_NAMES)
        ns_semantic = np.zeros((N, n_semantic), dtype=np.float32)

        # 用户活跃度 [0,1]，高斯混合：低活跃(0.3) + 高活跃(0.7) 两类用户
        user_type = rng.binomial(1, 0.4, N)  # 40% 高活跃用户
        ns_semantic[:, 0] = np.where(
            user_type,
            rng.normal(0.75, 0.1, N),
            rng.normal(0.25, 0.1, N),
        ).clip(0, 1)

        # 酒店价格匹配分：中等价格(0.5)最优，过高/过低都差，用倒 U 形
        price_raw = rng.beta(2, 2, N).astype(np.float32)  # 集中在 0.3~0.7
        ns_semantic[:, 1] = price_raw

        # 城市匹配：二值，60% 匹配
        ns_semantic[:, 2] = rng.binomial(1, 0.6, N).astype(np.float32)

        # 酒店星级评分 [0,1]，偏高分分布
        ns_semantic[:, 3] = rng.beta(3, 1.5, N).astype(np.float32)

        # 距离分 [0,1]，越近越好，指数分布截断
        ns_semantic[:, 4] = np.exp(-rng.exponential(0.5, N)).astype(np.float32).clip(0, 1)

        # 用户消费等级 [0,1]，与用户活跃度正相关
        ns_semantic[:, 5] = (ns_semantic[:, 0] * 0.6 + rng.uniform(0, 0.4, N)).clip(0, 1).astype(np.float32)

        # 时段分：晚上高峰(0.8)、白天(0.4)、凌晨(0.1)
        time_slot = rng.choice([0, 1, 2], N, p=[0.5, 0.35, 0.15])
        ns_semantic[:, 6] = np.where(time_slot == 0, 0.8,
                             np.where(time_slot == 1, 0.4, 0.1)).astype(np.float32)

        # 库存紧张度 [0,1]，影响 CVR
        ns_semantic[:, 7] = rng.beta(1.5, 3, N).astype(np.float32)

        # 补齐噪声特征到 ns_dim
        if ns_dim > n_semantic:
            ns_noise = rng.standard_normal((N, ns_dim - n_semantic)).astype(np.float32) * 0.05
            ns_all = np.concatenate([ns_semantic, ns_noise], axis=1)
        else:
            ns_all = ns_semantic[:, :ns_dim]

        # 特征名列表（超出语义名的用 feat_N 补齐）
        feat_names = self.NS_FEAT_NAMES[:ns_dim]
        if ns_dim > n_semantic:
            feat_names = feat_names + [f"feat_{i}" for i in range(n_semantic, ns_dim)]

        self.ns_feature_specs = [
            {"name": name, "type": "continuous", "scale": 1.0}
            for name in feat_names
        ]
        self._ns_tensors = {
            name: torch.FloatTensor(ns_all[:, i])
            for i, name in enumerate(feat_names)
        }

        # ── 构造序列特征（模拟历史行为）──────────────────────────────────────
        # 每条序列的 dim 含义：[hotel_embedding(1), price_norm(1), star_norm(1)] 或子集
        # hotel_embedding 用随机向量模拟（实际是 embedding lookup 后的值）
        # 序列中最近的几个 hotel 与候选 hotel 的相似度影响 CTR
        self.sequences = []
        self.timestamps = []
        self.seq_masks  = []

        for L, dim in zip(seq_lengths, seq_dims):
            # 序列特征：price 和 star 与候选酒店有一定相关性（模拟用户偏好一致性）
            seq_price = rng.beta(2, 2, (N, L)).astype(np.float32)  # 历史点击价格分布
            seq_star  = rng.beta(3, 1.5, (N, L)).astype(np.float32)

            if dim == 1:
                seq_feat = seq_price[:, :, np.newaxis]
            elif dim == 2:
                seq_feat = np.stack([seq_price, seq_star], axis=2)
            else:
                # dim >= 3：第一维用小随机数模拟 hotel id embedding
                seq_id = rng.standard_normal((N, L, dim - 2)).astype(np.float32) * 0.05
                seq_feat = np.concatenate([seq_id, seq_price[:, :, np.newaxis],
                                           seq_star[:, :, np.newaxis]], axis=2)

            self.sequences.append(torch.FloatTensor(seq_feat))

            # 时间戳：相对时间差，降序（早→近），最近事件接近 0
            ts_raw = np.sort(
                rng.uniform(0.01, 1.0, (N, L)).astype(np.float32), axis=1
            )[:, ::-1].copy()
            self.timestamps.append(torch.FloatTensor(ts_raw))

            # padding mask：模拟真实数据中序列长度不足的情况
            # 80% 样本序列填满，20% 有 padding（长度为 L//2 ~ L）
            actual_lens = np.where(
                rng.uniform(0, 1, N) < 0.8,
                L,
                rng.integers(L // 2, L, N),
            )
            mask = np.zeros((N, L), dtype=bool)
            for j, al in enumerate(actual_lens):
                mask[j, :al] = True
            self.seq_masks.append(torch.BoolTensor(mask))

        # ── 构造标签（多因子打分，模拟真实 CTR/CVR 逻辑）────────────────────
        # CTR 影响因子：
        #   + 用户活跃度（最重要）
        #   + 城市匹配
        #   + 酒店评分
        #   + 时段（晚上高峰）
        #   + 价格适中（倒 U 形：中等最优）
        #   - 距离远（负向）
        price_fit    = 1 - 2 * np.abs(ns_semantic[:, 1] - 0.5)  # 中等价格最优 [-1,1]→[0,1]
        ctr_logit = (
            2.0 * ns_semantic[:, 0]       # 用户活跃度（权重最高）
            + 1.5 * ns_semantic[:, 2]     # 城市匹配
            + 1.0 * ns_semantic[:, 3]     # 酒店评分
            + 0.8 * ns_semantic[:, 6]     # 时段
            + 0.6 * price_fit             # 价格适中
            - 0.5 * (1 - ns_semantic[:, 4])  # 距离惩罚
            + rng.normal(0, 0.3, N)       # 随机噪声
            - 2.5                         # 偏置，控制正样本率
        )
        ctr_prob = 1 / (1 + np.exp(-ctr_logit))

        # CVR 影响因子（在 CTR 基础上，更受价格和库存影响）
        cvr_logit = (
            ctr_logit * 0.7               # 继承 CTR 信号
            + 1.2 * ns_semantic[:, 7]     # 库存紧张（稀缺性促转化）
            + 0.8 * ns_semantic[:, 5]     # 用户消费等级
            - 0.5 * ns_semantic[:, 1]     # 高价抑制转化
            + rng.normal(0, 0.3, N)
            - 1.0                         # CVR 比 CTR 更低
        )
        cvr_prob = 1 / (1 + np.exp(-cvr_logit))

        self.labels = {}
        for col in self.label_cols:
            prob = ctr_prob if "ctr" in col else cvr_prob
            self.labels[col] = torch.FloatTensor(
                (rng.uniform(0, 1, N) < prob).astype(np.float32)
            )

        # 打印正样本率，方便验证
        for col in self.label_cols:
            rate = self.labels[col].mean().item()
            print(f"[MockData] {col} 正样本率: {rate:.3f}")

    @property
    def ns_dim(self) -> int:
        return len(self.ns_feature_specs)

    def __len__(self):
        return self.sequences[0].size(0)

    def __getitem__(self, idx):
        return {
            "ns_inputs":  {k: v[idx] for k, v in self._ns_tensors.items()},
            "sequences":  [s[idx] for s in self.sequences],
            "timestamps": [t[idx] for t in self.timestamps],
            "seq_masks":  [m[idx] for m in self.seq_masks],
            **{col: self.labels[col][idx] for col in self.label_cols},
        }


# ---------------------------------------------------------------------------
# 辅助：校验 seq_configs 格式
# ---------------------------------------------------------------------------

def _normalize_seq_configs(seq_configs: List) -> List[Dict]:
    """
    确保 seq_configs 中每条都是标准 dict 格式：
      {"name", "timestamps", "max_len", "fields": [{"col", "scale"}, ...]}
    """
    normalized = []
    for cfg in seq_configs:
        if isinstance(cfg, dict):
            normalized.append(cfg)
        else:
            raise ValueError(
                f"seq_configs 格式已升级为 dict，不再支持元组格式。\n"
                f"请使用 feature_config.yaml 或参考 SEQ_CONFIGS 默认值。\n"
                f"收到: {cfg}"
            )
    return normalized


# ---------------------------------------------------------------------------
# 模式二：Parquet 流式数据集（parquet）
# ---------------------------------------------------------------------------

class ParquetRecDataset(IterableDataset):
    """
    流式读取多个 Parquet 文件，每次只把一个文件加载进内存。
    多进程时自动按 worker 数量切分文件，避免重复读取。
    """

    def __init__(
        self,
        data_dir: str,
        ns_cols: List = None,
        seq_configs: List[Dict] = None,
        label_cols: List[str] = None,
        shuffle: bool = True,
    ):
        self.files      = sorted(glob.glob(os.path.join(data_dir, "*.parquet")))
        self.ns_cols    = ns_cols      or NS_NUM_COLS
        self.seq_cfgs   = _normalize_seq_configs(seq_configs or SEQ_CONFIGS)
        self.label_cols = label_cols   or LABEL_COLS
        self.shuffle    = shuffle

        assert len(self.files) > 0, (
            f"在 {data_dir} 下没有找到 .parquet 文件\n"
            f"请先从 Hive 导出并下载：hdfs dfs -get /your/path/ {data_dir}"
        )
        print(f"[Parquet] 找到 {len(self.files)} 个文件，"
              f"NS维度={self.ns_dim}，序列数={len(self.seq_cfgs)}")
        # 扫描第一个文件做缺失率统计（P2-3）
        self._check_missing(self.files[0])

    # ── 供外部读取模型配置 ──
    @property
    def ns_feature_specs(self) -> List[Dict]:
        """
        返回标准 dict 格式的特征 spec 列表，供 NSFeatureEncoder 使用。
        兼容旧版 (name, scale) 元组，自动转换为 continuous dict。
        """
        specs = []
        for s in self.ns_cols:
            if isinstance(s, tuple):
                specs.append({"name": s[0], "type": "continuous", "scale": s[1]})
            else:
                specs.append(s)
        return specs

    @property
    def ns_dim(self) -> int:
        """ns_features 向量总维度（continuous=1维，discrete/multihot=emb_dim维）"""
        return ns_specs_to_total_dim(self.ns_cols)

    @property
    def seq_dims(self) -> List[int]:
        """每条序列的向量维度 = 该序列声明的 fields 数量"""
        return [len(cfg["fields"]) for cfg in self.seq_cfgs]

    @property
    def seq_lengths(self) -> List[int]:
        return [cfg["max_len"] for cfg in self.seq_cfgs]

    def _check_missing(self, filepath: str):
        """扫描第一个文件，打印各字段缺失率，缺失率>50%时发出警告（P2-3）"""
        df = pd.read_parquet(filepath)
        ns_names  = [s["name"] if isinstance(s, dict) else s[0] for s in self.ns_cols]
        # 收集所有序列子字段名和时间戳字段名
        seq_names = []
        for cfg in self.seq_cfgs:
            seq_names.extend(f["col"] for f in cfg["fields"])
            if cfg.get("timestamps"):
                seq_names.append(cfg["timestamps"])
        all_cols  = ns_names + seq_names + self.label_cols
        print("\n[数据检查] 字段缺失率（基于第一个文件）：")
        warned = False
        for col in all_cols:
            if col not in df.columns:
                print(f"  ⚠️  {col:<30} 字段不存在！")
                warned = True
            else:
                miss_rate = df[col].isna().mean()
                flag = " ⚠️  缺失率过高" if miss_rate > 0.5 else ""
                print(f"  {col:<30} 缺失率={miss_rate:.1%}{flag}")
                if miss_rate > 0.5:
                    warned = True
        if warned:
            print("[数据检查] 存在警告，请确认字段名和数据是否正确\n")
        else:
            print("[数据检查] 全部字段正常 ✓\n")

    def __iter__(self):
        files = self.files.copy()
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            files = files[worker_info.id :: worker_info.num_workers]
        if self.shuffle:
            random.shuffle(files)

        for filepath in files:
            df = pd.read_parquet(filepath)
            if self.shuffle:
                df = df.sample(frac=1).reset_index(drop=True)
            # P0-2：用向量化批处理替代 iterrows()
            yield from self._process_dataframe(df)

    def _process_dataframe(self, df: pd.DataFrame):
        """向量化处理整个 DataFrame，逐行 yield（避免 iterrows 的 Python 对象开销）"""
        # 预先提取所有列为 numpy array，避免每行重复 dict 查找
        col_arrays = {col: df[col].values for col in df.columns}
        n = len(df)

        # 预计算时间戳参考点（每个序列取最大时间戳作为"当前时间"，P0-3）
        ts_refs = {}
        for cfg in self.seq_cfgs:
            col_ts = cfg.get("timestamps")
            if col_ts and col_ts in col_arrays:
                ts_refs[col_ts] = self._extract_ts_ref(col_arrays[col_ts], n)

        for i in range(n):
            ns_inputs = self._build_ns_row(col_arrays, i)
            seqs, tss, masks = self._build_seqs_row(col_arrays, i, ts_refs)
            labels = {col: torch.tensor(float(col_arrays[col][i])) for col in self.label_cols}
            yield {"ns_inputs": ns_inputs, "sequences": seqs, "timestamps": tss,
                   "seq_masks": masks, **labels}

    def _extract_ts_ref(self, col_arr, n: int) -> np.ndarray:
        """
        提取每行序列的最大时间戳（作为相对时间的参考点）。
        向量化实现：先把所有行展开为 DataFrame，再用 apply 取最大值。
        """
        def row_max(raw):
            if raw is None or (isinstance(raw, float) and np.isnan(raw)):
                return 0.0
            vals = [float(x) for x in str(raw).split(",") if x.strip()]
            return max(vals) if vals else 0.0

        # 用 pandas Series.map 替代 Python for 循环，性能提升约 3-5x
        refs = pd.Series(col_arr).map(row_max).to_numpy(dtype=np.float64)
        return refs

    def _build_ns_row(self, col_arrays: Dict, i: int) -> Dict[str, torch.Tensor]:
        """构建单行的 ns_inputs 字典（P0-1：discrete 只返回整数 ID）"""
        result = {}
        for spec in self.ns_cols:
            if isinstance(spec, tuple):
                # 旧版元组兼容：连续特征
                col, scale = spec
                raw = float(col_arrays[col][i]) if col in col_arrays and not _is_nan(col_arrays[col][i]) else 0.0
                val = math.log1p(max(raw, 0.0)) if scale == -1 else raw / scale
                result[col] = torch.tensor(val, dtype=torch.float32)
                continue

            col   = spec["name"]
            ftype = spec["type"]
            raw   = col_arrays[col][i] if col in col_arrays else None
            is_missing = raw is None or _is_nan(raw)

            if ftype == "continuous":
                scale = spec["scale"]
                v = float(raw) if not is_missing else 0.0
                val = math.log1p(max(v, 0.0)) if scale == -1 else v / scale
                result[col] = torch.tensor(val, dtype=torch.float32)

            elif ftype in ("discrete_id", "discrete_str"):
                idx = int(float(raw)) if not is_missing else 0
                idx = max(0, min(idx, spec["vocab_size"] - 1))
                result[col] = torch.tensor(idx, dtype=torch.long)

            elif ftype == "multihot":
                max_len = spec["max_len"]
                vocab_size = spec["vocab_size"]
                if not is_missing:
                    ids = [int(float(x)) for x in str(raw).split(",") if x.strip()][:max_len]
                    ids = [max(0, min(x, vocab_size - 1)) for x in ids]
                else:
                    ids = []
                # padding 到 max_len，用 0 填充（0 是 padding_idx）
                ids += [0] * (max_len - len(ids))
                result[col] = torch.tensor(ids, dtype=torch.long)

        return result

    def _build_seqs_row(self, col_arrays: Dict, i: int, ts_refs: Dict):
        """
        构建单行的序列特征。
        每条序列按 cfg["fields"] 动态拼向量，支持任意字段组合。
        """
        seqs, tss, masks = [], [], []
        for cfg in self.seq_cfgs:
            fields   = cfg["fields"]
            col_ts   = cfg.get("timestamps")
            max_len  = cfg["max_len"]

            def parse_field(col, scale, _max_len=max_len):
                """解析一个逗号分隔字符串列，返回归一化后的 float 列表"""
                if col not in col_arrays or _is_nan(col_arrays[col][i]):
                    return []
                vals = [float(x) / scale for x in str(col_arrays[col][i]).split(",") if x.strip()]
                return vals[-_max_len:]

            # 按 fields 顺序解析每个子字段，以第一个字段的有效长度作为 actual_len
            field_vals = [parse_field(f["col"], f["scale"]) for f in fields]
            actual_len = len(field_vals[0]) if field_vals else 0

            # P0-3：时间戳改为相对时间差（ref - t）/ ref，越近越接近 0
            ts_raw = []
            if col_ts and col_ts in col_arrays and not _is_nan(col_arrays[col_ts][i]):
                ts_raw = [float(x) for x in str(col_arrays[col_ts][i]).split(",") if x.strip()]
                ts_raw = ts_raw[-max_len:]
                ref = ts_refs.get(col_ts, np.zeros(1))[i]
                ts_raw = [(ref - t) / ref for t in ts_raw] if ref > 0 else [0.0] * len(ts_raw)

            # padding 到 max_len
            pad = max_len - actual_len
            mask = [True] * actual_len + [False] * pad

            parts = []
            for vals in field_vals:
                vals_padded = vals + [0.0] * (max_len - len(vals))
                parts.append(torch.tensor(vals_padded, dtype=torch.float32).unsqueeze(1))
            ts_raw += [0.0] * (max_len - len(ts_raw))

            seqs.append(torch.cat(parts, dim=1))          # shape: (max_len, num_fields)
            tss.append(torch.tensor(ts_raw, dtype=torch.float32))
            masks.append(torch.tensor(mask, dtype=torch.bool))

        return seqs, tss, masks


def _is_nan(val) -> bool:
    """安全的 NaN 检测，兼容 None、float nan、字符串"""
    if val is None:
        return True
    try:
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 统一入口：get_dataloader
# ---------------------------------------------------------------------------

def get_dataloader(
    mode: str,
    batch_size: int = 256,
    num_workers: int = 0,
    shuffle: bool = True,
    # 外部 YAML 特征配置（优先级最高）
    feature_config: str = None,
    # mock 模式参数
    n_samples: int = 10000,
    ns_dim: int = 160,
    seq_lengths: List[int] = None,
    seq_dims: List[int] = None,
    label_cols: List[str] = None,
    seed: int = 42,
    # parquet 模式参数
    data_dir: str = None,
    ns_cols: List = None,
    seq_configs: List[Dict] = None,
) -> Tuple[DataLoader, object]:
    """
    统一数据加载入口。

    Args:
        mode:           "mock" 使用模拟数据；"parquet" 从文件加载
        feature_config: YAML 配置文件路径（优先级高于代码内默认值）
        返回 (DataLoader, dataset)，dataset 上有 ns_dim / seq_dims / seq_lengths 属性
    """
    # 如果提供了 YAML 配置，从文件加载字段定义（仅 parquet 模式有效）
    if feature_config is not None:
        if mode == "mock":
            print(f"[Config] 注意：mock 模式下 feature_config 不生效，使用随机生成数据")
        else:
            ns_cols, seq_configs, label_cols = load_feature_config(feature_config)
            total_dim = ns_specs_to_total_dim(ns_cols)
            print(f"[Config] 从 {feature_config} 加载特征配置："
                  f" NS={len(ns_cols)} 个字段(总维度={total_dim}), "
                  f"序列={len(seq_configs)} 组, 标签={label_cols}")

    seq_lengths = seq_lengths or [50, 10, 20]
    seq_dims    = seq_dims    or [32, 32, 32]
    label_cols  = label_cols  or LABEL_COLS

    if mode == "mock":
        dataset = MockRecDataset(
            n_samples=n_samples,
            ns_dim=ns_dim,
            seq_lengths=seq_lengths,
            seq_dims=seq_dims,
            label_cols=label_cols,
            seed=seed,
        )
        # MockRecDataset 已有 ns_dim 属性，补充其余属性与 parquet 模式对齐
        dataset.seq_dims    = seq_dims
        dataset.seq_lengths = seq_lengths

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    elif mode == "parquet":
        assert data_dir is not None, "parquet 模式需要指定 data_dir"
        dataset = ParquetRecDataset(
            data_dir=data_dir,
            ns_cols=ns_cols,
            seq_configs=seq_configs,
            label_cols=label_cols,
            shuffle=shuffle,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
            pin_memory=torch.cuda.is_available(),
        )

    else:
        raise ValueError(f"mode 只支持 'mock' 或 'parquet'，收到: {mode!r}")

    return loader, dataset
