"""
OneTrans Tokenizer 模块

实现论文 Section 3.2 中的两种 Tokenizer：
- Non-Sequential Tokenizer（Group-wise 和 Auto-Split）
- Sequential Tokenizer（timestamp-aware 和 timestamp-agnostic）

NS 特征编码流程：
  data.py  → 连续特征归一化为 float，离散特征转为整数 ID
  NSFeatureEncoder（本模块）→ Embedding lookup + concat → 统一向量
  AutoSplitNSTokenizer → MLP → L_NS 个 token
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# NS 特征编码器：负责 Embedding lookup，将混合特征转为统一向量
# ---------------------------------------------------------------------------

class NSFeatureEncoder(nn.Module):
    """
    将 data.py 输出的混合特征字典转为一个拼接向量：
      - continuous 特征：直接使用归一化后的 float 值
      - discrete_id / discrete_str 特征：nn.Embedding lookup
      - multihot 特征：nn.Embedding lookup + mean/sum/max pooling

    输出：(B, total_dim) 的拼接向量，total_dim 由 ns_feature_specs 决定
    """

    def __init__(self, ns_feature_specs: List[Dict]):
        super().__init__()
        self.specs = ns_feature_specs
        self.embeddings = nn.ModuleDict()

        for spec in ns_feature_specs:
            name  = spec["name"]
            ftype = spec["type"]
            if ftype in ("discrete_id", "discrete_str", "multihot"):
                self.embeddings[name] = nn.Embedding(
                    num_embeddings=spec["vocab_size"],
                    embedding_dim=spec["emb_dim"],
                    padding_idx=0,   # ID=0 保留为 padding，输出全零
                )

    @property
    def output_dim(self) -> int:
        total = 0
        for spec in self.specs:
            total += 1 if spec["type"] == "continuous" else spec["emb_dim"]
        return total

    def forward(self, ns_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            ns_inputs: data.py 输出的字典，key=特征名
                continuous   → (B,) float tensor
                discrete_id  → (B,) long tensor（整数 ID）
                discrete_str → (B,) long tensor（整数 ID）
                multihot     → (B, max_len) long tensor（ID 序列，0 为 padding）
        Returns:
            (B, output_dim) float tensor
        """
        parts = []
        for spec in self.specs:
            name  = spec["name"]
            ftype = spec["type"]
            val   = ns_inputs[name]

            if ftype == "continuous":
                parts.append(val.unsqueeze(1).float())          # (B, 1)

            elif ftype in ("discrete_id", "discrete_str"):
                emb = self.embeddings[name](val.long())         # (B, emb_dim)
                parts.append(emb)

            elif ftype == "multihot":
                emb = self.embeddings[name](val.long())         # (B, max_len, emb_dim)
                mask = (val != 0).float().unsqueeze(-1)         # (B, max_len, 1)，非 padding 位为 1
                pooling = spec.get("pooling", "mean")
                if pooling == "mean":
                    n = mask.sum(dim=1).clamp(min=1)            # (B, 1)
                    pooled = (emb * mask).sum(dim=1) / n        # (B, emb_dim)
                elif pooling == "sum":
                    pooled = (emb * mask).sum(dim=1)
                else:  # max
                    pooled = (emb * mask + (1 - mask) * -1e9).max(dim=1).values
                parts.append(pooled)

        return torch.cat(parts, dim=1)                          # (B, output_dim)


class GroupWiseNSTokenizer(nn.Module):
    """
    Group-wise Non-Sequential Tokenizer（对齐 RankMixer）
    将非序列特征手动分组，每组通过独立 MLP 映射为一个 token。
    """

    def __init__(self, group_dims: List[int], d_model: int):
        """
        Args:
            group_dims: 每个特征组的输入维度列表（各组维度可以不同）
            d_model: 输出 token 的维度
        """
        super().__init__()
        self.group_dims = group_dims   # 保留供 OneTrans.forward 按维度切分用
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            for dim in group_dims
        ])
        self.L_NS = len(group_dims)

    def forward(self, groups: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            groups: 每个特征组的 embedding，shape [(B, dim_i)]
        Returns:
            NS tokens, shape (B, L_NS, d_model)
        """
        tokens = [mlp(g) for mlp, g in zip(self.mlps, groups)]
        return torch.stack(tokens, dim=1)  # (B, L_NS, d)


class AutoSplitNSTokenizer(nn.Module):
    """
    Auto-Split Non-Sequential Tokenizer
    所有非序列特征拼接后通过单个 MLP 映射，再 split 成 L_NS 个 token。
    减少 kernel 启动开销，论文实验中默认使用此方式。
    """

    def __init__(self, total_ns_dim: int, d_model: int, L_NS: int):
        """
        Args:
            total_ns_dim: 所有非序列特征拼接后的总维度
            d_model: 输出 token 的维度
            L_NS: 非序列 token 数量
        """
        super().__init__()
        self.L_NS = L_NS
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(total_ns_dim, d_model * L_NS),
            nn.LayerNorm(d_model * L_NS),   # 防止输入方差过大时输出爆炸
            nn.ReLU(),
            nn.Linear(d_model * L_NS, d_model * L_NS),
        )

    def forward(self, ns_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ns_features: 拼接后的非序列特征，shape (B, total_ns_dim)
        Returns:
            NS tokens, shape (B, L_NS, d_model)
        """
        B = ns_features.size(0)
        out = self.mlp(ns_features)  # (B, d_model * L_NS)
        return out.view(B, self.L_NS, self.d_model)


class SequentialTokenizer(nn.Module):
    """
    Sequential Tokenizer
    处理多行为序列，支持 timestamp-aware 和 timestamp-agnostic 两种融合方式。
    每种行为序列有独立的 MLP 将 event embedding 映射到统一维度 d。
    """

    def __init__(
        self,
        seq_input_dims: List[int],
        d_model: int,
        use_sep_tokens: bool = True,
        timestamp_aware: bool = True,
    ):
        """
        Args:
            seq_input_dims: 每种行为序列的 event embedding 维度
            d_model: 统一输出维度
            use_sep_tokens: 是否在序列间插入可学习 [SEP] token
            timestamp_aware: True=时间戳感知融合, False=按意图排序拼接
        """
        super().__init__()
        self.n_seqs = len(seq_input_dims)
        self.d_model = d_model
        self.use_sep_tokens = use_sep_tokens
        self.timestamp_aware = timestamp_aware

        # 每种行为序列独立的投影：Linear + LayerNorm，防止序列特征方差过大
        self.seq_mlps = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, d_model), nn.LayerNorm(d_model))
            for dim in seq_input_dims
        ])

        # 可学习的 [SEP] token（每个序列边界一个）
        if use_sep_tokens and not timestamp_aware:
            self.sep_tokens = nn.Parameter(
                torch.randn(self.n_seqs - 1, d_model) * 0.02
            )

        # 序列类型 indicator embedding（timestamp-aware 时使用）
        if timestamp_aware:
            self.seq_type_emb = nn.Embedding(self.n_seqs, d_model)

    def forward(
        self,
        sequences: List[torch.Tensor],
        timestamps: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Args:
            sequences: 多行为序列列表，每个 shape (B, L_i, dim_i)
            timestamps: 每个序列的时间戳，shape (B, L_i)，timestamp-aware 时需要
        Returns:
            S-tokens: shape (B, L_S, d_model)
            L_S: 序列 token 总长度
        """
        B = sequences[0].size(0)

        # 将每种序列投影到统一维度
        projected = []
        for i, (seq, mlp) in enumerate(zip(sequences, self.seq_mlps)):
            proj = mlp(seq)  # (B, L_i, d)
            if self.timestamp_aware:
                type_emb = self.seq_type_emb(
                    torch.full((B, seq.size(1)), i, device=seq.device, dtype=torch.long)
                )
                proj = proj + type_emb
            projected.append(proj)

        if self.timestamp_aware and timestamps is not None:
            # 按时间戳交错融合所有事件
            s_tokens = self._timestamp_aware_merge(projected, timestamps, B)
        else:
            # 按意图排序拼接，并插入 [SEP] token
            s_tokens = self._timestamp_agnostic_merge(projected, B)

        return s_tokens, s_tokens.size(1)

    def _timestamp_aware_merge(
        self,
        projected: List[torch.Tensor],
        timestamps: List[torch.Tensor],
        B: int,
    ) -> torch.Tensor:
        """
        按时间戳将所有事件交错排列。

        时间戳为相对时间差（data.py 计算：(ref - t) / ref）：
          - 越早的事件值越大（接近 1）
          - 越近的事件值越小（接近 0）

        Causal 约定：序列从左到右表示从早到近，最近的事件在末尾。
        因此按相对时间差降序排列（大的在前 = 早的在前，小的在后 = 近的在后）。
        """
        all_tokens = torch.cat(projected, dim=1)  # (B, sum_L_i, d)
        all_ts = torch.cat(timestamps, dim=1)      # (B, sum_L_i)

        # 降序：值大（早）→ 前，值小（近）→ 后，符合 causal 约定
        sort_idx = torch.argsort(all_ts, dim=1, descending=True)
        all_tokens = torch.gather(
            all_tokens,
            1,
            sort_idx.unsqueeze(-1).expand_as(all_tokens),
        )
        return all_tokens

    def _timestamp_agnostic_merge(
        self,
        projected: List[torch.Tensor],
        B: int,
    ) -> torch.Tensor:
        """按意图顺序拼接，序列间插入 [SEP] token"""
        parts = []
        for i, proj in enumerate(projected):
            parts.append(proj)
            if self.use_sep_tokens and i < self.n_seqs - 1:
                sep = self.sep_tokens[i].unsqueeze(0).unsqueeze(0).expand(B, 1, -1)
                parts.append(sep)
        return torch.cat(parts, dim=1)  # (B, L_S, d)
