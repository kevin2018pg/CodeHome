"""
OneTrans 核心模型模块

实现论文 Section 3.3 ~ 3.5 中的：
- Mixed Causal Attention（S-tokens 共享参数，NS-tokens 独立参数）
- Mixed FFN
- OneTrans Block（Pre-Norm + RMSNorm）
- Pyramid Stack（渐进式 token 剪枝）
- Cross-Request KV Caching
- 完整 OneTrans 模型
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizer import AutoSplitNSTokenizer, GroupWiseNSTokenizer, NSFeatureEncoder, SequentialTokenizer


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先 cast 到 float32 计算，防止 fp16 下溢出
        x_f = x.float()
        rms = x_f.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * rms * self.weight).to(x.dtype)


# ---------------------------------------------------------------------------
# Mixed Multi-Head Attention
# ---------------------------------------------------------------------------

class MixedMHA(nn.Module):
    """
    Mixed 参数化的多头因果注意力。
    - S-tokens（索引 0..L_S-1）：共享一组 Q/K/V 投影权重
    - NS-tokens（索引 L_S..L_S+L_NS-1）：每个 token 独立的 Q/K/V 权重

    因果掩码：
    1. S-token 只能关注更早的 S-token（标准 causal）
    2. NS-token 可以关注所有 S-token 以及更早的 NS-token
    """

    def __init__(self, d_model: int, n_heads: int, L_NS: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads
        self.L_NS    = L_NS
        # 用 d_model 而非 d_head 做缩放，防止 d_head 小时 score 过大
        self.scale   = math.sqrt(self.d_head)

        # S-tokens 共享的 Q/K/V 投影（用较小初始化）
        self.W_Q_S = nn.Linear(d_model, d_model, bias=False)
        self.W_K_S = nn.Linear(d_model, d_model, bias=False)
        self.W_V_S = nn.Linear(d_model, d_model, bias=False)
        # 缩小初始化幅度，防止初始 attn score 过大
        for w in [self.W_Q_S, self.W_K_S, self.W_V_S]:
            nn.init.normal_(w.weight, std=0.02)

        # NS-tokens 独立 Q/K/V：合并为 (L_NS, d_model, d_model) 的权重张量
        self.W_Q_NS = nn.Parameter(torch.empty(L_NS, d_model, d_model))
        self.W_K_NS = nn.Parameter(torch.empty(L_NS, d_model, d_model))
        self.W_V_NS = nn.Parameter(torch.empty(L_NS, d_model, d_model))
        for w in [self.W_Q_NS, self.W_K_NS, self.W_V_NS]:
            nn.init.normal_(w, std=0.02)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        L_S: int,
        query_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L, d = x.shape
        x_S  = x[:, :L_S, :]   # (B, L_S, d)
        x_NS = x[:, L_S:, :]   # (B, L_NS, d)

        # ---- K/V（全序列） ----
        K_S  = self.W_K_S(x_S)
        V_S  = self.W_V_S(x_S)
        K_NS = torch.einsum("bnd,ndo->bno", x_NS, self.W_K_NS)
        V_NS = torch.einsum("bnd,ndo->bno", x_NS, self.W_V_NS)
        K_full = torch.cat([K_S, K_NS], dim=1)
        V_full = torch.cat([V_S, V_NS], dim=1)

        # ---- Q（仅 query 部分） ----
        if query_mask is not None:
            Q_S = self.W_Q_S(x_S[:, query_mask, :])
        else:
            Q_S = self.W_Q_S(x_S)
        Q_NS  = torch.einsum("bnd,ndo->bno", x_NS, self.W_Q_NS)
        Q     = torch.cat([Q_S, Q_NS], dim=1)

        q_S   = Q_S.size(1)
        q_len = Q.size(1)
        kv_len = K_full.size(1)

        # ---- Multi-Head split ----
        def split_heads(t):
            B_, L_, _ = t.shape
            return t.view(B_, L_, self.n_heads, self.d_head).transpose(1, 2)

        Q_h = split_heads(Q)
        K_h = split_heads(K_full)
        V_h = split_heads(V_full)

        attn_scores = torch.matmul(Q_h, K_h.transpose(-2, -1)) / self.scale

        # ---- 因果掩码（一次性用 triu 构建，逻辑简单不易出错）----
        causal_mask = self._build_causal_mask(q_S, q_len, kv_len, L_S, query_mask, x.device)
        attn_scores = attn_scores + causal_mask

        # ---- padding mask：屏蔽 kv 中 S-token 的 padding 位 ----
        if padding_mask is not None:
            # padding_mask: (B, L_S)，True=有效；False 的位置在 kv 维度上加 -inf
            pad_bias = torch.zeros(B, 1, 1, kv_len, device=x.device)
            pad_bias[:, :, :, :L_S] = (
                (~padding_mask).float().unsqueeze(1).unsqueeze(2) * -1e4
            )
            attn_scores = attn_scores + pad_bias

        # clamp 防止极端值导致 softmax NaN
        attn_scores = attn_scores.clamp(min=-1e4, max=1e4)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V_h)
        out = out.transpose(1, 2).contiguous().view(B, q_len, d)
        return self.out_proj(out)

    def _build_causal_mask(
        self,
        q_S: int,
        q_len: int,
        kv_len: int,
        L_S: int,
        query_mask: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """
        构建因果掩码，shape (1, 1, q_len, kv_len)，-inf 表示不可见。

        kv 布局：[S(0..L_S-1), NS(L_S..L_S+L_NS-1)]
        q  布局：[selected-S(0..q_S-1), NS(q_S..q_S+L_NS-1)]

        规则：
          - S-query[i] 可见 S-kv[0..orig_pos[i]]，不可见 NS-kv
          - NS-query[i] 可见所有 S-kv，可见 NS-kv[0..i]
        """
        NEG_INF = -1e4   # 用有限大负数而非 float('-inf')，避免 fp16 溢出
        mask = torch.full((q_len, kv_len), NEG_INF, device=device)

        # S-query 部分：每个 query 对应的原始位置
        if query_mask is not None:
            orig_pos = torch.where(query_mask)[0]   # (q_S,) 原始位置索引
        else:
            orig_pos = torch.arange(L_S, device=device)   # (L_S,)

        if q_S > 0 and L_S > 0:
            # kv_col (1, L_S) <= orig_pos (q_S, 1)  →  (q_S, L_S) bool
            kv_col  = torch.arange(L_S, device=device).unsqueeze(0)   # (1, L_S)
            q_pos   = orig_pos.unsqueeze(1)                            # (q_S, 1)
            visible = kv_col <= q_pos                                  # (q_S, L_S)
            mask[:q_S, :L_S] = torch.where(
                visible,
                torch.zeros(q_S, L_S, device=device),
                torch.full((q_S, L_S), NEG_INF, device=device),
            )
            # S-query 不可见 NS-kv，保持 NEG_INF（已初始化）

        # NS-query 部分：可见所有 S-kv + causal NS-kv
        if self.L_NS > 0:
            mask[q_S:, :L_S] = 0.0   # 所有 S-kv 可见
            # NS-kv causal：NS-query[i] 可见 NS-kv[0..i]
            ns_i   = torch.arange(self.L_NS, device=device).unsqueeze(1)  # (L_NS, 1)
            ns_j   = torch.arange(self.L_NS, device=device).unsqueeze(0)  # (1, L_NS)
            ns_vis = (ns_j <= ns_i)                                        # (L_NS, L_NS)
            mask[q_S:, L_S:] = torch.where(
                ns_vis,
                torch.zeros(self.L_NS, self.L_NS, device=device),
                torch.full((self.L_NS, self.L_NS), NEG_INF, device=device),
            )

        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, q_len, kv_len)


# ---------------------------------------------------------------------------
# Mixed FFN
# ---------------------------------------------------------------------------

class MixedFFN(nn.Module):
    """
    Mixed 参数化的 FFN。
    - S-tokens 共享一组 FFN 权重
    - NS-tokens 各自独立的 FFN 权重（用批量矩阵乘实现，避免 for 循环）
    """

    def __init__(self, d_model: int, L_NS: int, ffn_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        d_ff = d_model * ffn_ratio
        self.L_NS = L_NS
        self.d_model = d_model
        self.d_ff = d_ff

        # S-tokens 共享 FFN
        self.ffn_S = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

        # NS-tokens 独立 FFN：合并为批量权重，用 einsum 一次性计算所有 token
        self.W1_NS = nn.Parameter(torch.empty(L_NS, d_model, d_ff))
        self.b1_NS = nn.Parameter(torch.zeros(L_NS, d_ff))
        self.W2_NS = nn.Parameter(torch.empty(L_NS, d_ff, d_model))
        self.b2_NS = nn.Parameter(torch.zeros(L_NS, d_model))
        # 用 normal(std=0.02) 替代 xavier，防止初始输出方差过大
        nn.init.normal_(self.W1_NS, std=0.02)
        nn.init.normal_(self.W2_NS, std=0.02)

        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, L_S: int) -> torch.Tensor:
        """
        Args:
            x: shape (B, L_S + L_NS, d_model)，L_S 可为 0
        """
        if L_S > 0:
            out_S = self.ffn_S(x[:, :L_S, :])
        else:
            out_S = None

        # NS-tokens 批量 FFN：(B, L_NS, d) -> (B, L_NS, d_ff) -> (B, L_NS, d)
        x_NS = x[:, L_S:, :]  # (B, L_NS, d)
        h = torch.einsum("bnd,ndf->bnf", x_NS, self.W1_NS) + self.b1_NS  # (B, L_NS, d_ff)
        h = self.dropout(self.act(h))
        out_NS = torch.einsum("bnf,nfd->bnd", h, self.W2_NS) + self.b2_NS  # (B, L_NS, d)

        if out_S is not None:
            return torch.cat([out_S, out_NS], dim=1)
        return out_NS


# ---------------------------------------------------------------------------
# OneTrans Block
# ---------------------------------------------------------------------------

class OneTransBlock(nn.Module):
    """
    单个 OneTrans Block：Pre-Norm + Mixed Causal Attention + Mixed FFN
    对应论文 Eq.(4)(5)
    """

    def __init__(self, d_model: int, n_heads: int, L_NS: int, ffn_ratio: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = MixedMHA(d_model, n_heads, L_NS, dropout)
        self.ffn = MixedFFN(d_model, L_NS, ffn_ratio, dropout)
        self.L_NS = L_NS

    def forward(
        self,
        x: torch.Tensor,
        L_S: int,
        query_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L_S + L_NS, d)
            L_S: S-token 数量
            query_mask: Pyramid 剪枝布尔掩码，shape (L_S,)
            padding_mask: S-token 有效位，shape (B, L_S)，True=有效（P1-1）
        Returns:
            output: (B, q_S + L_NS, d)，q_S = query_mask.sum() 或 L_S
        """
        normed = self.norm1(x)
        attn_out = self.attn(normed, L_S, query_mask, padding_mask)

        # 残差连接：x 需要裁剪到与 attn_out 相同的 token 数
        if query_mask is not None:
            # query_mask 只对应 S-token 部分（前 L_S 个 token）
            x_S_pruned = x[:, :L_S, :][:, query_mask, :]
            x_NS = x[:, L_S:, :]
            x_res = torch.cat([x_S_pruned, x_NS], dim=1)
        else:
            x_res = x

        x_out = x_res + attn_out

        # FFN：L_S_for_ffn = 剪枝后的 S-token 数量
        q_S = query_mask.sum().item() if query_mask is not None else L_S
        x_out = x_out + self.ffn(self.norm2(x_out), int(q_S))

        return x_out


# ---------------------------------------------------------------------------
# Pyramid Schedule
# ---------------------------------------------------------------------------

def build_pyramid_schedule(L_S_init: int, L_NS: int, n_layers: int) -> List[int]:
    """
    构建 Pyramid Schedule：每层线性缩减 S-token query 数量。
    从 L_S_init 线性缩减到 L_NS，每层对齐到 32 的倍数。
    """
    schedule = []
    for layer in range(n_layers):
        ratio = 1.0 - layer / max(n_layers - 1, 1)
        L = L_NS + int((L_S_init - L_NS) * ratio)
        L = max(L_NS, (L // 32) * 32) if L >= 32 else max(L_NS, L)
        schedule.append(L)
    return schedule


# ---------------------------------------------------------------------------
# 完整 OneTrans 模型
# ---------------------------------------------------------------------------

class OneTrans(nn.Module):
    """
    完整 OneTrans 模型，对应论文 Figure 2(a)。

    流程：
    1. Tokenizer 将 NS/S 特征映射为 token 序列
    2. Pyramid 堆叠的 OneTrans Block 联合建模
    3. 最终 NS-tokens 送入多任务预测头
    """

    def __init__(
        self,
        ns_tokenizer_type: str = "auto_split",
        ns_feature_specs: Optional[List[Dict]] = None,
        total_ns_dim: int = 512,
        group_dims: Optional[List[int]] = None,
        L_NS: int = 16,
        seq_input_dims: List[int] = None,
        timestamp_aware: bool = True,
        use_sep_tokens: bool = True,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 6,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
        task_names: List[str] = None,
        L_S_init: int = 1190,
    ):
        super().__init__()

        if seq_input_dims is None:
            seq_input_dims = [64]
        if task_names is None:
            task_names = ["ctr", "cvr"]

        self.d_model = d_model
        self.L_NS = L_NS
        self.n_layers = n_layers
        self.task_names = task_names

        # ---- NS 特征编码器（Embedding lookup，P0-1）----
        # 若提供 ns_feature_specs，则用 NSFeatureEncoder 处理混合特征
        # 否则退回到纯数值模式（兼容 mock 数据）
        if ns_feature_specs:
            self.ns_encoder = NSFeatureEncoder(ns_feature_specs)
            actual_ns_dim = self.ns_encoder.output_dim
        else:
            self.ns_encoder = None
            actual_ns_dim = total_ns_dim

        # ---- Tokenizers ----
        if ns_tokenizer_type == "auto_split":
            self.ns_tokenizer = AutoSplitNSTokenizer(actual_ns_dim, d_model, L_NS)
        else:
            assert group_dims is not None
            self.ns_tokenizer = GroupWiseNSTokenizer(group_dims, d_model)

        self.seq_tokenizer = SequentialTokenizer(
            seq_input_dims, d_model, use_sep_tokens, timestamp_aware
        )

        # ---- Pyramid Schedule ----
        self.pyramid_schedule = build_pyramid_schedule(L_S_init, L_NS, n_layers)

        # ---- OneTrans Blocks ----
        self.blocks = nn.ModuleList([
            OneTransBlock(d_model, n_heads, L_NS, ffn_ratio, dropout)
            for _ in range(n_layers)
        ])

        self.final_norm = RMSNorm(d_model)

        # ---- 多任务预测头 ----
        ns_flat_dim = L_NS * d_model
        self.task_heads = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(ns_flat_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1),
            )
            for name in task_names
        })

        # ---- 位置偏置校准层（Calibration）----
        # 每个展示位置学一个加性偏置，用于修正位置效应和分布漂移。
        # num_positions=20 表示支持最多 20 个展示坑位（0=未知/默认）。
        # 参数量极小（task数 × 20），可每小时独立微调，不影响主干。
        self.num_positions = 20
        self.position_bias = nn.ModuleDict({
            name: nn.Embedding(self.num_positions, 1)
            for name in task_names
        })
        # 初始化为全零：训练初期不引入位置偏置，让主干先收敛
        for emb in self.position_bias.values():
            nn.init.zeros_(emb.weight)

    def forward(
        self,
        ns_inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        sequences: List[torch.Tensor],
        timestamps: Optional[List[torch.Tensor]] = None,
        seq_masks: Optional[List[torch.Tensor]] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], None]:
        """
        Args:
            ns_inputs:    非序列特征，Dict[str, Tensor] 或 Tensor (B, total_ns_dim)
            sequences:    多行为序列列表，每个 (B, L_i, dim_i)
            timestamps:   时间戳列表，每个 (B, L_i)，相对时间差已归一化
            seq_masks:    序列 padding mask 列表，每个 (B, L_i)，True=有效
            position_ids: 展示位置 ID，shape (B,)，整数，范围 [0, num_positions)
                          0 = 未知/默认，不传则不做位置校准
        Returns:
            predictions: 各任务的预测分数字典
            None: 保留接口（KV Cache 预留）
        """
        B = sequences[0].size(0)

        # ---- NS 特征编码 ----
        if isinstance(ns_inputs, dict):
            if self.ns_encoder is not None:
                ns_vec = self.ns_encoder(ns_inputs)
            else:
                ns_vec = torch.stack(list(ns_inputs.values()), dim=1).float()
        else:
            ns_vec = ns_inputs

        # ---- Tokenization ----
        if isinstance(self.ns_tokenizer, GroupWiseNSTokenizer):
            ns_groups = list(ns_vec.split(self.ns_tokenizer.group_dims, dim=-1))
            ns_tokens = self.ns_tokenizer(ns_groups)
        else:
            ns_tokens = self.ns_tokenizer(ns_vec)

        s_tokens, L_S = self.seq_tokenizer(sequences, timestamps)
        x = torch.cat([s_tokens, ns_tokens], dim=1)

        # ---- padding mask ----
        if seq_masks is not None:
            combined_mask = torch.cat(seq_masks, dim=1)
            padding_mask = combined_mask[:, :L_S]
        else:
            padding_mask = None

        # ---- Pyramid Stacked OneTrans Blocks ----
        current_L_S = L_S
        for layer_idx, block in enumerate(self.blocks):
            target_q_len = self.pyramid_schedule[layer_idx]
            if target_q_len < current_L_S:
                query_mask = torch.zeros(current_L_S, dtype=torch.bool, device=x.device)
                query_mask[current_L_S - target_q_len:] = True
            else:
                query_mask = None
            x = block(x, current_L_S, query_mask, padding_mask)
            if query_mask is not None:
                current_L_S = target_q_len
                if padding_mask is not None:
                    padding_mask = padding_mask[:, -current_L_S:]

        # ---- 最终只取 NS-tokens 做预测 ----
        ns_out = x[:, current_L_S:, :]
        ns_out = self.final_norm(ns_out)
        ns_flat = ns_out.view(B, -1)

        # ---- Task Head + Calibration ----
        # position_ids clamp 到合法范围，防止越界
        if position_ids is not None:
            pos = position_ids.clamp(0, self.num_positions - 1)

        predictions = {}
        for name, head in self.task_heads.items():
            logit = head(ns_flat).squeeze(-1)                   # (B,)
            if position_ids is not None:
                bias = self.position_bias[name](pos).squeeze(-1)  # (B,)
                logit = logit + bias
            # 返回 logit（未经 sigmoid），由 loss 函数用 BCEWithLogitsLoss 处理
            # 推理时在外部调用 torch.sigmoid(logit) 得到概率
            predictions[name] = logit

        return predictions, None

    def forward_with_kv_cache(
        self,
        ns_inputs: Union[torch.Tensor, Dict[str, torch.Tensor]],
        sequences: List[torch.Tensor],
        timestamps: Optional[List[torch.Tensor]] = None,
        seq_masks: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], None]:
        """Stage I 推理接口（KV Cache 演示用）"""
        return self.forward(ns_inputs, sequences, timestamps, seq_masks)
