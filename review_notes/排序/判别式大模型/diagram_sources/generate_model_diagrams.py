"""生成判别式推荐模型的中文“一图读懂”PNG。

依赖：pip install pillow
运行：python generate_model_diagrams.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 2400, 1600
MARGIN = 72
FONT_REGULAR = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"

BG = "#F7F9FC"
TEXT = "#172033"
MUTED = "#5D687A"
LINE = "#CAD2E1"
WHITE = "#FFFFFF"

ROUTE_COLORS = {
    "A": ("#EAF2FF", "#2F6BDE", "#C6D9FF"),
    "B": ("#EAF8F1", "#16845B", "#BFE8D5"),
    "C": ("#F4EEFF", "#7047B8", "#DCCCF7"),
}


@dataclass(frozen=True)
class Poster:
    filename: str
    route: str
    title: str
    subtitle: str
    paper: str
    attention: str
    tokens: tuple[str, ...]
    flow: tuple[str, ...]
    formulas: tuple[str, ...]
    engineering: tuple[str, ...]
    evidence: tuple[str, ...]
    boundary: tuple[str, ...]
    takeaway: str


POSTERS = (
    Poster(
        filename="04-wukong-architecture.png",
        route="A",
        title="Wukong：显式交叉阶数 Scaling",
        subtitle="把 FM 变成可堆叠交互层，使最高交叉阶数随深度按 2^l 增长",
        paper="ICML 2024 · arXiv:2403.02545",
        attention="无 Attention",
        tokens=(
            "稀疏特征：embedding lookup / multi-hot pooling",
            "重要 field 可分配多个 embedding；弱 field 可低维分组后投影",
            "连续特征：MLP 投影；统一得到 X0:[B,N,D]",
        ),
        flow=(
            "输入 Token\nX_i:[N,D]",
            "FMB\nFM 二阶交互\n→ MLP",
            "LCB\nToken 维线性压缩\nW_L X_i",
            "Concat\n+ ResidualProject",
            "LayerNorm\nX_(i+1)",
            "Flatten / Pool\n→ Task Tower",
        ),
        formulas=(
            "Basic FM(X)=X X^T，保留 N×N 交互矩阵",
            "Optimized FM=(X X^T)Y=X(X^T Y)",
            "复杂度：O(N²D) → O(NKD)，K≪N",
            "第 l 层覆盖 1…2^l 阶；LCB 不提升交叉阶数",
        ),
        engineering=(
            "FMB 负责升阶，LCB/残差保留低阶与信息通路",
            "Interaction Stack 可通过层数、宽度和压缩 token 数扩展",
            "比一次性 FM 更重，但比任意手工高阶组合更系统",
        ),
        evidence=(
            "论文在 6 个公开数据集及内部大规模数据验证 Scaling",
            "内部实验覆盖超过 100 GFLOP/example 的复杂度区间",
            "公开证据主要是离线质量与 Scaling，不等同于低延迟 serving",
        ),
        boundary=(
            "FMB 交互矩阵和 MLP 仍可能成为计算瓶颈",
            "不原生处理行为时序；通常需要外接序列模块",
            "高阶可达不代表每个高阶组合都可解释或有效",
        ),
        takeaway="普通 FM 把二阶交叉压成分数；Wukong 把交叉重新编码成 token，再交给下一层继续升阶。",
    ),
    Poster(
        filename="05-rankmixer-token-mixing.png",
        route="A",
        title="RankMixer：硬件友好的 Ranking Backbone",
        subtitle="用零参数 Token Mixing 做全局通信，用 Per-token FFN 扩大异构特征容量",
        paper="ByteDance · arXiv:2507.15551",
        attention="无 Attention",
        tokens=(
            "user / item / context / sequence summary 按语义分组",
            "组内 concat → 独立 Projector → token_g:[B,D]",
            "stack 后 X0:[B,T,D]；原始长序列通常先由外部模块聚合",
        ),
        flow=(
            "语义 Token\nX:[B,T,D]",
            "Split Channel\n[B,T,H,D/H]",
            "Transpose T,H\n[B,H,T,D/H]",
            "Flatten\n[B,H,T·D/H]",
            "Per-token FFN\n或 ReLU MoE",
            "Mean Pool\n→ Task Towers",
        ),
        formulas=(
            "原版设置 H=T，因此 Mixing 输出仍为 [B,T,D]",
            "PFFN_t(x)=W2_t·GELU(W1_t x+b1_t)+b2_t",
            "不同 token 不共享 W1_t/W2_t；只做 token 内 channel 建模",
            "PFFN 参数约 2·T·r·D²，主要计算可落为大 GEMM",
        ),
        engineering=(
            "Mixing 只有 reshape/transpose/concat，无 QKV 与 T² 权重矩阵",
            "ReLU Routing + Dense-Training/Sparse-Inference",
            "训练仍计算全部专家；推理按激活稀疏执行",
        ),
        evidence=(
            "100M：D=768,T=16,L=2；1B：D=1536,T=32,L=2",
            "参数约 70×，MFU 4.47%→44.57%，延迟 14.5ms→14.3ms",
            "移除 Mixing：AUC -0.50%；PFFN 改共享 FFN：-0.31%",
        ),
        boundary=(
            "固定重排不理解候选、时间或内容相似度",
            "Mix 后位置语义变化，却与原 token 直接残差相加",
            "H=T、Post-Norm 与相邻层残差限制深层扩展",
        ),
        takeaway="RankMixer 用“重排 + 独立大 MLP”替代 Attention：通信便宜、参数可扩，但原版深层残差并不理想。",
    ),
    Poster(
        filename="06-tokenmixer-large-forward.png",
        route="A",
        title="TokenMixer-Large：Mixing & Reverting",
        subtitle="修复 RankMixer 的残差布局与深层训练问题，并升级为真正训练/推理双稀疏",
        paper="ByteDance · arXiv:2602.06563",
        attention="无 Attention",
        tokens=(
            "语义分组 token + 1 个 Global Token，输入 X:[B,T,D]",
            "组内由 MLP 映射；Global Token 读取所有原始特征后投影",
            "H 与 T 解耦：中间可为 [B,H,T·D/H]，Block 输出恢复 [B,T,D]",
        ),
        flow=(
            "Pre-RMSNorm\nX:[T,D]",
            "Mixing\n[T,D]→[H,T·D/H]",
            "Per-token\nSwiGLU / S-P MoE",
            "Reverting\n→ ΔX:[T,D]",
            "原 Token 空间\nSwiGLU / MoE",
            "Aligned Residual\n+ Inter-Residual",
        ),
        formulas=(
            "Revert(Mix(X))=X；但 Revert(FFN(Mix(X)))=ΔX≠X",
            "残差：X_next = X + ΔX，恢复的是槽位布局而非原值",
            "SwiGLU(x)=Wdown[SiLU(Wgate x) ⊙ (Wup x)]",
            "Pre-RMSNorm 保留恒等梯度；每 2–3 层再加跨层残差",
        ),
        engineering=(
            "GroupedGEMM、Token Parallel、FP8 serving",
            "固定 Top-k Sparse Per-token MoE + per-token Shared Expert",
            "Wdown 小初始化，使 Block 初期接近恒等映射",
        ),
        evidence=(
            "论文报告线上 7B、离线 15B 参数规模",
            "移除 Mixing & Reverting：ΔAUC -0.27%（论文消融）",
            "线上报告：电商 Order +1.66%、人均预览支付 GMV +2.98%",
        ),
        boundary=(
            "Mix/Revert 自身不学习语义，只恢复坐标约定",
            "双 SwiGLU/MoE 比浅层 RankMixer 更重，需要足量数据",
            "原文公式仍像 Post-Norm，与正文 Pre-RMSNorm 描述存在冲突",
        ),
        takeaway="Revert 不是“把值变回去”，而是把混合空间产生的更新量重新放回原 token 槽位，建立可深堆叠的 Block 契约。",
    ),
    Poster(
        filename="08-hstu-block.png",
        route="B",
        title="HSTU：推荐原生序列 Backbone",
        subtitle="用 Pointwise Aggregated Attention 与行为语义建模替代通用 Transformer 配方",
        paper="Meta · arXiv:2402.17152",
        attention="Causal Self-Attention",
        tokens=(
            "按时间排列内容 Φ_i 与动作 a_i，可形成交错行为流",
            "内容包含 item/属性；动作包含 click、skip、购买等反馈",
            "既可做判别式序列表示，也可做 next-item 生成式训练",
        ),
        flow=(
            "行为流\nX:[B,L,D]",
            "Fused Projection\n+ SiLU",
            "Split\nU,V,Q,K",
            "SiLU(QK^T+bias)\n× V",
            "LayerNorm\n⊙ U",
            "Output Projection\n+ Residual",
        ),
        formulas=(
            "U,V,Q,K = Split(SiLU(XW1+b1))",
            "A = SiLU(QK^T + relative/time bias)，不做行 Softmax",
            "Y = X + W2[LayerNorm(AV) ⊙ U]",
            "不归一化权重和，可保留行为数量/兴趣强度信号",
        ),
        engineering=(
            "Jagged tensor、fused kernel、长度采样降低 padding 与访存",
            "因果 Mask 支持自回归训练；缓存可复用历史表示",
            "理论注意力仍为 O(L²D)，超长依赖系统优化和采样",
        ),
        evidence=(
            "8192 长度：encoder 训练最高 15.2×、推理最高 5.6× 加速",
            "HSTU-large 公开指标相对 SASRec 最高 +65.8%",
            "1.5T 与线上 +12.4% 属于完整 Generative Recommender 系统",
        ),
        boundary=(
            "SiLU Attention 不是概率分布，数值尺度需配套归一化",
            "本身不解决大量异构 NS field 的高阶交叉",
            "短序列、小数据场景未必优于 DIN/BST 的成本收益",
        ),
        takeaway="HSTU 仍是 Self-Attention，但不把历史权重强制归一成 1，并把 action/时间偏置作为推荐序列的一等信息。",
    ),
    Poster(
        filename="09-longer-hybrid-attention.png",
        route="B",
        title="LONGER：端到端长序列建模",
        subtitle="先在局部组内建模并合并，再用 Hybrid Attention 读取压缩后的长历史",
        paper="ByteDance · 2025",
        attention="Cross + Causal Self",
        tokens=(
            "原始行为 S:[B,L,D]，相邻 K 个行为构成一个局部组",
            "Global Token + 近期短序列 token 作为固定长度 Query",
            "输出序列表示通常继续进入 RankMixer/精排交叉 Backbone",
        ),
        flow=(
            "Raw History\n[B,L,D]",
            "K-token Group\nInnerTrans",
            "Token Merge\n[B,L/K,D′]",
            "Layer 1 Cross\nQ←Merged K/V",
            "Later Layers\nCausal Self-Attn",
            "Sequence Rep\n→ NS Ranking",
        ),
        formulas=(
            "组内：InnerTrans([x_i…x_(i+K-1)]) → concat/project",
            "Query:[B,M+R,D]；K/V:[B,M+L/K,D]",
            "首层复杂度约 O((M+R)(L/K)D)，替代 O(L²D)",
            "InnerTrans 的作用是避免简单 concat 丢失组内次序关系",
        ),
        engineering=(
            "Mixed Precision、Activation Recompute、KV Cache",
            "稀疏 embedding 与稠密长序列模块协同训练/服务",
            "Token Merge 用长度压缩换取端到端覆盖更多历史",
        ),
        evidence=(
            "已在字节 10 多个工业场景部署并服务十亿级用户",
            "TokenMerge：FLOPs 3.73e9→3.03e9；加入 InnerTrans 后 AUC +1.63%",
            "核心收益来自更长细粒度历史，而非只检索少量行为",
        ),
        boundary=(
            "合并后不可避免损失部分跨组细节",
            "Hybrid Attention 后仍是分离式“先序列、后交叉”",
            "不适合百万级生命周期历史，后者通常需检索/聚类",
        ),
        takeaway="LONGER 不是把全部 L 个行为做全局 Self-Attention，而是“局部保真压缩 + 固定 Query 读取长历史”。",
    ),
    Poster(
        filename="10-stca-target-cross.png",
        route="B",
        title="STCA：Stacked Target-to-History Cross Attention",
        subtitle="只让候选反复读取固定历史，删除昂贵的 history-to-history Self-Attention",
        paper="ByteDance · 2025",
        attention="Target / Cross-Attention",
        tokens=(
            "候选 query q0:[B,1,D]；历史 H:[B,L,D]",
            "同一请求多个候选共享历史 K/V",
            "历史 token 在层间保持不变，只有 target state 逐层更新",
        ),
        flow=(
            "Target q0\n[1,D]",
            "CrossAttn\nq1 ← H1\n得到 o1",
            "Fuse\nTarget + o1\n→ q2",
            "再次 CrossAttn\nq2 ← H2",
            "Fuse 所有 Summary\n得到 z",
            "Task Tower\npCTR / pCVR",
        ),
        formulas=(
            "o_i=CrossAttn(q_i, LN(SwiGLU_i(H)))",
            "q_(i+1)=SwiGLU(concat(target,o_1…o_i)Wc_i)",
            "注意力矩阵从 Self-Attn 的 [L,L] 变为 [1,L]",
            "结合律重排避免物化长 K/V；长度主项 O(Ldh)",
        ),
        engineering=(
            "RLB：候选维批量化，共享历史；带宽下降 77%～84%",
            "平均约 2K 长度训练，最长 10K 推理",
            "RLB 吞吐 2.2×；叠加专用 Kernel 后 5.1×",
        ),
        evidence=(
            "STCA+RLB+Ext 在三项 Douyin 离线任务均取得最佳整体结果",
            "最大可训练序列长度约提升 8×，已部署全量流量",
            "系统收益来自 STCA、RLB、外推训练与 Kernel 的组合",
        ),
        boundary=(
            "history token 不互相更新，无法直接建模行为组合/演化",
            "每层历史投影不同，不能默认层间 K/V 共享",
            "强依赖 history-history 关系时应选 HSTU/LONGER",
        ),
        takeaway="STCA 用表达力换复杂度：保留排序最关键的 target↔history，主动删除 history↔history。",
    ),
    Poster(
        filename="13-onetrans-mixed-block.png",
        route="C",
        title="OneTrans：S / NS 单流统一 Transformer",
        subtitle="统一 Tokenizer + Mixed Parameterization + Pyramid，把序列与非序列交互放入同一 Backbone",
        paper="ByteDance · WWW 2026",
        attention="Causal Self-Attention",
        tokens=(
            "S：每个行为事件 concat ID/属性/action/time → 共享序列映射",
            "NS：按 field group concat → token-specific MLP → ns token",
            "排列 [S1…SL, NS1…NSN]，X0:[B,L+N,D]",
        ),
        flow=(
            "Unified Tokens\nS…S | NS…NS",
            "Pre-RMSNorm\nMixed MHA",
            "S 共享参数\nNS 独立参数",
            "Mixed FFN\n+ Residual",
            "Pyramid\n逐层缩短 Query",
            "KV Cache\n→ Task Towers",
        ),
        formulas=(
            "Z_l=X_l+MixedMHA(RMSNorm(X_l))",
            "X_(l+1)=Z_l+MixedFFN(RMSNorm(Z_l))",
            "S 共享 Wq/Wk/Wv/FFN；第 i 个 NS 使用独立参数",
            "Pyramid 只裁剪尾部 Query；K/V 仍读取可见上下文",
        ),
        engineering=(
            "Mixed Parameterization 同时适配同质行为与异构 field",
            "Pyramid 降低上层 Query 数；KV Cache 复用用户 S-side",
            "多候选可共享历史 K/V，但 candidate-specific 特征需谨慎布局",
        ),
        evidence=(
            "OneTransL：CTR AUC/UAUC +1.53%/+2.79%",
            "Feeds：click/order/GMV per user +7.737%/+4.351%/+5.685%",
            "Feeds p99 延迟 -3.91%；Pyramid 是位置截断而非内容 Top-k",
        ),
        boundary=(
            "因果排列使 S 不能读取后置 NS，交互方向非完全对称",
            "NS token-specific 参数随 field/token 数增长",
            "尾部 Query 偏置可能弱化较早行为的高层更新",
        ),
        takeaway="OneTrans 不是把 pooled sequence 塞进 RankMixer，而是让行为 token 与 field token 从底层开始共享同一注意力上下文。",
    ),
    Poster(
        filename="14-hyformer-query-boosting.png",
        route="C",
        title="HyFormer：Query Decoding + Query Boosting",
        subtitle="用少量 Global Query 作为瓶颈，在每层交替吸收序列证据和 NS 交叉信息",
        paper="ByteDance · 2025/2026",
        attention="Cross-Attention + Mixer",
        tokens=(
            "NS 语义组生成 F1…FM；序列侧产生 H_l:[B,L,D]",
            "concat(F1…FM, MeanPool(S)) → N 个初始化 Global Query",
            "Query 数 N 固定且远小于序列长度 L",
        ),
        flow=(
            "Sequence Layer\n得到 H_l",
            "Query Decoding\nQ←CrossAttn(Q,H_l)",
            "Concat\nQ_hat + NS Tokens",
            "Query Boosting\nToken Mixing + PFFN",
            "Enhanced Q_l\n进入下一层",
            "Query Pool/Fusion\n→ Towers",
        ),
        formulas=(
            "Q_hat_l=CrossAttn(Q_(l-1), K=H_lWk, V=H_lWv)",
            "Q_all=concat(Q_hat_l,F1…FM) → Mixer/PFFN → Q_l",
            "Cross 部分复杂度约 O(NLD)，N≪L",
            "论文主链路更新 Query；没有明确把增强 Query 写回 H_l",
        ),
        engineering=(
            "Query Bottleneck 避免让全部 S/NS token 做平方 Self-Attn",
            "Query Boosting 可复用 RankMixer 式硬件友好算子",
            "Sequence Encoder 与 Query 交叉可按层流水执行",
        ),
        evidence=(
            "抖音搜索：70 天日志、约 30 亿样本、64 GPU 训练",
            "线上：观看时长 +0.293%、完播数 +1.111%、Query 改写率 -0.236%",
            "13 个 NS + 3 个 Query 是论文场景配置，不是通用常量",
        ),
        boundary=(
            "Query 是信息瓶颈，N 太小可能丢失细粒度兴趣",
            "Boost 后 T→N 的 query 选择接口在论文公式中被省略",
            "与 Hiformer 名称相似，但后者是 Google 异构特征交叉模型",
        ),
        takeaway="HyFormer 的统一发生在 Query 上：Query 先读序列，再与 NS 做 Mixer 交叉，然后带着新语义继续读下一层序列。",
    ),
    Poster(
        filename="16-tokenformer-bfts-nlir.png",
        route="C",
        title="TokenFormer：缓解 Sequential Collapse Propagation",
        subtitle="用 BFTS Attention 控制深层感受野，用 NLIR 门控增强非线性交互与表示区分度",
        paper="Tencent · 2026",
        attention="Full + Sliding Causal Self",
        tokens=(
            "统一排列：[Field Tokens, SEP, Sequence Tokens, SEP, Target Tokens]",
            "静态 field 位置可设为 0；行为按时间编码；target 放在行为之后",
            "SCP 是论文观察到的故障模式，不是统一模型必然定理",
        ),
        flow=(
            "Unified Tokens\nF | SEP | S | SEP | V",
            "Bottom Layers\nFull Causal Attn",
            "建立全局\nS/NS 交互",
            "Top Layers\nSliding Window",
            "窗口逐层缩小\n抑制污染/降算",
            "NLIR Gate\n→ Target Towers",
        ),
        formulas=(
            "BFTS：底层 Full，顶层 window_1>window_2>…",
            "Mask_l[i,j]=0 当 j≤i 且 i-j<window_l，否则 -∞",
            "I_l=X_l+sigmoid(X_lWg)⊙Attention(RMSNorm(X_l))",
            "再接 SwiGLU 式 FFN 与残差，形成 NLIR",
        ),
        engineering=(
            "上层缩窗减少注意力计算并限制 NS 污染继续传播",
            "进入顶层后可丢弃已充分注入的部分前置 NS token",
            "乘性 gate 让每个位置动态控制 Attention 更新强度",
            "用 MI、奇异值谱与 effective rank 诊断表示坍缩",
        ),
        evidence=(
            "TokenFormer-L：Macro AUC 0.86282，相对 Transformer +8.15‰",
            "2F2S 比 4F 高 0.85‰，GFLOPs 下降约 20.1%",
            "微信视频号 GMV +4.03%；解耦 serving 吞吐 126→695 QPS",
        ),
        boundary=(
            "底部 Full Attention 仍包含平方成本",
            "顶部窗口过小可能损失长期依赖",
            "解耦 serving 未充分证明与完整 unified forward 数值等价",
        ),
        takeaway="TokenFormer 不只是追求“统一”，而是关注统一后深层表示会不会被低秩异构 token 持续污染。",
    ),
)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size)


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> float:
    return draw.textlength(text, font=fnt)


def wrap(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in text.splitlines() or [""]:
        current = ""
        # 英文/公式作为整体 token，中文按字切分，避免把 Expert、Transformer 等拆到两行。
        tokens = re.findall(r"[A-Za-z0-9_./+·×²≪→←↔\-]+|\s+|.", paragraph)
        for token in tokens:
            candidate = current + token
            if current and text_width(draw, candidate, fnt) > max_width:
                lines.append(current.rstrip())
                current = token.lstrip()
            else:
                current = candidate
        lines.append(current.rstrip())
    return lines


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str,
    max_width: int,
    spacing: int = 9,
    max_lines: int | None = None,
) -> int:
    lines = wrap(draw, text, fnt, max_width)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1][:-1] + "…"
    x, y = xy
    line_height = fnt.size + spacing
    for line in lines:
        draw.text((x, y), line, font=fnt, fill=fill)
        y += line_height
    return y


def rounded_box(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    fill: str = WHITE,
    outline: str = LINE,
    width: int = 2,
    radius: int = 22,
) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str,
    width: int = 5,
) -> None:
    draw.line((start, end), fill=color, width=width)
    ex, ey = end
    sx, sy = start
    if abs(ex - sx) >= abs(ey - sy):
        direction = 1 if ex > sx else -1
        points = [(ex, ey), (ex - 18 * direction, ey - 11), (ex - 18 * direction, ey + 11)]
    else:
        direction = 1 if ey > sy else -1
        points = [(ex, ey), (ex - 11, ey - 18 * direction), (ex + 11, ey - 18 * direction)]
    draw.polygon(points, fill=color)


def draw_tag(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    bg: str,
    fg: str,
) -> int:
    fnt = font(24, bold=True)
    tag_width = int(text_width(draw, text, fnt)) + 34
    draw.rounded_rectangle((x, y, x + tag_width, y + 46), radius=20, fill=bg)
    draw.text((x + 17, y + 7), text, font=fnt, fill=fg)
    return x + tag_width


def draw_bullets(
    draw: ImageDraw.ImageDraw,
    items: Iterable[str],
    x: int,
    y: int,
    width: int,
    fnt: ImageFont.FreeTypeFont,
    color: str = TEXT,
    bullet_color: str = "#2F6BDE",
    line_spacing: int = 8,
    item_gap: int = 12,
) -> int:
    for item in items:
        draw.ellipse((x, y + 11, x + 9, y + 20), fill=bullet_color)
        y = draw_wrapped(draw, (x + 22, y), item, fnt, color, width - 22, line_spacing) + item_gap
    return y


def draw_card(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    items: tuple[str, ...],
    accent: str,
    body_size: int = 24,
) -> None:
    rounded_box(draw, box)
    x1, y1, x2, _ = box
    draw.rounded_rectangle((x1, y1, x2, y1 + 62), radius=22, fill="#F0F4FA")
    draw.rectangle((x1, y1 + 38, x2, y1 + 62), fill="#F0F4FA")
    draw.rectangle((x1, y1, x1 + 8, y1 + 62), fill=accent)
    draw.text((x1 + 24, y1 + 13), title, font=font(28, bold=True), fill=TEXT)
    draw_bullets(
        draw,
        items,
        x1 + 24,
        y1 + 82,
        x2 - x1 - 48,
        font(body_size),
        bullet_color=accent,
        line_spacing=7,
        item_gap=10,
    )


def render(poster: Poster, output_dir: Path) -> Path:
    route_bg, accent, route_border = ROUTE_COLORS[poster.route]
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    # Header
    draw.rounded_rectangle((MARGIN, 48, WIDTH - MARGIN, 236), radius=30, fill=WHITE, outline=LINE, width=2)
    draw.text((MARGIN + 38, 70), poster.title, font=font(54, bold=True), fill=TEXT)
    draw_wrapped(
        draw,
        (MARGIN + 40, 142),
        poster.subtitle,
        font(29),
        MUTED,
        1600,
        spacing=7,
        max_lines=2,
    )
    tag_x = WIDTH - MARGIN - 590
    tag_x = draw_tag(draw, tag_x, 80, f"路线 {poster.route}", route_bg, accent) + 14
    draw_tag(draw, tag_x, 80, poster.attention, "#EEF1F6", "#38445A")
    draw.text((WIDTH - MARGIN - 590, 154), poster.paper, font=font(24), fill=MUTED)

    # Tokenization card
    token_box = (MARGIN, 270, 660, 680)
    draw_card(draw, token_box, "1  输入与 Token 化", poster.tokens, accent, body_size=25)

    # Main flow
    flow_box = (696, 270, WIDTH - MARGIN, 680)
    rounded_box(draw, flow_box, fill=WHITE, outline=route_border, width=3)
    draw.text((726, 292), "2  整体前向链路", font=font(30, bold=True), fill=TEXT)
    flow_x1, flow_y1, flow_x2, _ = flow_box
    inner_x1, inner_x2 = flow_x1 + 30, flow_x2 - 30
    node_gap = 20
    node_count = len(poster.flow)
    node_width = int((inner_x2 - inner_x1 - node_gap * (node_count - 1)) / node_count)
    node_top, node_bottom = 390, 592
    centers: list[tuple[int, int]] = []
    for idx, label in enumerate(poster.flow):
        x1 = inner_x1 + idx * (node_width + node_gap)
        x2 = x1 + node_width
        node_fill = route_bg if idx in (0, node_count - 1) else "#F8FAFD"
        rounded_box(draw, (x1, node_top, x2, node_bottom), fill=node_fill, outline=route_border, width=2)
        lines = wrap(draw, label, font(23, bold=True), node_width - 24)
        total_height = len(lines) * 34
        y = node_top + (node_bottom - node_top - total_height) // 2
        for line in lines:
            line_w = text_width(draw, line, font(23, bold=True))
            draw.text((x1 + (node_width - line_w) / 2, y), line, font=font(23, bold=True), fill=TEXT)
            y += 34
        centers.append(((x1 + x2) // 2, (node_top + node_bottom) // 2))
        if idx > 0:
            prev_x = inner_x1 + (idx - 1) * (node_width + node_gap) + node_width
            arrow(draw, (prev_x + 4, (node_top + node_bottom) // 2), (x1 - 4, (node_top + node_bottom) // 2), accent, 4)

    # Middle cards
    card_y1, card_y2 = 716, 1160
    gap = 24
    card_width = (WIDTH - 2 * MARGIN - 2 * gap) // 3
    draw_card(
        draw,
        (MARGIN, card_y1, MARGIN + card_width, card_y2),
        "3  核心公式与操作",
        poster.formulas,
        accent,
        body_size=23,
    )
    draw_card(
        draw,
        (MARGIN + card_width + gap, card_y1, MARGIN + 2 * card_width + gap, card_y2),
        "4  复杂度与工程设计",
        poster.engineering,
        accent,
        body_size=23,
    )
    draw_card(
        draw,
        (MARGIN + 2 * (card_width + gap), card_y1, WIDTH - MARGIN, card_y2),
        "5  论文证据与定位",
        poster.evidence,
        accent,
        body_size=23,
    )

    # Boundary + takeaway
    bottom_y1, bottom_y2 = 1194, HEIGHT - 54
    boundary_width = 930
    draw_card(
        draw,
        (MARGIN, bottom_y1, MARGIN + boundary_width, bottom_y2),
        "6  使用边界",
        poster.boundary,
        "#C45A47",
        body_size=22,
    )
    takeaway_box = (MARGIN + boundary_width + 26, bottom_y1, WIDTH - MARGIN, bottom_y2)
    rounded_box(draw, takeaway_box, fill=route_bg, outline=route_border, width=3)
    tx1, ty1, tx2, _ = takeaway_box
    draw.text((tx1 + 30, ty1 + 24), "一句话抓重点", font=font(30, bold=True), fill=accent)
    draw_wrapped(
        draw,
        (tx1 + 30, ty1 + 80),
        poster.takeaway,
        font(27, bold=True),
        TEXT,
        tx2 - tx1 - 60,
        spacing=12,
        max_lines=4,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / poster.filename
    image.save(path, format="PNG", optimize=True)
    return path


def main() -> None:
    output_dir = Path(__file__).resolve().parent.parent / "images"
    for poster in POSTERS:
        path = render(poster, output_dir)
        print(f"generated: {path.name} ({WIDTH}x{HEIGHT})")


if __name__ == "__main__":
    main()
