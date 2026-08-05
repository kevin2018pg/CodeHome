# RankMixer：原理、工程实现与复习要点

> 论文：**RankMixer: Scaling Up Ranking Models in Industrial Recommenders**（ByteDance，2025）  
> 本文目标：理解 RankMixer 为什么这样设计、每一步的张量变化、工业特征如何变成 token，以及当前仓库实现与论文的差距。
>
> 架构定位：RankMixer 属于非序列特征交叉的 Scaling 路线，不是完整的序列—交叉统一模型。整体演进见[《序列建模、特征交叉与统一架构演进》](./序列建模、特征交叉与统一架构演进.md)。

---

## 一、RankMixer 要解决什么问题

工业精排输入通常包含：

- 高基数 ID：user_id、item_id、author_id；
- 类别特征：类目、设备、渠道、会员等级；
- 数值与统计特征：历史 CTR/CVR、价格、活跃度；
- 行为序列：点击、观看、购买、搜索序列；
- 显式交叉：user-item 相似度、query-item 匹配分。

传统精排主要有两类问题：

1. **交叉模块碎片化**：FM、DCN、Attention、各种人工 Cross Block 混搭，结构复杂。
2. **硬件利用率低**：大量小算子、稀疏访存和不规则交叉不适合现代 GPU，模型 FLOPs 不高，但实际延迟仍大。

RankMixer 的目标不是单纯追求更复杂的交叉公式，而是：

> 用统一、规则、GPU 友好的结构完成异构特征交互，并沿 token 数、宽度、层数和专家数稳定扩展模型容量。

核心组件只有两个：

```text
Multi-Head Token Mixing：参数无关地交换 token 子空间
Per-token FFN：每个 token 使用独立参数做非线性建模
```

整体结构：

```text
原始特征
  → Embedding / Sequence Module
  → Semantic Tokenization
  → RankMixer Block × L
       ├─ Token Mixing + Residual + LN
       └─ Per-token FFN/MoE + Residual + LN
  → Mean Pooling
  → 任务塔（CTR、CVR、时长等）
```

---

## 二、输入特征如何变成 token

### 2.1 字段级 Embedding 与 token 不是一回事

RankMixer 通常不是“一个字段一个 token”，而是：

```text
每个字段先得到 embedding
  → 按语义组织字段
  → 组内拼接、切分和投影
  → 得到少量、等宽的 feature token
```

字段级处理：

```text
高基数 ID       → Embedding / Hash Embedding
低基数枚举      → Embedding
连续值          → 归一化后 Linear，或分桶后 Embedding
Multi-hot       → EmbeddingBag / Attention Pooling
行为序列        → DIN / Transformer / GRU / Pooling
```

Embedding 表一般按字段独立；具有同一语义空间的字段可以共享。例如候选 item_id 与历史行为 item_id 通常共享 Item Embedding。

### 2.2 Semantic-based Tokenization

论文不建议两个极端：

- 每字段一个 token：数百个字段产生数百个 token，小矩阵多、GPU 利用率低，每个 token 分到的容量不足。
- 全部字段一个 token：退化成普通 DNN，强势字段容易淹没长尾字段。

典型语义域：

```text
user_profile
user_statistics
candidate_id
candidate_attributes
context/query
short_sequence
long_sequence
cross_features
```

论文的通用写法是：

```text
1. 各字段 embedding 按语义顺序 concat 成 e_input
2. 将 e_input 按合适的宽度切成 T 段
3. 每一段通过 Proj 映射到统一 D 维

x_i = Proj(e_input[d*(i-1) : d*i])
X_0 = stack(x_1, ..., x_T) ∈ R^(T×D)
```

因此，“语义组”和“token”不一定一一对应：

- 大组可以拆成多个 token；
- 小而相近的组可以合并；
- 重要 ID 或重要序列可以单独占一个 token。

### 2.3 Projector 的作用

假设 user 组包含：

```text
user_id embedding       32维
会员等级 embedding       8维
设备 embedding           8维
用户分层 embedding       8维
5个统计特征投影          5×16维
总计                    136维
```

组内拼接后：

```text
user_group ∈ R^(B×136)
```

Projector：

```text
user_token = LayerNorm(GELU(Linear(136, D)(user_group)))
```

当 `D=320` 时：

```text
[B,136] → [B,320]
```

Projector 同时完成：

1. **维度对齐**：让不同长度的组都变成 D 维；
2. **组内融合**：每个输出维可学习所有组内字段的组合；
3. **非线性变换**：GELU 增强组内表达；
4. **尺度稳定**：LayerNorm 避免某组仅因数值幅度大而支配其他 token。

不要直接对不同字段 embedding 求和。Concat 保留固定字段位置，Projector 才能学习字段专属权重。

### 2.4 行为序列怎么处理

RankMixer 不是长序列编码器。原始序列应先由专门模块压成向量：

```text
[item_1, ..., item_L]
  → item/action/time embedding
  → DIN Target Attention / Transformer / GRU / Pooling
  → e_seq
  → Projector
  → sequence token
```

复杂业务可以拆出多个序列 token：

```text
短期点击 token
长期购买 token
搜索 token
直播观看 token
```

不建议把长度 100 的行为序列直接当作 100 个 RankMixer token。序列模块负责序列内部时序关系，RankMixer 负责 user/item/context/sequence 等域之间的交互。

### 2.5 Token 划分的工程原则

- 语义相近、量级相近的弱字段可以合并；
- 高基数主 ID、重要序列通常独立；
- 避免一个 token 输入维度远大于其他 token；
- 不要为了凑 token 数打乱语义边界；
- `T` 与 `D` 要协同设计，标准 RankMixer 需要 `D % T == 0`；
- 最终以 AUC/UAUC、吞吐、显存和端到端延迟联合选型。

---

## 三、RankMixer Block

设输入：

```text
X_(n-1) ∈ R^(B×T×D)
```

论文采用 Post-Norm：

```text
S_(n-1) = LN(TokenMixing(X_(n-1)) + X_(n-1))
X_n     = LN(PFFN(S_(n-1)) + S_(n-1))
```

一个 Block 有两个子层，因此有两次残差：

1. Token Mixing 负责跨 token 信息交换；
2. PFFN 负责每个 token 内部的非线性建模。

每个子层都保留恒等路径，避免某个子层学坏时破坏原始信息，也有利于梯度传播。

### Post-Norm 与 Pre-Norm

```text
Post-Norm：x → SubLayer → +x → LN
Pre-Norm： x → LN → SubLayer → +x
```

原论文与当前仓库使用 Post-Norm。层数很深时，Pre-Norm 通常更容易训练，但若要改变应做独立消融，不能只移动 LN 而不重新调整初始化、学习率和深度。

---

## 四、Multi-Head Token Mixing

### 4.1 核心不是 Attention

RankMixer 没有：

```text
Q = XW_Q
K = XW_K
V = XW_V
softmax(QK^T)V
```

Token Mixing 是无参数的 `reshape + permute + concat`，主要进行数据重排。

### 4.2 张量变化

输入有 T 个 token，每个 token D 维。把每个 token 均匀切成 H 个 head：

```text
x_t = [x_t^1 | x_t^2 | ... | x_t^H]
x_t^h ∈ R^(D/H)
```

然后将所有 token 的同编号 head 拼在一起：

```text
s^h = concat(x_1^h, x_2^h, ..., x_T^h)
s^h ∈ R^(T*D/H)
```

输出：

```text
S ∈ R^(H × T*D/H)
```

论文设置：

```text
H = T
```

于是：

```text
S ∈ R^(T×D)
```

输出与输入形状一致，才能直接做残差相加。

### 4.3 具体例子

假设：

```text
T = 5
D = 320
H = 5
head_dim = 64
```

输入：

```text
X: [B,5,320]
```

变形：

```text
[B,5,320]
→ [B,5,5,64]      # token × head × head_dim
→ permute
→ [B,5,5,64]      # head × token × head_dim
→ [B,5,320]
```

Mix 后每个新 token 都包含所有原 token 的一个 64 维子空间：

```text
new_token_1 = [user.head1, item.head1, context.head1, seq.head1, cross.head1]
new_token_2 = [user.head2, item.head2, context.head2, seq.head2, cross.head2]
...
```

### 4.4 为什么这样能做交互

Token Mixing 本身只负责“把不同域的信息放到一起”，真正的可学习交互发生在后面的 PFFN：

```text
重排前：一个 token 主要属于一个语义域
重排后：一个 token 同时含所有域的一个子空间
PFFN：对拼在一起的子空间做非线性组合
```

因此不能只看 Token Mixing 而忽略 PFFN。前者负责通信，后者负责学习。

### 4.5 与 Self-Attention 对比

| 维度 | RankMixer | AutoInt/Hiformer |
| ---- | --------- | ---------------- |
| 交互 | 固定重排 + PFFN | 动态 QK 权重 |
| Mixing 参数 | 无 | Q/K/V/O 参数 |
| 两两矩阵 | 无 `T×T` Attention | 有 |
| 计算特点 | 规则、GPU 友好 | 表达动态，但计算/IO 更高 |
| 异构语义 | 由 token 划分和 PFFN 隔离 | Hiformer 用异构投影处理 |

RankMixer 并不声称 Attention 在所有任务都无效，而是认为在工业异构特征交互与严格延迟约束下，参数无关 Mixing 的性能/效率比更合适。

---

## 五、Per-token FFN（PFFN）

### 5.1 完整流程

第 t 个 token 独立经过：

```text
s_t ∈ R^D
  → Linear_t(D, kD)
  → GELU
  → Linear_t(kD, D)
  → v_t ∈ R^D
```

随后在 PFFN 外部做：

```text
x_t' = LN(v_t + s_t)
```

残差不属于 FFN 内部，而属于 Block。

### 5.2 为什么是 `D → kD → D`

- 中间升维到 `kD`，提供更宽的非线性空间；
- 最后回到 D，保证可与输入做残差；
- `k` 是表达力与算力的旋钮，常见为 2～4。

单个 PFFN 参数量近似：

```text
D*kD + kD*D ≈ 2kD²
```

一层 T 个 token：

```text
Param ≈ 2kTD²
```

L 层：

```text
Param ≈ 2kLTD²
```

### 5.3 为什么参数增加而 FLOPs 与共享 FFN近似不变

对比共享 FFN：

```text
共享 FFN：
  T 个 token 都调用同一套 D→kD→D
  参数只有 1 套，计算执行 T 次

PFFN：
  每个 token 调用自己的一套 D→kD→D
  参数有 T 套，计算仍执行 T 次
```

因此相对同宽共享 FFN：

- 参数量约扩大 T 倍；
- 每个样本仍处理 T 个 token，乘加量近似不变；
- 权重读取和实现细节会影响实际延迟，所以“FLOPs 不变”不等于“端到端耗时绝对不变”。

### 5.4 为什么不能共享 FFN

Mix 后的不同 token 对应不同 head 子空间。论文观察到共享 FFN 容易让各 token 表示趋同（representation collapse）。

PFFN 的作用：

- 隔离不同子空间参数；
- 避免高频域支配长尾域；
- 在不加宽单次计算路径的情况下增加容量；
- 为后续 MoE 扩展提供自然位置。

---

## 六、Sparse-MoE 扩展

### 6.1 为什么在 PFFN 上做 MoE

把每个 token 的单个 FFN 扩展为多个 expert：

```text
v_i = sum_j G_(i,j) * Expert_(i,j)(s_i)
```

总参数可随专家数 E 增长，而推理只激活部分专家，计算量受控。

### 6.2 ReLU Routing 与平衡 / 稀疏 Loss

传统 MoE 常用：

```text
Top-k + Softmax
```

并配合 Switch 类 load balancing：

```text
L_balance = α · N · Σ_i (f_i · P_i)   # N=专家数，不是 token 数
```

- `f_i`：实际选中 expert i 的比例（Top-K mask）
- `P_i`：Softmax 平均概率
- 精排 MMoE→SMoE：在**样本维**上统计（对 `B` 平均；多任务则各 Gate 各算一份再相加）
- 若用在 per-token MoE：在**token 维**上统计（对 `B×T` 平均）

RankMixer 主推 ReLU gate + L1，控制整体激活预算（不是 \(f_i\) 均匀性）：

```text
G_{b,t,j} = ReLU(router(s_{b,t}))_j
L_reg = mean_{b,t}( Σ_j G_{b,t,j} )    # 所有样本×token 的平均「门控激活总量」
L     = L_task + λ · L_reg
```

直觉：

- 信息量高的 token 可以激活更多 expert；
- 信息量低的 token 可以少激活；
- 不强迫所有 token 使用完全相同的 top-k 预算；
- L1 压的是总激活量，**不是**每个专家的路由率 \(f_i\)；专家充分训练靠 DTSI。

极端情况下所有 gate 为 0，需要工程上设置数值保护或保底 expert。

两套对照：

```text
Softmax balance：拉齐各专家负载（样本维或 token 维上的 f、P）
ReLU + L1：      把平均激活总量压到稀疏预算（token 维 mean ΣG）
```

更完整的数值例子见[《序列建模、特征交叉与统一架构演进》](./序列建模、特征交叉与统一架构演进.md) §5.6。

### 6.3 Dense Training / Sparse Inference（DTSI）

纯稀疏训练可能导致部分 expert 样本不足、训练不充分。DTSI 的思想：

```text
训练：让更多/全部 expert 获得梯度
推理：只激活稀疏 expert，降低成本
```

论文描述了训练/推理路由配合的方案。当前仓库实现是简化版：

- 训练时可使用 dense gate；
- 推理时对同一 router 做 top-k；
- 没有完整复刻论文中的双 router 与生产调度。

因此当前代码只能称为“DTSI 风格骨架”，不能等同于生产实现。

### 6.4 MoE 常见坑

- 路由塌缩：少数专家吃掉大部分流量；
- 专家欠训练：样本过少；
- gate 全零或数值不稳；
- 训练 dense、推理 sparse 带来分布差异；
- 理论 FLOPs 下降但路由、通信、Gather/Scatter 使延迟不降；
- 专家数增加后显存和参数同步成为瓶颈。

算力有限时，应先把 Dense PFFN 做稳，再上 Sparse-MoE。

---

## 七、Pooling 与多任务

经过 L 层后：

```text
X_L ∈ R^(B×T×D)
```

论文使用 mean pooling：

```text
o = mean(X_L, dim=token) ∈ R^(B×D)
```

再接任务塔：

```text
o → CTR head
o → CVR head
o → 时长/互动/负反馈 head
```

RankMixer 主要是 **Backbone / Feature Interaction Layer**，与多任务结构正交：

- 简单 Shared-Bottom；
- MMoE；
- PLE；
- ESMM；
- 多目标排序融合；

都可以接在 RankMixer 后面。

当前仓库只是：

```text
RankMixer → shared MLP
  ├─ CTR Linear Head
  └─ CVR Linear Head

loss = BCE_ctr + BCE_cvr + aux_loss
score = pCTR * pCVR
```

这不是 MMoE，也不是完整 ESMM。PFFN 内的 Sparse-MoE 是**特征子空间专家**，不是任务级专家，不要混淆。

---

## 八、Scaling Law 与效率理解

Dense RankMixer 的主要参数来自 PFFN：

```text
Param ≈ 2kLTD²
FLOPs ≈ 4kLTD²
```

可沿四个方向扩展：

| 方向 | 增大后主要效果 | 风险 |
| ---- | -------------- | ---- |
| Token 数 T | 更细语义隔离 | 小算子增加，D 必须可整除 |
| 宽度 D | 每个 token 容量显著增加 | 参数/FLOPs 近似按 D² 增长 |
| 层数 L | 增加交互阶数与深度 | 优化困难、延迟线性增长 |
| 专家数 E | 增加总参数 | 路由、显存、通信复杂 |

不要只比较理论 FLOPs。工业部署还要看：

```text
MFU = 实际有效 FLOPs / 硬件理论 FLOPs
```

低 FLOPs 的碎片化模型可能比高 FLOPs、规则大矩阵模型更慢。RankMixer 的核心价值之一，就是把计算规整成更适合 GPU 的算子，提高 MFU。

---

## 九、训练与评估建议

### 9.1 训练顺序

推荐逐步增加复杂度：

```text
1. Semantic Tokenization + Dense PFFN
2. 调 T / D / L / k
3. 检查 token 表示与梯度
4. 再引入 Sparse-MoE
5. 最后接 MMoE/PLE/ESMM 等任务结构
```

不要一次同时引入新 token 划分、MoE 和复杂多任务，否则难以定位收益来源。

### 9.2 需要监控的指标

效果：

- LogLoss；
- AUC、GAUC/UAUC；
- NDCG；
- 分用户、分场景、分冷启动人群指标；
- 多任务之间是否负迁移。

效率：

- 参数量（Embedding 与 Dense 分开统计）；
- FLOPs；
- 训练吞吐；
- GPU MFU；
- P50/P95/P99 推理延迟；
- 显存与模型加载时间。

MoE 额外监控：

- 每 token 平均激活专家数；
- 各专家流量、梯度和更新次数；
- dense train 与 sparse infer 的预测差；
- 路由正则占总 loss 的比例。

### 9.3 初始化与优化

- Embedding 与 Dense 参数可使用不同优化器与学习率；
- 大规模稀疏 Embedding 常用 Adagrad 类优化器；
- Dense Backbone 可用 AdamW/RMSProp，需以平台经验为准；
- 加深网络时重新调 warmup、梯度裁剪和 Norm；
- `D`、`T` 改变后不要沿用原学习率直接比较。

论文的特定优化器配置属于其生产系统，不应机械照搬。

---

## 十、当前仓库代码映射

路径：`rank-mixer/rank_mixer/model.py`

| 论文模块 | 当前代码 | 说明 |
| -------- | -------- | ---- |
| Tokenization | `categorical_embeddings`、`numeric_projections`、`group_projectors` | 简化为一组一个 token |
| Multi-Head Token Mixing | `MultiHeadTokenMixing` | `view + permute`，`H=T` |
| Dense PFFN | `DensePerTokenFFN` | 每 token 独立两层 MLP |
| Sparse MoE | `SparsePerTokenMoE` | 简化 ReLU routing / DTSI |
| Block | `RankMixerBlock` | 两个残差，Post-Norm |
| Pooling | `x.mean(dim=1)` | 与论文一致 |
| 多任务 | `ctr_head`、`cvr_head` | 简单双 head |

当前代码的关键形状：

```text
feature groups = T
model_dim = D
head_dim = D/T

tokenize: [B, 各组输入维度] → [B,T,D]
mixing:   [B,T,D] → [B,T,T,D/T] → permute → [B,T,D]
pffn:     每个 [B,D] → [B,kD] → [B,D]
pool:     [B,T,D] → [B,D]
```

### 当前实现没有完整覆盖

- 真实行为序列编码器；
- 论文的通用 concat 后切片 tokenization；
- 生产级 fused kernel 与高 MFU 优化；
- 完整 DTSI 双路由；
- 分布式稀疏 Embedding；
- MMoE/PLE/ESMM；
- 量化、蒸馏和在线 serving 优化。

### 当前实现需特别留意

1. `model_dim % num_tokens == 0`；
2. 一组一个 token 是工程简化，不是论文唯一方式；
3. Python 循环逐 token/逐 expert 在小实验可用，生产效率不高；
4. `aux_loss` 的定义与论文不完全等价；
5. 当前 CVR 双 head 不能宣称为 ESMM；
6. 合成数据只能验证链路，不能证明模型效果。

---

## 十一、与常见模型对比

| 模型 | 特征交互方式 | 主要特点 |
| ---- | ------------ | -------- |
| DeepFM | FM 二阶 + DNN 高阶 | 经典、便宜 |
| DCNv2 | 显式 Cross Network | 交叉可控 |
| AutoInt | Field-level Self-Attention | 动态交互，但有 QKV 与 `T²` |
| Hiformer | 异构 Attention + Composite Projection | Google，强调异构语义 |
| HHFT | 分块异构 Transformer + Hiformer | 阿里，层级异构建模 |
| RankMixer | 参数无关 Token Mixing + PFFN/MoE | 字节，强调统一结构与 GPU scaling |

算力有限时可优先：

```text
较少 T + 较小 D + 1~2 层 Dense PFFN
```

而不是一开始就上大 Sparse-MoE。是否优于 DeepFM/DCNv2 必须通过同数据、同预算消融验证。

---

## 十二、常见踩坑

### Tokenization

- 每字段一个 token，导致 T 过大；
- 全部字段一个 token，退化成 DNN；
- 将原始长序列直接当大量 RankMixer token；
- 语义完全不同的字段被硬塞入同一 token；
- 各 token 输入规模严重失衡；
- 候选 item 与历史 item 没共享语义 Embedding。

### Token Mixing

- `D` 不能被 `T` 整除；
- 错把 H 当作可随意设置的 Attention head；标准实现取 `H=T`；
- 认为无参数重排本身能学习交互，忽略后面的 PFFN；
- 误以为 Token Mixing 会动态选择重要特征；它没有 Attention 权重。

### PFFN / MoE

- 将 PFFN 的专家与 MMoE 任务专家混淆；
- 只看参数/FLOPs，不测实际延迟；
- MoE 专家欠训练或路由塌缩；
- L1 太大导致 gate 接近全零；
- 训练 dense、推理 sparse 的分布差未校准。

### 多任务与评估

- 简单 CTR/CVR 双 head 被误称为 ESMM；
- 只看整体 AUC，不看 UAUC、分层指标和线上目标；
- Backbone 与多任务结构同时修改，无法归因；
- 离线小数据收益直接外推到工业 scaling 结论。

---

## 十三、高频问答

### 1. RankMixer 是一个完整精排模型吗？

更准确地说，它是可扩展的特征交互 Backbone。Embedding、序列模块、多任务塔、loss 和 serving 仍需业务系统补齐。

### 2. token 是随机初始化的吗？

不是固定的可学习占位 token。每个样本的 token 来自该样本字段 embedding/序列向量，经 Projector 动态计算。Embedding 表和 Projector 参数才是随机初始化并训练的参数。

### 3. 一个字段还是一组字段对应一个 token？

先逐字段 Embedding，再按语义组合为少量 token。大组可拆、小组可合，不要求严格一组一个。

### 4. 为什么 `H=T`？

Mix 输出形状是 `[H, TD/H]`。取 `H=T` 后恢复 `[T,D]`，才能与输入直接做残差。

### 5. PFFN 为什么参数增加但计算近似不变？

共享 FFN 和 PFFN 都要处理 T 个 token、执行 T 次同宽 MLP；区别是共享一套权重还是 T 套权重。

### 6. 为什么一个 Block 有两次残差？

Token Mixing 与 PFFN 是两个独立子层，每个子层各保留一条恒等路径。

### 7. RankMixer 是否完全不需要 Attention？

Backbone 不依赖 Self-Attention，但前置序列模块仍可使用 DIN/Transformer Attention，两者不冲突。

### 8. RankMixer 的 MoE 等于 MMoE 吗？

不等于。RankMixer MoE 扩展 token 内 FFN 容量；MMoE 用任务 gate 组合共享 expert，解决多任务关系。

### 9. 最先调哪个超参数？

先确定合理 token 语义与 T，再调 D、L、k，最后才是 E 与稀疏率。错误 tokenization 往往不是堆参数能补救的。

---

## 十四、30 秒复述

> RankMixer 是字节提出的工业精排特征交互 Backbone。它先将 ID、统计、序列和交叉特征按语义压成 T 个 D 维 token；每个 Block 用无参数 Multi-Head Token Mixing 将所有 token 的同编号子空间重新拼接，再由独立 Per-token FFN 学习非线性交互。设置 H=T 保证 Mixing 前后都是 `[T,D]`，两个子层分别使用残差与 LayerNorm。PFFN 相比共享 FFN增加参数但保持近似计算量，并可进一步扩展成 ReLU Routing 的 Sparse-MoE。模型主要沿 T、D、L、E 四个方向扩展，价值不仅是效果，也包括规则矩阵计算带来的高 MFU。序列编码、多任务 MMoE/ESMM 和生产 serving 都属于 Backbone 外的配套模块。

---

## 参考

- Zhu et al. **RankMixer: Scaling Up Ranking Models in Industrial Recommenders**, arXiv:2507.15551, 2025.
- 论文地址：https://arxiv.org/abs/2507.15551
- 本仓库实现：`rank-mixer/rank_mixer/model.py`
