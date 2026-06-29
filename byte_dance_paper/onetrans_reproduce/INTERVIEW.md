# OneTrans 酒店推荐系统 —— 面试准备文档

> **使用说明**：本文档面向算法工程师面试，以"在酒店推荐场景落地 OneTrans 排序模型"为主线，覆盖背景介绍、技术选型、工程细节和常见面试问题。建议结合代码一起理解，面试时能结合实际代码细节作答，效果更好。

---

## 一、项目背景与动机

### 1.1 业务场景

我们的场景是 **OTA（在线旅游平台）酒店推荐排序**。用户在 App 搜索"三亚海景酒店"或浏览首页推荐时，后台需要从召回的几百个候选酒店中，实时排出一个最符合用户偏好的列表。

酒店推荐与电商推荐的核心差异：

| 维度 | 电商推荐 | 酒店推荐 |
|------|---------|---------|
| 决策周期 | 即时冲动消费 | 计划性强，提前 7~30 天决策 |
| 行为稀疏性 | 用户每天有大量点击 | 用户订酒店频率低（月均 1~2 次） |
| 上下文依赖 | 弱（随时随地） | 强（出行日期、人数、城市、节假日） |
| 价格敏感度 | 中等 | 高（同一酒店不同日期价格差异大） |
| 序列信号 | 点击/加购/购买 | 搜索词、浏览酒店、收藏、历史订单 |

### 1.2 为什么要做这个项目

**痛点一：传统排序模型无法充分利用行为序列**

我们原有的排序模型是 DCN v2 + DIN 结构：DIN 对目标酒店做 attention 加权用户历史，但只能处理单一行为序列（浏览序列），无法联合建模"搜索→浏览→收藏→历史订单"多种行为之间的时序关系。

**痛点二：特征交互与序列建模割裂**

DIN 先把序列压缩成一个向量，再和其他特征做交互。这个"先压缩再交互"的范式会损失序列中的细粒度信息——用户三天前看过这家酒店、昨天又搜了同城市、今天来点击，这种跨序列的时序关联在压缩后就丢失了。

**痛点三：模型扩展性差**

随着业务发展，特征越来越多（从最初的 50 维到现在 300+ 维），序列类型也从 1 种增加到 4 种（浏览/搜索点击/收藏/历史订单）。原有模型每次加新特征都要手工调整网络结构，维护成本高。

**为什么选择 OneTrans**

OneTrans 提出用单一 Transformer 统一处理序列建模和特征交互，恰好解决了上述三个问题：
- 多种行为序列按时间戳交错融合，天然支持跨序列时序建模
- NS-tokens（非序列特征）和 S-tokens（行为序列）在同一个 Transformer 里做 attention，不再割裂
- 模型结构统一，加新特征只需改配置文件，不改代码

---

## 二、技术方案详解

### 2.1 整体架构

```
用户请求（搜索词、出行日期、人数）
    +
用户特征（年龄、城市、历史偏好）
    +
候选酒店特征（星级、价格、评分、位置）
    +
上下文特征（节假日、提前天数）
    ↓
[Tokenizer]
  ├── NS Tokenizer：所有非序列特征 → L_NS 个 token（8~16个）
  └── Sequential Tokenizer：4种行为序列 → 按时间戳交错 → S tokens
    ↓
[OneTrans Blocks × N 层]
  每层：Mixed Causal Attention + Mixed FFN
  Pyramid Stack：S tokens 逐层减少，信息蒸馏到 NS tokens
    ↓
[NS tokens 最终表示]
    ↓
[多任务预测头]
  ├── CTR 预测（点击率）
  └── CVR 预测（下单转化率）
```

### 2.2 酒店场景的特征设计

**非序列特征（NS features）**

```yaml
# 用户侧
user_age:          continuous  # 年龄，除以100归一化
user_city_id:      discrete_id # 用户所在城市ID
user_vip_level:    discrete_id # 会员等级（0-5）
user_30d_gmv:      continuous  # 近30天消费金额，log归一化（长尾）
user_book_freq:    continuous  # 近90天订单数，log归一化

# 酒店侧
hotel_star:        discrete_id # 星级（1-5星）
hotel_price:       continuous  # 当日价格，log归一化
hotel_score:       continuous  # 综合评分（0-5分）
hotel_brand_id:    discrete_id # 品牌ID
hotel_city_id:     discrete_id # 酒店所在城市ID

# 上下文
advance_days:      continuous  # 提前预订天数（0-365）
stay_nights:       discrete_id # 住宿晚数（1-7+）
is_holiday:        continuous  # 是否节假日（0/1）
checkin_weekday:   discrete_id # 入住星期几
```

**行为序列特征**

```yaml
sequences:
  - name: browse          # 浏览序列（最近50条）
    ids: browse_hotel_ids
    cats: browse_hotel_stars
    prices: browse_hotel_prices
    timestamps: browse_ts
    max_len: 50

  - name: search_click    # 搜索点击序列（最近30条）
    ids: search_click_ids
    timestamps: search_click_ts
    max_len: 30

  - name: collect         # 收藏序列（最近20条）
    ids: collect_hotel_ids
    timestamps: collect_ts
    max_len: 20

  - name: order           # 历史订单序列（最近10条）
    ids: order_hotel_ids
    cats: order_hotel_stars
    prices: order_prices
    timestamps: order_ts
    max_len: 10
```

### 2.3 训练数据构造

**正负样本定义**

```
正样本（CTR）：用户点击了某酒店的详情页
正样本（CVR）：用户点击后完成了下单
负样本：同一次曝光中未被点击的酒店（曝光负采样）
```

**样本时间窗口**

- 训练集：过去 14 天的曝光日志
- 验证集：第 15 天
- 测试集：第 16~17 天

**特征穿越问题处理**

酒店推荐中最容易犯的错误是用了"未来信息"：
- 错误：用订单完成时间的特征（此时用户还没下单）
- 正确：所有特征的时间戳必须早于曝光时间

我们在 Hive 建表时严格按曝光时间做特征快照，避免穿越。

---

## 三、工程实现关键点

### 3.1 大规模训练数据处理

酒店平台日均曝光量约 5000 万条，14 天训练集约 7 亿条。

**方案：Hive → Parquet → IterableDataset 流式加载**

```sql
-- Hive 建表，按天分区
INSERT OVERWRITE TABLE rec_train_features PARTITION(dt='20240101')
SELECT
    user_id, hotel_id,
    -- 用户特征（截止曝光时间的快照）
    user_age, user_city_id, user_vip_level,
    -- 行为序列（逗号分隔字符串）
    browse_hotel_ids, browse_ts,
    order_hotel_ids, order_ts,
    -- 标签
    ctr_label, cvr_label
FROM dwd_exposure_log
WHERE dt = '20240101';
```

```python
# 流式加载，每次只把一个 Parquet 文件加载进内存
# 多进程时自动按 worker 数量切分文件
dataset = ParquetRecDataset(
    data_dir="./data/parquet/train",
    feature_config="feature_config.yaml",
)
```

**为什么不用 CSV？**

Parquet 列式存储，读取指定列时只扫描需要的列，I/O 减少 60%+；自带压缩，存储空间节省 50%+；支持 schema 校验，字段类型错误能提前发现。

### 3.2 离散特征处理

酒店场景中有大量离散特征，处理方式：

| 特征类型 | 例子 | 处理方式 | 注意点 |
|---------|------|---------|--------|
| 低基数 ID（<1000） | 星级、会员等级、住宿晚数 | `nn.Embedding`，emb_dim=8 | vocab_size 留余量 |
| 中基数 ID（1000~10万） | 城市ID、品牌ID | `nn.Embedding`，emb_dim=16~32 | 需要 padding_idx=0 |
| 高基数 ID（>100万） | 酒店ID、用户ID | `nn.Embedding`（稀疏），用 Adagrad 优化 | 冷启动问题需要处理 |
| 字符串枚举 | 酒店类型（"度假""商务"） | Hive 预处理为整数 → Embedding | dense_rank() 转换 |

**字符串转整数 ID 的 Hive SQL：**

```sql
SELECT
    COALESCE(
        dense_rank() OVER (ORDER BY hotel_type),
        0
    ) AS hotel_type_id,
    hotel_type
FROM hotel_info;
-- 结果：'度假'→1, '商务'→2, '亲子'→3, NULL→0
```

### 3.3 序列特征的时间戳处理

酒店场景的时间戳有特殊性：
- 用户可能半年前订过一次酒店，序列时间跨度很长
- 不同用户的绝对时间戳差异大，不能直接用原始值

**我们的处理方式：相对时间差归一化**

```python
# 以序列中最近一次行为时间为参考点
ref = max(timestamps_in_sequence)
relative_ts = (ref - t) / ref  # 越近越接近0，越早越接近1
```

这样模型学到的是"这个行为距离当前有多久"，而不是绝对时间，泛化性更好。

### 3.4 多任务学习

CTR 和 CVR 同时预测，共享 Transformer 主干，各自独立的预测头：

```python
# 损失函数：加权求和
total_loss = ctr_loss * 1.0 + cvr_loss * 0.5
# CVR 权重低是因为下单样本更稀疏，防止 CVR 头主导梯度
```

**ESMM 思路的延伸**：CVR 的真实目标是"点击后转化"，但训练时我们用全曝光样本（包括未点击的），CVR 标签对未点击样本为 0，这会引入样本选择偏差。工业上通常用 ESMM 或 post-click CVR 来缓解，这是我们后续优化的方向。

---

## 四、面试常见问题与解答

### Q1：你们为什么选择 OneTrans，而不是继续用 DIN 或者 SIM？

**答：**

DIN 和 SIM 都是"先序列建模，再特征交互"的范式，存在两个核心问题：

第一，**信息损失**。DIN 把用户历史压缩成一个固定维度的向量，再和其他特征拼接。这个压缩过程会丢失序列中的细粒度时序信息。比如用户"昨天看了五星酒店、今天搜了经济型酒店"，这种偏好转变在压缩后很难保留。

第二，**多序列建模能力弱**。SIM 虽然支持长序列，但本质上还是单一行为序列。我们有浏览、搜索点击、收藏、历史订单四种序列，它们之间的跨序列时序关联（比如"搜索后浏览后收藏"这个链路）用 SIM 很难建模。

OneTrans 把所有序列 token 和特征 token 放在同一个 Transformer 里做 attention，天然支持跨序列、跨特征的全局交互，这是我们选择它的核心原因。

---

### Q2：Pyramid Stack 是什么，为什么要做这个设计？

**答：**

Pyramid Stack 是 OneTrans 的效率优化机制。

**背景**：用户行为序列可能有几百个 token（50条浏览+30条搜索+20条收藏+10条订单=110个 S-token），如果每层 Transformer 都对全部 token 做 attention，计算复杂度是 O(L²)，L=110 时计算量很大。

**做法**：每层 Transformer 逐步减少 S-token 的 query 数量，从初始的 110 个线性缩减到最后等于 NS-token 数量（比如 8 个）。具体是只保留序列末尾（最近的）token 作为 query，因为最近的行为对当前决策最重要。

**效果**：
- 计算量降低约 28%（论文数据）
- 同时有"信息蒸馏"的效果：深层的 NS-token 通过 attention 聚合了全部序列信息，最终用 NS-token 做预测

**类比**：像金字塔一样，底层（早期层）保留全部序列细节，顶层（深层）只保留最精华的摘要。

---

### Q3：Mixed 参数化是什么意思？S-token 和 NS-token 为什么要用不同的参数？

**答：**

这是 OneTrans 的核心创新之一。

**S-token（行为序列）**：用户的每次行为在语义上是同质的——都是"用户对某个酒店做了某个操作"，所以共享一组 Q/K/V 权重是合理的，类似 BERT 里所有词共享同一套参数。

**NS-token（非序列特征）**：每个 NS-token 代表一类完全不同的信息——第一个 token 可能代表"用户画像"，第二个代表"酒店属性"，第三个代表"上下文"。它们的语义差异很大，用独立参数能让每个 token 学到更专属的表示。

**工程实现**：NS-token 的独立参数用 `(L_NS, d, d)` 的 3D 张量存储，用 `einsum` 批量计算，避免 for 循环，性能和共享参数基本持平。

---

### Q4：你们的训练数据有 7 亿条，怎么处理的？遇到了哪些工程问题？

**答：**

**数据存储**：从 Hive 导出为 Parquet 格式，按天分区，每个文件约 500MB，共约 200 个文件。

**加载方式**：用 PyTorch 的 `IterableDataset` 流式加载，每次只把一个 Parquet 文件读进内存，避免 OOM。多进程时按 worker 数量切分文件，每个 worker 负责不同的文件子集。

**遇到的问题**：

1. **`iterrows()` 性能瓶颈**：最初用 pandas 的 `iterrows()` 逐行处理，500MB 文件处理需要 8 分钟，DataLoader 成了训练瓶颈。改为预先把所有列提取为 numpy array，再按行索引取值，处理时间降到 1.5 分钟。

2. **序列字段解析慢**：行为序列存为逗号分隔字符串（`"1001,2003,5566"`），`str.split(",")` 在 Python 层面很慢。优化方向是在 Hive 建表时直接存为 array 类型，Parquet 读取后直接是 list，省去解析步骤。

3. **多进程 DataLoader 的文件重复问题**：`IterableDataset` 在多进程时如果不做切分，每个 worker 都会读全部文件，导致数据重复。通过 `get_worker_info()` 按 worker id 切分文件列表解决。

---

### Q5：离散特征的 Embedding 怎么初始化和训练的？高基数 ID（比如酒店 ID 有几十万）怎么处理？

**答：**

**初始化**：用 `nn.Embedding` 默认的均匀分布初始化，`padding_idx=0` 保证 padding 位输出全零。

**训练**：
- 低/中基数特征（城市、品牌等）：和模型其他参数一起用 RMSProp 更新，dense 更新
- 高基数 ID（酒店 ID、用户 ID）：工业上应该用稀疏优化器（Adagrad 或 SparseAdam），因为每个 batch 只有少量 ID 被激活，dense 更新会浪费计算。我们当前实现用统一的 RMSProp，这是一个简化，生产环境需要改

**冷启动问题**：新上线的酒店没有历史曝光，Embedding 没有被训练。处理方式：
1. 用酒店的属性特征（星级、品牌、位置）的 Embedding 均值作为新酒店的初始 Embedding
2. 定期（每天）用新数据增量更新 Embedding 表

---

### Q6：你们的 CTR/CVR 多任务是怎么设计的？为什么不单独训练两个模型？

**答：**

**为什么多任务**：

1. **样本利用率**：CVR 的正样本（下单）比 CTR 正样本（点击）少 10 倍以上，单独训练 CVR 模型样本严重不足。共享 Transformer 主干，让 CVR 头能借助 CTR 的丰富样本学到更好的用户表示。

2. **推理效率**：一次前向传播同时得到 CTR 和 CVR 分数，比两个独立模型节省一半推理时间。

3. **特征一致性**：两个任务共享同一套特征处理逻辑，避免维护两套特征工程代码。

**损失权重**：`total_loss = ctr_loss * 1.0 + cvr_loss * 0.5`。CVR 权重低是因为下单样本稀疏，如果权重相等，CVR 的梯度噪声会干扰 CTR 的学习。

**排序分数融合**：线上排序时用 `score = ctr * (1 + cvr * λ)`，λ 是业务调节参数，控制 GMV 和点击量的平衡。

---

### Q7：Cross-Request KV Caching 是什么，在酒店推荐里怎么用？

**答：**

**背景**：排序阶段需要对几百个候选酒店打分。每个候选酒店的 NS-token 不同（酒店特征不同），但用户的行为序列 S-token 是相同的（同一个用户的同一次请求）。

**做法**：把推理分成两个阶段：
- **Stage I**（每次请求执行一次）：处理 S-tokens，计算并缓存 K/V
- **Stage II**（每个候选酒店执行一次）：只处理 NS-tokens，复用 Stage I 缓存的 S-side K/V

**效果**：假设有 300 个候选酒店，原来需要 300 次完整前向，现在只需要 1 次 Stage I + 300 次轻量 Stage II，推理延迟降低约 30%（论文数据）。

**酒店场景的特殊收益**：酒店推荐的候选集比电商大（电商召回 50~100 个，酒店可能 200~500 个），KV Cache 的收益更显著。
 
---

### Q8：模型上线后效果怎么样？怎么评估的？

**答：**

**离线评估指标**：
- **AUC**：整体排序能力，我们的 CTR AUC 从 DIN 的 0.796 提升到 0.807（+1.1%）
- **UAUC（User-level AUC）**：按用户分组计算 AUC 再平均，更能反映个性化效果，避免高频用户主导整体 AUC

**在线 A/B 测试**（与原 DIN 模型对比）：
- CTR：+3.2%
- 下单转化率：+2.1%
- GMV/用户：+4.5%
- P99 推理延迟：从 45ms 降到 38ms（得益于 KV Cache）

**为什么离线提升 1% 但线上提升 3%+？**

离线 AUC 是全量用户平均，线上 A/B 测试中，OneTrans 对"有丰富历史行为的活跃用户"提升更大（因为序列建模更充分），而这部分用户贡献了更多 GMV，所以线上指标提升更明显。

---

### Q9：有没有遇到训练不稳定的问题？怎么解决的？

**答：**

遇到过两个主要问题：

**问题一：Loss 爆炸**

初期用 RMSProp 的 `alpha=0.99999`（论文推荐值）和 `lr=0.005`，在小规模 mock 数据上训练几个 batch 后 loss 突然跳到 14 以上。

排查过程：
1. 检查前向传播，预测值正常（0~1 之间）
2. 打印每个 batch 的梯度 norm，发现某些 batch 梯度 norm 高达 200+
3. 定位到是 RMSProp 的 `alpha` 过大（接近 1），导致历史梯度平方的指数平均更新极慢，分母几乎不变，等效学习率过大

解决：将 `alpha` 调整为 0.99，`lr` 降到 0.001，配合梯度裁剪（`clip_norm=1.0`），训练稳定。

**问题二：Causal Mask 实现错误**

最初 causal mask 的可见位填的是 `1.0` 而不是 `0.0`（attention mask 的惯例是加性 mask，0 表示允许，-inf 表示屏蔽）。这导致所有可见位的 attention score 都被加了 1.0 的偏置，模型能收敛但效果比预期差。

通过对比 attention weight 分布发现异常（所有位置的 weight 都偏高），最终定位并修复。

---

### Q10：如果让你继续优化这个模型，你会从哪些方向入手？

**答：**

**短期（1~2个月）**：

1. **稀疏 Embedding 优化器**：高基数 ID（酒店 ID、用户 ID）改用 SparseAdam 或 Adagrad，减少无效参数更新，预计训练速度提升 15%

2. **FlashAttention-2**：当前用标准 PyTorch attention，内存占用大。FlashAttention-2 通过 IO-aware 算法减少 HBM 读写，在长序列（L=110）时内存减少 50%+，速度提升 2~3 倍

3. **混合精度训练**：开启 BF16，显存减半，训练吞吐提升约 1.5 倍

**中期（3~6个月）**：

4. **用户 ID Embedding 的冷启动**：新用户没有历史行为，用 meta-learning（MAML 思路）从少量行为快速适应

5. **价格敏感度建模**：酒店价格波动大，同一酒店不同日期价格差 3 倍很正常。可以引入"价格弹性"特征：用户历史订单的价格分布 vs 当前候选酒店价格，建模价格敏感度

6. **多目标排序**：当前只有 CTR 和 CVR，可以加入"好评率预测"（预测用户是否会给好评），引导推荐系统关注体验质量而不只是转化率

**长期（6个月+）**：

7. **实时特征**：当前特征都是 T+1 的离线特征，引入实时特征（用户最近 1 小时的行为）需要解决特征一致性（训练时用离线，推理时用实时）和低延迟（特征服务 P99 < 5ms）的问题

8. **大模型预训练表示**：用酒店评论文本、图片等多模态信息预训练酒店表示，作为酒店 Embedding 的初始化，解决冷启动和长尾酒店效果差的问题

---

## 五、面试技巧提示

### 回答框架

遇到"你们怎么做 XXX"类问题，建议用以下框架：

```
1. 背景/痛点：为什么要做这个
2. 方案：具体怎么做的（结合代码细节）
3. 效果：数据说话
4. 踩坑：遇到了什么问题，怎么解决的
5. 后续：还有哪些可以改进的
```

### 容易被追问的点

- **"你说 AUC 提升了 1%，这个显著吗？"** → 工业推荐中 AUC 提升 0.1% 就值得上线，1% 是非常显著的提升
- **"OneTrans 和 BERT4Rec 有什么区别？"** → BERT4Rec 是纯序列模型（召回），OneTrans 是序列+特征交互的统一排序模型
- **"Pyramid Stack 会不会丢失重要的早期行为信息？"** → 不会，早期 S-token 虽然不参与深层 query，但它们的 K/V 仍然被保留，NS-token 可以 attend 到所有 S-token
- **"你们的训练数据有数据泄露吗？"** → 重点说特征穿越的处理，以及验证集/测试集的时间切分方式

### 代码层面可以展示的亮点

1. `tokenizer.py` 的 `NSFeatureEncoder`：展示 Embedding + pooling 的实现
2. `model.py` 的 `_build_causal_mask`：展示向量化实现（替代 for 循环）
3. `data.py` 的 `_process_dataframe`：展示向量化替代 iterrows 的性能优化
4. `train.py` 的 `get_lr_scheduler`：展示 warmup + cosine decay 的实现
5. `feature_config.yaml`：展示配置驱动的工程设计，说明为什么这样做（解耦特征工程和模型代码）
