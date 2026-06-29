# 图召回（GNN：DeepWalk / Node2Vec / EGES / PinSage / LightGCN）· 实战手记

> 视角：召回算法工程师。图召回把 user/item 当节点、交互当边，用**图结构**学 embedding，核心优势是**捕捉高阶关系 + 缓解稀疏**。淘宝 EGES、Pinterest PinSage 是工业经典。

---

## 一、一句话讲清

把行为构成图（item-item 共现图 / user-item 二部图），用图算法（随机游走 or 图卷积）学节点 embedding，再走 ANN 召回。**高阶连通**：A-B 没直接共现，但 A-C-B 连得上，也能学到 A、B 相似。

> 和 I2I 共现的区别：I2I 只看一阶共现，图召回能传播多阶关系，长尾/稀疏节点也能借邻居信息学好。

---

## 二、两大流派

### 2.1 随机游走类（Graph Embedding）
- **DeepWalk**：图上随机游走生成"节点序列"，当句子喂 word2vec(skip-gram) 学 embedding。
- **Node2Vec**：带偏置的游走(p/q 参数控制 BFS/DFS)，平衡同质性(社区)和结构性。
- **EGES（淘宝，重点）**：在 DeepWalk 基础上融合 **side information**(类目/品牌/店铺)，每个 item 的多种属性 embedding 加权融合 → **冷启 item 也有合理向量**(靠属性)。解决纯 id graph embedding 的冷启问题。

### 2.2 图神经网络类（GNN）
- **GraphSAGE / PinSage**：采样邻居 + 聚合(mean/pool/attention)，归纳式(inductive)能给新节点生成向量；PinSage 是 Pinterest 工业落地(随机游走采样 + 重要性聚合)。
- **LightGCN（推荐主力）**：去掉 GCN 的特征变换和非线性，只保留邻居聚合 + 层组合，简单高效，专为协同过滤设计，效果好。

---

## 三、线上链路

1. 离线建图(user-item / item-item)；
2. 训练节点 embedding(游走或 GNN)；
3. item embedding 建 ANN；
4. I2I 形态：trigger item 找近邻；U2I 形态：user 节点向量检索 item。

---

## 四、版本迭代

### v0 · DeepWalk/Node2Vec
纯 id 游走 embedding。问题：冷启 item 没向量、side info 没用上。

### v1 · EGES
融合类目/品牌等 side info，冷启 item 靠属性也能得到向量。效果：冷启覆盖提升。

### v2 · LightGCN
换 GNN，显式建模 user-item 二部图高阶连通。效果：协同信号更强，召回质量提升。

### v3 · 归纳式 + 大规模工程化
PinSage/GraphSAGE 采样聚合，支持新节点和亿级图；邻居采样 + 负采样优化训练效率。

> 负采样体系(未曝光随机负打底 + 中段难负 + 假负过滤等)跨通路通用，详见 [`负样本挖掘和召回效果评估.md`](../负样本挖掘和召回效果评估/负样本挖掘和召回效果评估.md)。图召回里还要额外注意：随机负可能采到图上的高阶邻居(其实相关)，是一类 false negative。

---

## 五、常见坑

**坑 1 · 冷启节点无 embedding**
原因：纯 id transductive 方法学不出新节点。解：EGES(side info)、GraphSAGE(归纳式聚合邻居)。

**坑 2 · 超级节点(热门)主导传播**
现象：热门节点连边太多，污染邻居聚合。解：邻居采样限制度数、热门降权、归一化(LightGCN 的对称归一)。

**坑 3 · 图太大训练不动**
解：邻居采样(GraphSAGE)、子图采样、负采样；分布式图引擎。

**坑 4 · 过平滑(over-smoothing)**
现象：GNN 层数多了所有节点向量趋同。解：层数控制(2~3 层)、LightGCN 层组合、残差。

**坑 5 · 图构建噪声**
原因：误点击/爬虫/活跃用户造假边。解：边权过滤、活跃用户降权、时间衰减。

---

## 六、常见问答（带追问）

**Q1：图召回比 I2I 共现强在哪？**
I2I 只看一阶共现，稀疏/长尾节点共现少就学不好；图召回通过多阶传播让节点借邻居信息，高阶关系和长尾覆盖更好。

**Q2：EGES 解决了什么问题？**
纯 id graph embedding 的**冷启动**——新 item 没游走序列就没向量。EGES 融合 side info(类目/品牌)，新 item 靠属性 embedding 也能得到合理向量。
> 追问·side info 怎么融合？ → 每个 item 的 id + 各属性各一个 embedding，用可学习权重(attention)加权求和成最终 item 向量。

**Q3：LightGCN 为什么去掉特征变换和非线性？**
推荐的协同过滤里，节点本身没丰富特征，GCN 的特征变换+非线性反而增加训练难度、易过拟合；去掉后只保留邻居聚合(协同信号本质)，更简单、更好、更快。

**Q4：transductive 和 inductive 区别？**
transductive(DeepWalk/LightGCN)只能给训练时见过的节点向量，新节点要重训；inductive(GraphSAGE/PinSage)学的是"聚合函数"，能给新节点现场生成向量，工业更友好。

**Q5：GNN 过平滑怎么办？**
层数别太深(2~3 层)、用 LightGCN 的层组合(各层加权)、加残差/初始连接(APPNP 思路)。

---

## 七、选型一句话

- 行为稀疏、长尾长、要高阶关系 → 图召回。
- 有大量 side info 且冷启严重 → EGES。
- 标准协同过滤要简单高效 → LightGCN。
- 图规模大 + 频繁新节点 → GraphSAGE/PinSage(归纳式)。

---

## 八、监控指标

- 离线：Recall@K、NDCG、长尾/冷启子集表现。
- 在线：召回覆盖率、长尾占比、冷启 item 起量、pvctr/uvctr。
