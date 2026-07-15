# 酒店券包多干预 Uplift · 实践

> 视角：营销算法工程师。场景不是简单预测“谁会下单”，而是面对不发券和多种酒店券包，判断**给这个用户哪一种券包，能够带来最大的真实增量价值**。本文覆盖因果理论、随机实验、样本构造、多干预模型、离线评估、预算约束、上线策略和常见追问。

---

## 一、项目介绍

酒店平台有多档券包可投放，例如：

| Treatment | 示例 | 成本与适用人群 |
|---|---|---|
| `T=0` | 不发券 | 对照组，无补贴 |
| `T=1` | 满 300 减 20 | 成本低，适合自然转化意愿较高的人群 |
| `T=2` | 满 500 减 50 | 刺激中高客单用户 |
| `T=3` | 满 800 减 100 | 补贴高，适合高价值但价格敏感的人群 |
| `T=4` | 连住/高星酒店专属券包 | 强场景约束，覆盖特定需求 |

如果只训练普通 CVR 模型，模型会把券发给“本来就会订酒店”的高意向用户。这些用户转化率很高，但大量订单即使不发券也会发生，最终只是用补贴购买自然订单。

Uplift 要识别四类人：

| 人群 | 不发券 | 发券 | 策略 |
|---|---:|---:|---|
| Persuadable 可说服人群 | 不下单 | 下单 | 核心投放对象 |
| Sure Thing 自然转化人群 | 下单 | 下单 | 尽量不补贴 |
| Lost Cause 无法转化人群 | 不下单 | 不下单 | 不浪费券 |
| Sleeping Dog 负向人群 | 下单 | 不下单 | 必须避免干预 |

项目目标不是最大化发券后的 CVR，而是：

> 在预算、券库存、频控和业务规则约束下，为每个用户选择增量利润最大的券包；如果所有券包的增量价值都不为正，就不发券。

完整链路：

```text
随机多组实验
  ↓
构造 treatment / outcome / pre-treatment features / propensity
  ↓
估计每种券包的潜在结果 μ_t(x)
  ↓
计算相对不发券的 CATE：τ_t(x)=μ_t(x)-μ_0(x)
  ↓
把转化增量换算成增量利润，扣除券成本
  ↓
预算约束下选择最优 treatment
  ↓
在线 A/B 验证增量转化、增量利润和补贴效率
```

### 1.1 先用三个真实项目建立统一认知

三个项目表面上分别属于活跃增长、乘客定价和广告扶持，底层其实是同一个问题：

| 项目 | Treatment | Outcome | 模型要回答的问题 | 最终策略 |
|---|---|---|---|---|
| 百度搜索悬浮球 | 展示/不展示活动入口 | 未来 7 日搜索量 | 哪些用户展示后会额外增加搜索 | 圈选搜索增量最大的人群 |
| 快车/特惠补贴对 | 33 个补贴组合 | 快车、特惠、出租车 ABR | 每个补贴对会怎样改变三个业务的应答率 | 最大化 `GMV + λ×毛利` |
| Bidding 流量扶持 | 不扶持/不同扶持或出价力度 | 曝光、转化、GMV、消耗 | 哪些流量会被扶持金真正撬动 | 预算内把资源给边际回报最高的流量 |

统一写法：

$$
\mu_{t,m}(x)=\mathbb{E}[Y_m(t)\mid X=x]
$$

- `t`：干预动作，可以是二元，也可以是多档补贴/出价；
- `m`：业务目标，可以只有一个，也可以同时有快车、特惠、出租车等多个；
- 二元单目标：只需估计 `μ_1(x)-μ_0(x)`；
- 多干预多目标：要估计一个矩阵：

```text
                 快车ABR      特惠ABR      出租车ABR
补贴对0          μ_0,fast     μ_0,bargain  μ_0,taxi
补贴对1          μ_1,fast     μ_1,bargain  μ_1,taxi
...
补贴对32         μ_32,fast    μ_32,bargain μ_32,taxi
```

Uplift 模型负责估计“动作会改变什么”，策略层负责把这些变化换算成业务价值并选动作。**模型层和决策层不能混为一谈。**

### 1.2 项目一：百度 APP 搜索活跃度 Uplift

#### 正确的因果定义

```text
X：展示悬浮球之前的用户画像、历史搜索活跃度、场景特征
T：是否展示悬浮球，1=展示，0=不展示
Y：分组后7日内搜索量
τ(x)=E[Y(1)-Y(0)|X=x]
```

普通搜索量回归模型找的是“未来搜索最多的人”，其中大量用户本身就是高活跃用户；Uplift 找的是“展示悬浮球后比不展示多搜多少的人”，这才是活动入口真正能影响的人。

#### XGBoost 在这里可能有三种建法

1. **T-Learner**：展示组和不展示组分别训练一个回归/分类模型，再作差；
2. **S-Learner**：把 `treatment` 作为 XGBoost 的一个特征，推理时分别设置 `T=1/0` 两次预测再作差；
3. **直接 Uplift Tree/Forest**：分裂目标直接优化 treatment-control 的响应差异。

仅凭“模型使用 XGBoost”无法判断是哪一种，必须看 treatment 是作为特征进入一个模型，还是拆成两个模型。

#### 为什么把搜索量分成 10 档可能缓解过拟合

7 日搜索量是典型长尾计数：

```text
大量用户：0～少量搜索
少数重度用户：几十甚至几百次搜索
```

直接回归原始搜索量时，平方误差会被极少数超高活跃用户主导，树容易记住这些异常样本。分成 10 档相当于：

- 截断极端搜索量的梯度影响；
- 把“精确预测 103 次还是 117 次”改成“识别处于哪个活跃层级”；
- 降低标签噪声，让模型更多学习稳定的人群差异。

但它不是免费的改进：

- 分箱会损失同一档内部的增量大小；
- 如果只预测档位概率，最终应计算期望搜索量或期望档位后再作差，不能直接把两个分类标签相减；
- 若业务目标仍是“增量搜索次数”，每档最好用该档真实搜索量均值作为 value：

$$
\hat Y_t(x)=\sum_{b=1}^{10}P(B=b\mid X=x,T=t)\cdot \overline{Y}_b
$$

然后计算：

$$
\hat\tau(x)=\hat Y_1(x)-\hat Y_0(x)
$$

更直接的替代方案包括 `log1p(search_count)` 回归、Poisson/Tweedie objective、Huber loss、顶部 winsorize。是否优于十档分类应由时间外 AUUC 和线上增量搜索验证。

#### DART 为什么可能有效

XGBoost 的 `booster=dart` 会在每轮随机丢弃一部分已有树，作用类似神经网络 Dropout：

- 减少后续树过度依赖前面少数强树；
- 降低树集成对训练样本细节的记忆；
- 对高维用户画像、长尾标签可能比普通 `gbtree` 更抗过拟合。

但 DART 只是一种正则化手段，不能修复 treatment 选择偏差、特征穿越或错误的评估口径。

#### 评估注意

- AUUC/Qini 必须基于未参与训练的实验样本；
- 若 `Y` 是连续搜索量，需要自定义累计增量曲线；很多现成 `sklift` 指标默认二元 outcome，不能机械套用；
- uplift 分桶图应看每桶的真实：

$$
\overline{Y}_{treat,b}-\overline{Y}_{control,b}
$$

而不只是展示组的平均搜索量。

### 1.3 项目二：33 个补贴对 × 三业务 ABR

这是一个标准的**多干预、多结果、带利润决策的因果响应建模**问题：

```text
T ∈ {0,...,32}：33个快车/特惠补贴对
Y_fast：快车 ABR
Y_bargain：特惠 ABR
Y_taxi：出租车 ABR
```

即使补贴只直接作用于快车和特惠，也要预测出租车 ABR，因为补贴可能产生跨品类替代：用户从出租车迁移到快车，或者从快车迁移到特惠。只看单一业务会把“业务间搬量”误判成大盘增量。

#### TARNet 在做什么

经典 TARNet：

```text
用户/订单/供需特征 X
        ↓
共享表示层 Φ(X)
        ├─ Treatment Head 0 → μ_0(X)
        ├─ Treatment Head 1 → μ_1(X)
        ├─ ...
        └─ Treatment Head 32 → μ_32(X)
```

多业务版本可让每个 treatment head 输出三个 ABR，或在 treatment head 下再拆三个 task head：

```text
Φ(X)
 ├─ 补贴对0
 │   ├─ 快车ABR
 │   ├─ 特惠ABR
 │   └─ 出租车ABR
 ├─ ...
 └─ 补贴对32
```

一条样本实际只接受一个补贴对，因此训练时只能监督真实 treatment 对应的三个输出，其他 32 个 treatment 输出是待估计的反事实。

#### 残差连接 + 门控的正确理解

设共享表示为 `h=Φ(X)`，某个 treatment/task 的专属网络为 `F_{t,m}(h)`。

普通 TARNet：

$$
h_{t,m}=F_{t,m}(h)
$$

加入残差：

$$
h_{t,m}=F_{t,m}(h)+P_{t,m}(h)
$$

其中 `P` 在维度相同时可以是恒等映射，维度不同时必须做线性投影。它的主要作用不是“保留原始输入 X”，而是**保留共享层表示 h**，让专属层学习相对共享表示的增量修正，并改善深层网络的梯度传播。

再加入门控：

$$
g_{t,m}=\sigma(W_{t,m}h+b_{t,m})
$$

一种常见写法：

$$
h_{t,m}=F_{t,m}(h)+g_{t,m}\odot P_{t,m}(h)
$$

或者：

$$
h_{t,m}=g_{t,m}\odot F_{t,m}(h)
+(1-g_{t,m})\odot P_{t,m}(h)
$$

门控让模型按样本、treatment 和业务动态决定：当前预测更依赖公共供需规律，还是依赖该补贴对的特异响应。它是 TARNet 的工程增强，不是 TARNet 因果识别成立的来源；因果可信度仍来自随机分配/正确 propensity 和 overlap。

#### 33 个补贴对如何排序

模型首先输出：

```text
pred[t] = [ABR_fast(t), ABR_bargain(t), ABR_taxi(t)]
```

然后结合各业务预估冒泡量、客单价、抽佣和补贴成本，计算：

$$
GMV_t=
\sum_m Bubble_m\cdot \hat{ABR}_{t,m}\cdot Price_m
$$

$$
Profit_t=
\sum_m Bubble_m\cdot \hat{ABR}_{t,m}
\cdot UnitProfit_{t,m}
$$

$$
Score_t=GMV_t+\lambda\cdot Profit_t
$$

最后：

$$
t^*(x)=\arg\max_{t\in\{0,\ldots,32\}}Score_t(x)
$$

这里有三个关键点：

1. 应比较**相对基准补贴的增量 Score**，否则容易把自然高需求当成补贴收益；
2. `λ` 是 GMV 与利润的业务兑换系数，不是模型 loss 权重，通常由预算目标、离线 Pareto 曲线和线上实验确定；
3. 还要满足最低/最高折扣、城市预算、用户频控和供需安全约束，不能只做逐单贪心。

出行场景还存在明显干扰效应：给部分乘客补贴会改变区域供需和司机应答，进而影响未接受该 treatment 的其他订单，违反“个体互不影响”的 SUTVA。实验应考虑城市×时段/地理网格分桶、switchback experiment，或至少把局部供需状态纳入分析，不能把普通用户级随机实验结果无条件外推。

### 1.4 项目三：基于 Uplift 的 Bidding 出价优化

传统流量扶持通常按预估 CTR/CVR/GMV 给高价值流量加价，容易把资源继续投给“即使不扶持也会曝光/转化”的流量。

Uplift Bidding 改为估计：

```text
T：是否扶持，或不同扶持金/出价档位
Y：曝光、转化、消耗、GMV或净收益
τ_t(x)：该流量在扶持档位t下相对基准档位的增量响应
```

若 treatment 是多档出价，模型学习响应曲线：

$$
\mu(x,b)=\mathbb{E}[Y(b)\mid X=x]
$$

策略选择：

$$
b^*(x)=\arg\max_b
\left[
Value\big(\mu(x,b)-\mu(x,b_0)\big)
-IncrementalCost(x,b)
\right]
$$

全局预算下继续加入影子价格：

$$
b^*_\lambda(x)=\arg\max_b
\left[
\Delta Value(x,b)-\lambda\Delta Cost(x,b)
\right]
$$

图中“消耗 +0.03%、GMV +0.12%”说明相近资源增量下 GMV 撬动更高，但仅凭这两个指标不能证明因果模型本身无偏，还要确认来自随机流量实验，并观察增量 ROI：

$$
iROI=\frac{\Delta GMV\ 或\ \Delta Profit}{\Delta Cost}
$$

广告场景尤其要注意：

- 历史 bidding 决定了谁能获得曝光，训练数据存在强选择偏差；
- 需要随机出价桶、探索流量或 propensity/DR 校正；
- 如果最终目标是转化，曝光是 treatment 后的中介变量，不能作为发起决策时的普通特征；
- 同一广告主/流量之间可能互相抢量，SUTVA 容易被破坏，需要流量桶或广告主级实验。

### 1.5 现有《价敏模型训练模板》到底实现了什么

文件：`价敏模型训练模板.ipynb`。它不是 TARNet，主体是：

```text
多档酒店折扣 treatment
  ↓
XGBoost S-Learner：把 trt 和用户特征一起输入，预测 is_call
  ↓
对同一用户遍历修改 trt，得到每档折扣的 p_t
  ↓
计算 p_t-p_control、AUUC 和价敏曲线
  ↓
KMeans 将用户压缩成价格敏感人群桶
  ↓
OR-Tools MIP 在补贴上下限约束下分配折扣
```

Notebook 还实验了二元 Uplift Random Forest，但主链路仍是 S-Learner + 运筹。

模板中需要特别警惕：

1. `trt=0` 映射的是“高星 96 折”，不是明确的“不发券组”。若把它叫 control，估计的是“其他券相对 96 折券”的增量，而不是“发券相对不发券”的 uplift；
2. treatment 编码同时混入高星/中星/低星和老客条件，不一定是可互换的 10 档剂量。若各人群没有共同实验覆盖，违反 overlap；
3. 采用随机 train/test split，正式评估应改为时间切分，并保证同一用户不穿越；
4. S-Learner 可能忽略 `trt`，应检查每档 counterfactual 预测是否真的有区分，并与 T-Learner/TARNet 基线比较；
5. Notebook 的 AUUC 是逐个 treatment 与 `0` 做 one-vs-control，不能代表“从全部档位中选最优券”的整体 policy value；
6. 最后的 MIP 思路是对的：Uplift/价敏模型负责产出响应曲线，运筹负责在平均补贴率、库存和预算约束下做全局最优分配。

---

## 二、理论地基：我们真正要估计什么

### 2.1 潜在结果

对用户特征 `X=x`，定义：

```text
Y(0)：不发券时是否会下单
Y(1)：发券包1时是否会下单
...
Y(K)：发券包K时是否会下单
```

同一个用户在同一时刻只能接受一种处理，所以只能观察到：

```text
Y_obs = Y(T)
```

其他结果全部是反事实，这就是因果推断的根本困难。

### 2.2 多干预 CATE

令：

$$
\mu_t(x)=\mathbb{E}[Y(t)\mid X=x]
$$

券包 `t` 相对不发券的条件平均处理效应：

$$
\tau_t(x)=\mu_t(x)-\mu_0(x), \quad t=1,\ldots,K
$$

模型最终为每个用户输出：

```text
[τ_1(x), τ_2(x), ..., τ_K(x)]
```

注意：多干预 Uplift 不是只预测一个 uplift 分，而是要比较多种券包各自相对 control 的增量。

### 2.3 三个识别假设

1. **一致性/SUTVA**：用户实际收到券包 `t` 时，观察结果等于 `Y(t)`；一个用户是否收到券，不能影响另一个用户的结果。
2. **可忽略性**：给定投放前特征 `X` 后，处理分配与潜在结果独立。随机实验天然最接近满足该条件。
3. **重叠性/Positivity**：对目标人群，每种券包都有非零分配概率：

$$
0 < P(T=t\mid X=x) < 1
$$

如果高星酒店券只发给高消费人群，普通用户从未被分到该组，就无法可靠估计普通用户收到高星券的效果。

---

## 三、实验设计：样本质量比模型更重要

### 3.1 优先使用随机多组实验

```text
符合准入条件的用户
  ├─ Control：不发券
  ├─ Treatment 1：券包1
  ├─ Treatment 2：券包2
  ├─ ...
  └─ Treatment K：券包K
```

必须记录：

- `user_id`
- `treatment_id`
- 分组概率 `propensity=P(T=t|X)`
- 分组时间、券到账时间、有效期
- 是否真正触达
- 观察窗口内是否下单、订单金额、利润、退款
- 实验版本和准入规则

如果各组等概率随机，propensity 是常数；如果为了控制预算采用不等比例随机，必须记录每组真实分配概率。

### 3.2 Treatment 是“被分配券”，不是“使用券”

正确口径：

```text
T = 用户是否被系统分配/提供某券包
```

错误口径：

```text
T = 用户最终是否领券或核销
```

是否领券、是否核销发生在 treatment 之后，受用户意愿影响。用核销作为 treatment 会把高意向用户筛进 treatment 组，产生严重选择偏差。

主模型应估计 **ITT（Intention To Treat）**：系统决定发某券包所产生的整体增量。ITT 与线上策略动作完全一致，也最容易通过随机实验识别。

### 3.3 Outcome 怎么定义

CTR/CVR 类目标必须明确观察窗口：

```text
Y_conv = 分组后7天内是否完成有效酒店订单
Y_gmv  = 分组后7天内有效订单GMV
Y_profit = 订单毛利 - 券补贴 - 渠道成本 - 退款损失
```

实践建议：

- 第一版用二分类转化目标，最稳定；
- 再把模型输出转换成增量利润做决策；
- GMV/利润长尾严重时，可用两阶段建模：

$$
\mathbb{E}[Profit(t)\mid x]
=P(Order(t)=1\mid x)\cdot
\mathbb{E}[Profit(t)\mid Order=1,x]
$$

### 3.4 特征必须来自发券之前

可用特征：

- 用户长期价值：历史订单数、间夜数、客单价、会员等级；
- 近期意图：近 1/3/7 天搜索、浏览、收藏、加购；
- 价格敏感度：历史用券率、价格区间、取消率；
- 酒店偏好：城市、高星/低星、连住、商旅/度假；
- 场景：出发时间、节假日、设备、渠道；
- 历史营销疲劳度：过去 30 天收到券次数，但不能包含当前 treatment 后的信息。

禁止使用：

- 当前券是否领取、核销；
- 发券后的访问、点击、搜索；
- 券到账后的实时价格反馈；
- 观察窗口内形成的任何统计特征。

这些都是 post-treatment 特征，会造成因果泄漏。

### 3.5 数据切分

- 按时间切 train/validation/test，模拟真实上线；
- 同一用户跨时间重复出现时，至少保证同一实验周期内不穿越；
- 指标使用 out-of-fold 预测，避免训练集 uplift 虚高；
- 多次触达要定义 washout period，避免上一张券影响下一次实验。

---

## 四、多干预模型怎么选

### 4.1 T-Learner：最稳妥的第一版

每个 treatment 单独训练一个 outcome 模型：

```text
Model_0：只用 control 样本，预测 μ_0(x)
Model_1：只用券包1样本，预测 μ_1(x)
...
Model_K：只用券包K样本，预测 μ_K(x)

τ_t(x)=μ_t(x)-μ_0(x)
```

优点：

- 简单、可解释、容易排查；
- 每个券包允许有完全不同的响应规律；
- XGBoost/LightGBM 就能快速上线。

缺点：

- 每组样本被拆散，稀疏 treatment 容易欠拟合；
- K 个模型维护成本高；
- 两个独立概率相减，误差会叠加。

适用：券包数量少、每组实验量充足、需要快速建立可信基线。

### 4.2 S-Learner：一个模型加入 treatment

```text
μ_t(x)=f(x, one_hot(t))
```

推理时把同一个用户分别和 `t=0...K` 组合，前向 K+1 次。

优点：共享全部样本，维护简单。

缺点：当用户特征远强于 treatment 时，模型容易忽略 treatment；简单树模型尤其容易把 treatment 当成弱特征。

增强方式：

- 加入 `feature × treatment` 显式交叉；
- treatment embedding；
- treatment-specific head；
- 对 treatment 相关参数增加容量。

### 4.3 TARNet / Shared Bottom + 多 Treatment Head：工业常用主结构

```text
用户特征 X
   ↓
Shared Bottom
   ├─ Head_0 → μ_0(x)
   ├─ Head_1 → μ_1(x)
   ├─ ...
   └─ Head_K → μ_K(x)
```

训练时一条样本只监督实际 treatment 对应的 head：

$$
L_y=\sum_i
\operatorname{BCE}(Y_i,\mu_{T_i}(X_i))
$$

这就是 TARNet（Treatment-Agnostic Representation Network）的核心形态：底层表示与 treatment 无关，顶部按 treatment 拆 outcome head。TARNet 本身通常**不直接输出 uplift**，而是先输出每个 treatment 的潜在结果，再在推理时做差：

$$
\hat\tau_t(x)=\hat\mu_t(x)-\hat\mu_0(x)
$$

多 outcome 场景可让每个 treatment head 输出多个任务，loss 只掩码监督实际 treatment：

$$
L=
\sum_i\sum_m
w_m\,
\ell_m\left(
Y_{i,m},
\hat\mu_{T_i,m}(X_i)
\right)
$$

其中 `m` 可以是快车、特惠、出租车 ABR；`w_m` 用于平衡任务尺度和业务重要性，但最终 `GMV+λ×毛利` 的 `λ` 仍属于策略目标，不应与训练 loss 权重混为一谈。

优势：

- 底层共享酒店意图、消费力等公共表征；
- 各 head 学不同券包的异质响应；
- 比 K 个完全独立模型更省样本；
- 比纯 S-Learner 更不容易忽略 treatment。

#### TARNet 的残差门控增强

在 treatment/task 专属层加入：

$$
h_{t,m}=F_{t,m}(h)+g_{t,m}(h)\odot P_{t,m}(h)
$$

可以保留共享表示、改善梯度传播，并让模型控制公共信息进入各专属任务的比例。需要同时防止两个极端：

- gate 长期接近 1：专属网络被 shortcut 掩盖，各 treatment 输出趋同；
- gate 长期接近 0：失去残差通路，退化回普通深层 head。

应监控 gate 分布、各 treatment 输出方差、不同 head 的梯度范数，以及 counterfactual 曲线是否合理。残差门控提升的是拟合和共享机制，**不能替代随机实验、propensity 校正和 overlap 检查**。

若不同券包冲突明显，可以从 Shared Bottom 升级为 MMoE/PLE：

```text
共享 Expert：学习共同的酒店需求
券包专属 Expert：学习不同门槛、面额的特异响应
Treatment Gate：为每个券包组合不同 expert
```

但第一版不建议直接上复杂 MMoE。先证明 shared multi-head 明显优于 T-Learner，再考虑结构升级。

### 4.4 加 Propensity Head：降低非完全随机数据的偏差

在 outcome heads 外增加 treatment propensity head：

```text
Shared Representation
   ├─ Outcome Heads：μ_0...μ_K
   └─ Propensity Head：e_0...e_K
```

$$
e_t(x)=P(T=t\mid X=x)
$$

随机实验中 propensity 已知，不一定需要模型估计；历史策略日志或随机比例随人群变化时，propensity head/外部 propensity 模型很重要。

### 4.5 DR-Learner：更适合最终策略学习

仅预测 `μ_t-μ_0` 容易受到 outcome 模型误差影响。多 treatment 的 Doubly Robust 伪标签：

$$
\phi_t=
\hat\mu_t(X)-\hat\mu_0(X)
+\frac{\mathbb{1}(T=t)}{\hat e_t(X)}
\left(Y-\hat\mu_t(X)\right)
-\frac{\mathbb{1}(T=0)}{\hat e_0(X)}
\left(Y-\hat\mu_0(X)\right)
$$

然后训练第二阶段模型：

$$
\hat\tau_t(x)=\mathbb{E}[\phi_t\mid X=x]
$$

“双重稳健”的含义：outcome 模型或 propensity 模型只要有一个估计正确，效应估计仍有机会保持一致。

工程注意：

- 必须 cross-fitting：生成某条样本的伪标签时，基础模型不能见过该样本；
- propensity 很小时权重会爆炸，要做 clipping；
- DR 方差较大，样本量不足时未必比简单 T-Learner 稳。

### 4.6 连续券力度/组合券包

如果 treatment 只是 3～5 个固定券包，直接按离散多干预建模。

如果面额、门槛、有效期组合非常多，直接为每个组合建 head 会数据稀疏，可把 treatment 表示成：

```text
treatment_features =
[面额, 门槛, 折扣率, 有效期, 适用酒店等级, 是否连住券]
```

再学习：

$$
\mu(x,t)=f(x,\operatorname{Emb}(t))
$$

这属于 treatment representation / dose-response 建模，可以泛化到未充分实验的新组合。但必须保证训练数据对各力度有足够 overlap；模型不能凭空外推从未覆盖过的高额券。

---

## 五、从 Uplift 到真正的券包决策

### 5.1 最大 uplift 不等于最大利润

例子：

| 券包 | 转化 uplift | 单次核销成本 | 结论 |
|---|---:|---:|---|
| 满300减20 | +2.0% | 20元 | 可能高 ROI |
| 满800减100 | +3.0% | 100元 | uplift 更高但可能亏损 |

不能直接选择 `argmax τ_t(x)`，而要计算预期增量价值。

若 `M_t(x)` 表示 treatment 下订单贡献毛利，`C_t(x)` 表示预期券成本：

$$
V_t(x)=
\mathbb{E}[M_t\mid X=x,T=t]
-\mathbb{E}[M_0\mid X=x,T=0]
-\mathbb{E}[C_t\mid X=x]
$$

简化版可写成：

$$
V_t(x)\approx
\tau_t(x)\cdot \overline{Margin}_t(x)
-P(Redeem(t)=1\mid x)\cdot CouponCost_t
$$

决策：

$$
\pi(x)=
\begin{cases}
\arg\max_{t=1,\ldots,K}V_t(x), & \max_t V_t(x)>0\\
0, & \text{否则不发券}
\end{cases}
$$

### 5.2 加预算约束

每日预算为 `B`，可以使用拉格朗日形式：

$$
\pi_\lambda(x)=
\arg\max_t
\left[V_t(x)-\lambda C_t(x)\right]
$$

通过离线二分或在线反馈调整 `λ`，直到：

$$
\sum_x C_{\pi_\lambda(x)}(x)\le B
$$

直觉：预算紧时提高 `λ`，高成本券更难被选中；预算宽松时降低 `λ`，允许覆盖更多正 uplift 用户。

### 5.3 还要满足业务约束

- 用户频控：7 天最多收到 N 次营销券；
- 同一用户券包互斥；
- 券库存和日预算；
- 酒店/城市/渠道准入；
- 高风险、退款、薅羊毛用户屏蔽；
- 最小可用门槛：用户预期客单价明显低于券门槛时不发；
- 探索流量：保留少量随机分配，持续更新因果数据和 propensity。

---

## 六、离线评估：不能只看 AUC

### 6.1 为什么 AUC 不够

Outcome AUC 高，只说明模型能预测谁会下单，不代表它能预测“谁会因为券而改变”。

一个只识别 Sure Thing 的模型可能有很高 CVR AUC，但 uplift 策略会浪费大量补贴。

### 6.2 单券包：AUUC / Qini

对每个券包 `t` 相对 control 单独计算：

1. 按 `τ_t(x)` 从高到低排序；
2. 取 Top p% 人群；
3. 比较该人群 treatment 与 control 的加权转化差；
4. 对累计 uplift 曲线积分得到 AUUC/Qini。

随机实验且等概率时可直接比较；非等概率分配时必须用 propensity 加权。

### 6.3 多券包：Policy Value 才是主指标

模型对每个用户给出策略 `π(x)`。离线评估策略价值可以使用 IPS：

$$
\hat V_{IPS}(\pi)
=\frac{1}{N}
\sum_i
\frac{\mathbb{1}[\pi(X_i)=T_i]Y_i}
{e_{T_i}(X_i)}
$$

SNIPS 做归一化以降低方差：

$$
\hat V_{SNIPS}(\pi)
=
\frac{
\sum_i \frac{\mathbb{1}[\pi(X_i)=T_i]Y_i}{e_{T_i}(X_i)}
}{
\sum_i \frac{\mathbb{1}[\pi(X_i)=T_i]}{e_{T_i}(X_i)}
}
$$

更稳健的 DR Policy Value：

$$
\hat V_{DR}(\pi)=
\frac{1}{N}\sum_i
\left[
\hat\mu_{\pi(X_i)}(X_i)
+\frac{\mathbb{1}[T_i=\pi(X_i)]}{\hat e_{T_i}(X_i)}
\left(Y_i-\hat\mu_{T_i}(X_i)\right)
\right]
$$

利润策略应把 `Y` 换成单样本真实利润/净价值，而不只是转化标签。

### 6.4 离线指标清单

- 每个 treatment vs control 的 AUUC/Qini；
- IPS/SNIPS/DR policy value；
- 增量转化人数；
- 增量利润、iROI；
- 每增量订单补贴成本：

$$
CPIO=\frac{CouponCost}{IncrementalOrders}
$$

- 各券包策略分配比例和预算消耗；
- uplift 分桶校准：预测 uplift 与实验真实 uplift 是否一致；
- overlap/propensity 分布；
- 不同人群、城市、会员等级、价格敏感度的异质效果。

### 6.5 最终一定要在线随机实验

建议至少三组：

```text
A：当前人工/规则策略
B：Uplift 最优券包策略
C：保留随机探索策略或纯 control
```

观察：

- 增量订单、增量间夜；
- 净收入/增量利润；
- 券领取率、核销率；
- CPIO、iROI；
- 自然订单被补贴比例；
- 长期复购、用户疲劳和补贴依赖。

---

## 七、完整迭代路线

### v0：规则投放

按会员等级、历史客单价和近期搜索意图人工选择券包。

问题：规则只能预测“看起来会转化”，无法区分自然转化和因券转化。

### v1：普通 CVR 模型

预测发券后转化率，对高分人群发券。

结果：CVR 高，但大量券发给 Sure Thing，增量 ROI 不高。

### v2：随机多组实验 + T-Learner

建立 control 和多券包随机实验，每组训练一个 LightGBM：

```text
τ_t(x)=P(Y=1|x,T=t)-P(Y=1|x,T=0)
```

价值：第一次从“响应预测”升级成“增量预测”，建立可信基线。

### v3：Shared Bottom + 多 Treatment Head

共享用户酒店需求表征，每种券包一个 head，缓解小 treatment 样本不足，并统一维护。

### v4：DR-Learner + Cross-Fitting

引入 propensity 和 doubly robust 伪标签，降低历史投放不完全随机以及 outcome 模型误差的影响。

### v5：从转化 uplift 升级到增量利润策略

融合毛利、核销概率、券成本和预算约束：

```text
最终策略 = argmax(增量利润 - λ × 预期补贴成本)
```

### v6：持续探索与动态券包

- 线上保留小比例随机探索流量；
- 周期性重估 propensity 和 treatment 效果；
- treatment embedding 支持更多券包组合；
- contextual bandit 在安全约束内动态探索，但不能完全替代因果实验。

---

## 八、常见坑

### 坑 1：把领券/核销当 treatment

核销由用户意愿决定，严重选择偏差。主模型必须用系统分配的券包作为 treatment。

### 坑 2：把发券后特征放进模型

发券后的点击、访问、领券状态属于 post-treatment leakage，会让离线指标虚高且因果含义失效。

### 坑 3：只比较 treatment 组平均转化率

多券包实验组用户结构不同或随机比例不同，直接比较均值可能错误；非等概率时要使用 propensity。

### 坑 4：只优化转化 uplift

高额券可能转化 uplift 最大，但补贴后利润最低。策略层必须扣除成本。

### 坑 5：每种券包单独取 Top 人群再合并

同一个用户可能同时是多个券包的高 uplift 人群，直接合并会冲突。应在用户粒度比较全部 treatment 的净增量价值，再唯一决策。

### 坑 6：用训练集直接算 Qini/AUUC

必须使用时间外测试集或 out-of-fold 预测，否则 uplift 曲线严重乐观。

### 坑 7：忽略 overlap

某类用户从未收到某券包，模型给出的效果只是无依据外推。上线前应限制策略只在实验支持域内决策。

### 坑 8：随机实验停止后完全吃历史策略数据

模型会逐渐继承旧策略偏差，且无法评估新券包。必须长期保留少量探索流量。

### 坑 9：短期订单上涨就认为成功

券可能提前透支未来订单、培养补贴依赖。需要观察长期复购、自然转化率和用户疲劳。

---

## 九、常见问答

**Q1：Uplift 和普通 CVR 模型的本质区别是什么？**

CVR 估计 `P(Y=1|X,T=t)`，回答“发券后谁会下单”；Uplift 估计 `P(Y(t)=1|X)-P(Y(0)=1|X)`，回答“谁会因为发券而额外下单”。前者容易补贴自然转化用户，后者针对可说服人群。

**Q2：为什么一定要有 control？**

没有不发券组就无法估计反事实基线 `Y(0)`，只能知道发券后是否下单，不知道订单是不是券带来的。

**Q3：多 treatment 怎么选择最终券包？**

分别估计每种券包相对 control 的潜在结果与增量价值，在用户粒度计算：

```text
V_t = 增量收益 - 预期券成本
```

选择价值最大的券包；所有 `V_t≤0` 时不发券，再通过拉格朗日乘子满足整体预算。

**Q4：T-Learner、S-Learner 和多头模型怎么选？**

券包少且每组样本充足，用 T-Learner 建可信基线；数据稀疏但 treatment 效应差异不大，可用 S-Learner；工业主模型更推荐 shared bottom + treatment heads，在共享统计强度和保留券包差异之间更平衡。

**Q5：随机实验下还需要 propensity 吗？**

等概率随机时 propensity 已知且简单；不等比例随机、分层随机或历史策略数据中仍需要记录/估计 propensity。离线 policy value 和 DR 估计都依赖它。

**Q6：为什么不直接预测用户会不会核销券？**

核销模型预测的是响应概率，会把本来就爱用券的人排在前面，但无法判断“不发券时他是否也会下单”。它可以用于估计成本，但不能替代 uplift。

**Q7：Uplift 可以解决薅羊毛吗？**

不能自动解决。薅羊毛用户可能表现出很高短期 uplift，但真实净利润为负。需要风控特征、净利润 outcome、退款成本和业务屏蔽规则共同处理。

**Q8：Qini/AUUC 高，为什么线上可能不涨？**

可能原因：离线只评估单券包而线上是多券包冲突策略；propensity 使用错误；存在特征穿越；没有扣券成本；实验人群与上线人群漂移；预算约束改变了目标人群；离线 uplift 未校准。

**Q9：多干预为什么不能把券面额当普通数值直接回归？**

不同券包不只面额不同，门槛、有效期、适用酒店也不同，而且效果未必单调。固定少量券包应先按离散 treatment 建模；只有组合很多且实验覆盖充分时，才考虑 treatment embedding 或 dose-response。

**Q10：怎样向业务证明 Uplift 比 CVR 更好？**

在相同预算下随机对比两种策略，核心看增量利润、增量订单、CPIO 和 iROI，而不是发券人群的表面 CVR。若 Uplift 策略能减少 Sure Thing 补贴，并把预算转移给 Persuadable 人群，就体现了真实价值。

**Q11：搜索量是连续/计数标签，还能使用 AUUC 和 Qini 吗？**

思想上可以，累计曲线中的响应从“转化人数”换成“搜索量总和/均值”即可；但 `scikit-uplift` 等现成实现通常按二元 outcome 设计，应检查实现或自行编写连续 outcome 的累计增量曲线。把搜索量分十档后，也应先还原为期望搜索量/档位价值再排序，不能把类别编号机械当连续真实值。

**Q12：为什么 TARNet 比普通多任务网络更像因果模型？**

TARNet 明确按 treatment 拆潜在结果 head，一条样本只监督其实际 treatment head，推理时对同一个样本得到多个反事实结果并作差。普通多任务网络通常只是同时预测多个可观察 label，并不天然表达 `Y(t)`。但 TARNet 只是结构，若 treatment 分配有偏且未校正，它同样不能自动获得因果性。

**Q13：残差连接为什么有用？门控是不是越复杂越好？**

残差让 treatment/task 专属层学习共享表示上的修正，保留公共信息并改善梯度传播；门控控制公共与专属信息的比例。结构变复杂不代表因果估计更准，必须监控 gate 是否饱和、各 treatment 输出是否坍缩，并通过时间外 AUUC、policy value 和线上随机实验验证。

**Q14：33 个补贴对为什么不能只看每个补贴下预测 ABR 最大的那个？**

ABR 最大不等于全局价值最大：补贴可能在快车、特惠、出租车间搬量，也可能提高 GMV 却损害毛利。必须把三个业务的预估 ABR 转成订单、GMV、毛利和补贴成本，再以相对基准 treatment 的增量价值排序，并满足预算和供需约束。

**Q15：模板里的 `trt=0` 可以直接当 control 吗？**

不可以仅凭编码为 0 就认定是 control。模板中 `trt=0` 实际映射“高星 96 折”，因此它只是一个基准券档。若实验没有真正的不发券组，模型只能估计其他券相对该券档的增量，不能回答“发券相对不发券是否值得”。

---

## 十、项目表达模板

### 10.1 酒店券包

> 酒店券包原来采用规则或普通 CVR 模型投放，模型倾向把券发给本来就会下单的高意向用户，存在严重自然订单补贴问题。我们设计了不发券和多档券包的随机实验，以系统分配券包作为 treatment、7 日有效订单作为 outcome，使用发券前的用户价值、近期酒店意图和价格敏感度特征，先通过 T-Learner 建立多干预基线，再升级为 shared bottom + treatment-specific heads，同时使用 propensity 与 DR cross-fitting 提高策略估计稳健性。决策层没有直接选择转化 uplift 最大的券，而是将 CATE 转换为增量利润，扣除核销成本，并通过拉格朗日乘子满足日预算和券库存约束。离线使用各券包 AUUC/Qini 和 DR policy value，线上通过随机 A/B 评估增量订单、增量利润、CPIO 与 iROI，最终实现从“预测谁会下单”到“判断给谁发哪种券才真正产生增量”的升级。

### 10.2 百度搜索悬浮球

> 项目目标不是预测高活跃搜索用户，而是识别展示悬浮球后 7 日搜索量会真正提升的人群。我们以是否展示悬浮球为二元 treatment，以展示前的用户画像和历史搜索活跃度为特征，以未来 7 日搜索量为 outcome，通过 XGBoost 估计 treatment/control 下的潜在响应并作差。针对搜索量长尾导致的过拟合，将标签按训练集分位数划为 10 档，用每档搜索量均值还原期望响应，同时将 booster 改为 DART 降低树间依赖。离线通过时间外 AUUC、Qini 和 uplift 分桶图验证排序与校准，线上圈选搜索增量最大的人群进行实验。

### 10.3 快车/特惠/出租车 ABR 多任务定价

> 项目面对 33 个快车与特惠补贴组合，需要同时评估其对快车、特惠和出租车 ABR 的影响。我们采用多 treatment、多 outcome TARNet：共享层学习用户、订单和供需公共表示，每个补贴对通过专属 head 输出三个业务的潜在 ABR；训练时只监督样本实际接受的 treatment。为减少专属层丢失公共信息并改善优化，在共享层到 treatment/task 层之间加入投影残差，并通过门控控制 shortcut 信息量。推理时对每单遍历 33 个补贴对，将三个业务的 ABR 转成预期 GMV、毛利与补贴成本，按相对基准方案的 `ΔGMV+λ×Δ毛利` 排序，并结合预算和供需约束选择最终补贴对。离线除 one-vs-control Qini 外，重点评估多 treatment 的 DR policy value，线上通过随机策略实验验证。

### 10.4 Bidding 流量扶持

> 传统扶持策略偏向把预算投给本身曝光/转化概率高的流量，无法区分自然响应与扶持增量。项目把是否扶持或扶持金档位作为 treatment，估计每个用户/流量在不同出价下的曝光、转化和 GMV增量，再扣除增量消耗形成边际价值。策略层通过拉格朗日影子价格在全局预算下选择最优出价，把资源集中到真正可被撬动的人群。历史 bidding 数据存在强选择偏差，因此训练和评估必须依赖随机出价桶/探索流量，并配合 propensity 或 DR 校正；最终以增量 GMV、增量消耗和 iROI 验证，而不是只看扶持后绝对转化率。

---

## 十一、一句话总结

> 多干预 Uplift 的核心不是训练 K 个转化模型，而是用可信实验识别每种券包相对不发券的反事实增量，再把增量转化换算成扣除补贴后的增量利润，最终在预算和业务约束下做唯一 treatment 决策。
