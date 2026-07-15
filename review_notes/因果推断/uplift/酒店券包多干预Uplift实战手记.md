# 酒店券包多干预 Uplift · 实战手记

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

### 4.3 Shared Bottom + 多 Treatment Head：工业常用主结构

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

优势：

- 底层共享酒店意图、消费力等公共表征；
- 各 head 学不同券包的异质响应；
- 比 K 个完全独立模型更省样本；
- 比纯 S-Learner 更不容易忽略 treatment。

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

---

## 十、项目表达模板

> 酒店券包原来采用规则或普通 CVR 模型投放，模型倾向把券发给本来就会下单的高意向用户，存在严重自然订单补贴问题。我们设计了不发券和多档券包的随机实验，以系统分配券包作为 treatment、7 日有效订单作为 outcome，使用发券前的用户价值、近期酒店意图和价格敏感度特征，先通过 T-Learner 建立多干预基线，再升级为 shared bottom + treatment-specific heads，同时使用 propensity 与 DR cross-fitting 提高策略估计稳健性。决策层没有直接选择转化 uplift 最大的券，而是将 CATE 转换为增量利润，扣除核销成本，并通过拉格朗日乘子满足日预算和券库存约束。离线使用各券包 AUUC/Qini 和 DR policy value，线上通过随机 A/B 评估增量订单、增量利润、CPIO 与 iROI，最终实现从“预测谁会下单”到“判断给谁发哪种券才真正产生增量”的升级。

---

## 十一、一句话总结

> 多干预 Uplift 的核心不是训练 K 个转化模型，而是用可信实验识别每种券包相对不发券的反事实增量，再把增量转化换算成扣除补贴后的增量利润，最终在预算和业务约束下做唯一 treatment 决策。
