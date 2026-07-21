# 多干预 Uplift 与约束决策 · 实践

> 本文用六个场景统一说明 Uplift 的工程落地：酒店券包与红包、旅行多品类券位 Top-2、搜索悬浮球、飞猪搜索广告、33 个补贴组合 ABR、Bidding 流量扶持。每个项目均按**样本、特征、模型、评估、决策**展开；第十二章汇总高频问答。

---

## 一、先明确：Uplift 到底解决什么问题

普通响应模型回答：

> 用户接受某个动作后，会不会下单、搜索或转化？

Uplift 模型回答：

> 与不采取动作相比，这个动作会让该用户额外增加多少订单、搜索量或业务价值？

以酒店券为例，普通 CVR 模型容易把券发给本来就会下单的高意向用户；Uplift 要识别“因为券才会下单”的用户，避免补贴自然订单。

### 1.1 潜在结果与 CATE

设：

- `X`：干预前特征；
- `T in {0,1,...,K}`：实际分配的 treatment；
- `Y(t)`：接受 treatment `t` 时的潜在结果；
- `t=0`：基准动作，通常是不干预。

潜在结果的条件期望：

```text
mu_t(x) = E[Y(t) | X=x]
```

treatment `t` 相对基准动作的条件平均处理效应：

```text
tau_t(x) = mu_t(x) - mu_0(x),  t=1,...,K
```

同一用户在同一时刻只能接受一个 treatment，因此只能观察：

```text
Y_i^obs = Y_i(T_i)
```

其他 `Y_i(t)` 都是反事实。模型做差发生在**预测阶段**：对同一用户预测多个潜在结果，再计算相对基准动作的差；不是要求一条训练样本同时拥有多个 label。

### 1.2 ATE、ATT、CATE 分别回答什么

| 符号 | 定义 | 工程含义 |
| ---- | ---- | -------- |
| ATE | `E[Y(1)-Y(0)]` | 对目标人群平均，处理相对不处理好多少 |
| ATT | `E[Y(1)-Y(0) \| T=1]` | **只对被实际处理的人**，处理值不值 |
| CATE | `E[Y(t)-Y(0) \| X=x]` | 给定特征 `x` 的个性化增量，Uplift 主目标 |

飞猪 Matching 估的是 ATT：解释「当前已投广告流量」的增量，不能无条件外推到从未投放的人群。个性化发券/选档主要估 CATE；全量开关实验常报 ATE。

主分析多用 ITT（Intention To Treat）：按系统分配的 treatment 估效应，而不是按领券、核销等后验行为。

### 1.3 四类用户

| 人群           | 不干预 | 干预  | 策略     |
| ------------ | --- | --- | ------ |
| Persuadable  | 不响应 | 响应  | 核心干预对象 |
| Sure Thing   | 响应  | 响应  | 避免浪费成本 |
| Lost Cause   | 不响应 | 不响应 | 无需干预   |
| Sleeping Dog | 响应  | 不响应 | 必须避免干预 |

四类人是概念解释。单个用户的两个潜在结果无法同时观察，不能直接给每个用户打真实四分类标签。

### 1.4 因果识别的三个条件

1. **一致性与 SUTVA**：用户实际接受 `t` 时，观察结果等于 `Y(t)`；一个用户的 treatment 不影响其他用户。
2. **可忽略性**：给定全部干预前混杂因素后，treatment 分配与潜在结果独立。
3. **重叠性**：目标人群中的每种 treatment 都有被分配的可能：

```text
0 < e_t(x) = P(T=t | X=x) < 1,  for all t
```

随机实验主要解决 treatment 分配偏差，但广告竞价、出行供需等场景还可能存在用户间干扰，不能只靠普通用户级随机分桶。

---

## 二、六个场景总览

| 项目           | Treatment  | Outcome              | 核心偏差                       | 推荐起点                  | 最终动作         |
| ------------ | ---------- | -------------------- | -------------------------- | --------------------- | ------------ |
| 酒店券包与红包      | 不发券 + 多档券  | 订单、GMV、贡献毛利          | 给高意向用户发券，补贴自然订单            | 多组 RCT + T-Learner    | 每人选择唯一券档或不发  |
| 旅行多品类券位 Top-2 | 酒店/机票/火车多档，展示 2 张 | 平台增量 GMV、贡献毛利     | 场景偏置、跨品类搬量、组合稀疏           | 单券 uplift + 组合决策      | 约束下选 2 张券   |
| 搜索悬浮球        | 展示/不展示     | 未来 7 日搜索量            | 高活跃用户更容易被展示                | 用户级 RCT + XGBoost     | 圈选真正增加搜索的人群  |
| 飞猪搜索广告       | 投广告/不投广告   | 平台整体 GMV、营收          | 广告由竞价和规则选择                 | Matching + Effect-Net | 广告准入、流量与预算分配 |
| 33 个补贴组合 ABR | 33 个补贴组合   | 三业务应答率（ABR）、GMV、贡献毛利 | 多 treatment、多 outcome、供需干扰 | TARNet 多头结构           | 约束下选择补贴组合    |
| Bidding 流量扶持 | 不扶持 + 多档出价 | 曝光、转化、GMV、消耗         | 只有胜出样本可见，选择偏差强             | 随机出价桶 + DR            | 预算内选择最优出价    |

共同链路：

```text
实验或去偏样本
  → pre-treatment features
  → 估计 μ_t(x) 或 τ_t(x)
  → 转换成增量 GMV / 毛利 / 成本
  → 预算和业务约束下选 treatment（多数不在网络内端到端求解）
  → 在线随机实验验证
```

---

## 三、通用数据基础：样本质量先于模型

### 3.1 Treatment 是系统动作，不是用户响应

正确：

```text
发券：系统是否分配某张券 / 展示哪个券组合
悬浮球：系统是否展示
广告：系统是否投放广告
Bidding：系统采用哪个出价档
```

错误：

```text
是否领券、核销、点击、购买
```

领取、点击、核销均发生在 treatment 之后。把这些行为当 treatment，会将本来就高意向的用户筛进处理组，产生选择偏差。主分析通常估计 ITT，即系统分配动作带来的整体增量。

### 3.2 优先保留随机探索流量

推荐记录：

- 用户/请求 ID；
- treatment ID（单券或展示组合）；
- 真实分流概率 `e_t(x)`；
- 实验版本、准入条件、分组时间；
- 触达状态；
- 统一观察窗口内的 outcome、GMV、毛利和成本；
- 退款、取消及长期结果。

分层随机、不等比例随机同样可信，但必须保存每条样本的真实分配概率。

### 3.3 非随机 / 场景偏置数据：国内工程怎么做

只有在“给定 `X` 后没有遗漏混杂因素”时，观察数据去偏才有因果解释。国内营销、发券、补贴团队**很少**「纯观察数据硬估因果后直接全量上线」，常见分层：

| 档位 | 常见做法 | 用途 |
| ---- | -------- | ---- |
| 保底 | 长期挂 1%–10% 随机探索桶，日志写死 `e_t(x)` | 离线评估与线上验收的真值锚点 |
| 常用 | 规则/模型大流量日志 + Matching / IPW / DR 去偏 | 训模型、筛策略、看量级 |
| 慎用 | 把 PSM 分数或 propensity 当线上发券打分 | 仅作对齐与 ATT，不当主策略分 |

标准流水线：

1. 还原当时 treatment 规则（准入、分档、白名单、场景限制），能复现概率则用日志概率；
2. 否则训多分类 propensity `e_t(x)`，特征必须覆盖规则用到的场景/意图变量；
3. 查 overlap；对极端 `e_t` 做 trim/clip（如 `[0.05, 0.95]`）；
4. 训练侧：T-Learner / TARNet，可选 DR 伪标签；
5. 评估侧：IPS / SNIPS / DR Policy Value，并**分场景**报指标；
6. 独立评估集重复平衡检查；
7. 同预算线上 A/B；探索桶长期保留防漂移。

场景偏置（搜酒店更容易拿到酒店券）不要假装随机：把 query/品类/行程阶段进特征与 propensity；只在「该场景可发」的人群里估效应；评估按场景分层。

Propensity 主要用于训练去偏、评估和 overlap 检查，通常不是线上业务模型的最终输出。

### 3.4 特征边界

只允许使用决策时已经存在的特征：

- 用户长期价值和生命周期；
- 历史行为、偏好、价格敏感度；
- 当前请求、场景、意图、时间和渠道；
- 历史营销疲劳；
- treatment 分配规则中的所有变量。

禁止使用：

- 当前券的领取、核销；
- 广告或悬浮球点击；
- 干预后的搜索、曝光和转化；
- treatment 改变后形成的价格、排序或页面统计。

### 3.5 数据切分

- 优先按时间切 train/validation/test；
- 同一实验周期内同一用户不得跨集合；
- 伪标签和离线指标使用 out-of-fold 预测；
- 多次触达设置 washout period；
- 匹配、propensity 和标准化器只在训练数据拟合，再应用于验证/测试。

### 3.6 PSM：怎么建、样本怎么构、算完怎么用

这里的 PSM 指 **Propensity Score Matching**（倾向分匹配），不是把 propensity 直接当 uplift。

**建模**

```text
样本：(X, T)，T ∈ {0,1,...,K}
模型：多分类（或一对多）e_t(X) = P(T=t | X)
特征：仅 treatment 前特征，含当时决策规则特征
```

常用 LightGBM / LR；若日志有真实分流概率，优先用日志概率。

**样本构建**

- 一行一次决策（用户 × 请求 × 发券/投放时刻）；
- `T` = 系统实际动作，不用领券、点击；
- 多 treatment：对每个 `t`，从 `T=t` 抽 treated，在 control（或对照组）中按倾向分相近匹配；
- 设置 caliper，丢弃距离过远样本；Top-K 近邻等权或距离加权。

飞猪「Original Space Matching」是在原始特征空间（自然队列 pCTR 向量）匹配，思想同类：对齐可比样本再比 outcome。

**匹配后怎么用**

| 用途 | 做法 |
| ---- | ---- |
| 估 ATT | `Y_treated - 加权匹配 control 的 Y`（见第六章） |
| 训前清洗 | 只保留匹配成功、SMD 合格子集，再训 T/TARNet |
| 诊断 | 匹配前后 SMD、丢弃率、control 复用次数、有效样本量 |
| 一般不做 | 把 propensity / 匹配分当线上发券主打分 |

与 IPW 区别：PSM 是「找相似人对齐再比」；IPW 是「按 `1/e_t` 加权」。发券策略更常 IPW/DR 评估 + 小随机桶；Matching 在二值投放、要解释 ATT 时更常见。

---



## 四、项目一：酒店券包与红包



### 4.1 业务定义

```text
T=0：不发券
T=1：满300减20
T=2：满500减50
T=3：满800减100
...
Y：7日内有效订单、GMV或贡献毛利
```

目标不是选择转化 uplift 最大的券，而是在预算、库存、频控和 ROI 约束下，选择增量净价值最大的券；若全部为负则不发。

### 4.2 样本

最优方案是多组随机实验：

```text
符合准入条件的用户
  ├─ Control：不发券
  ├─ Treatment 1：券包1
  ├─ Treatment 2：券包2
  └─ Treatment K：券包K
```

注意：

- `T=0` 必须是真实不发券，不能只是编码为 0 的某档折扣；
- 同一用户实验期固定 treatment；
- 各组观察窗口一致；
- 新券档必须先有探索流量，不能依赖模型外推；
- 历史运营数据若非随机，使用多分类 propensity、DR 或多 treatment matching。



### 4.3 特征

用户：

- 历史订单、间夜、客单价、贡献毛利；
- 近 1/3/7/30 日搜索、浏览、收藏、加购；
- 历史折扣、用券率和门槛偏好；
- 酒店星级、城市、商旅/度假偏好；
- 会员等级、生命周期、渠道；
- 历史发券次数和营销疲劳。

Treatment 属性：

```text
[面额, 门槛, 折扣率, 有效期,
 适用星级, 城市限制, 新老客限制]
```



### 4.4 模型

第一版使用 T-Learner：每个券档训练一个 LightGBM/XGBoost，建立清晰可信的基线。

样本量不足或券档较多时，升级为 Shared Bottom + Treatment Heads：

```text
用户特征 X
   ↓
Shared Bottom
   ├─ Head 0 → μ₀(x)
   ├─ Head 1 → μ₁(x)
   └─ Head K → μ_K(x)
```

**训练与预测不要混：**

- 训练：一条样本只有一个 factual label。若 `T_i=2`，只对 Head 2 算 loss 并回传；Head 0/1/3… **本步不算 loss**，不是废头。
- 其他头靠「历史上拿到过该 treatment 的样本」更新，学的是反事实潜在结果 `μ_t(x)`。
- 预测：同一用户必须过全部 head（或一次前向多 head），得到 `μ_0…μ_K`，再做差选券。

```text
L_outcome = (1/N) * sum_{i=1..N} loss(Y_i, mu_hat_{T_i}(X_i))
```

```text
tau_hat_t(x) = mu_hat_t(x) - mu_hat_0(x)
```

金额类 outcome（GMV、贡献毛利）零膨胀且长尾时，可用 ZILN / Hurdle，见 9.6 节。非随机数据可使用 DR-Learner，见第九章。

### 4.5 评估

- 每个券档相对不发券的 AUUC/Qini；
- uplift 分桶校准；
- 多券策略的 IPS/SNIPS/DR Policy Value；
- 券档分配比例、预算消耗和 overlap；
- 增量订单、每增量订单成本（CPIO）、增量 GMV、增量贡献毛利；
- 线上同预算 A/B 验证。

### 4.6 决策

对每个用户计算各券档的增量净价值 `V_{i,t}`，而不是只比较转化 uplift：

```text
t_i* = argmax_{t in {0,...,K}} V_{i,t}
```

将不发券定义为 `V_{i,0}=0`。预算充足时只发放 `V_{i,t}>0` 的最优券；预算受限时再使用第十一章的拉格朗日影子价格、ROI、库存和频控约束。**预算/补贴率一般不在 uplift 网络内端到端求解，而在决策层处理。**

---

## 四（续）、旅行多品类券位：9 券选 2

### 4.7 业务定义

同一曝光位候选：酒店券 3 档、机票券 3 档、火车票券 3 档，最终只展示 2 张。

```text
原子券：c ∈ {H1,H2,H3, F1,F2,F3, T1,T2,T3}
展示动作：无序对 {a,b}（C(9,2)=36）或有序对（考虑左右位）
Y：统一窗口平台增量订单 / GMV / 贡献毛利（不是单业务核销）
```

Treatment 是系统展示的组合（ITT），不是用户点了哪张、领了哪张。

### 4.8 合在一起还是分业务

| 做法 | 优点 | 问题 | 建议 |
| ---- | ---- | ---- | ---- |
| 9 券扁平多分类 | 实现简单 | 样本稀、overlap 差，忽略同屏交互 | 仅作探索或只发 1 张时 |
| 三业务各做 uplift 再拼 top | 品类内单调好做 | 看不见跨品类搬量；各自 top1 ≠ 全局最优 | 业务强隔离考核时 |
| **品类内共享 + 跨品类组合决策** | 可估、可贴业务 | 需轻量交互与约束 | **首选** |

推荐架构：

```text
共享用户/场景表征
  ├─ 酒店 3 档 head（品类内可加面额单调约束）
  ├─ 机票 3 档 head
  └─ 火车 3 档 head
       ↓
单券增量净价值 V(x,c)
       ↓
组合层：V(x,{a,b}) ≈ V(x,a)+V(x,b)+Interact(x,a,b)-Cost
       ↓
约束下选 Top-2（品类配额、同品类至多 1 张、预算、疲劳、位次）
```

不要一上来把 36/72 个组合当独立 treatment 硬训（非随机 + 稀疏会很脆）；长尾组合用品类级交互或成分共享外推，并保留随机 2 券探索流量。

### 4.9 即使有随机，旅行场景仍要注意

| 问题 | 表现 | 处理 |
| ---- | ---- | ---- |
| 意图强异质 | 搜酒店却随机到机票券，效应≈0 | 场景分层随机/准入；特征加意图；分场景评估 |
| 跨品类搬量 | 火车涨、机票掉，单业务好看大盘没有 | 平台级 Y 或多 outcome 再合成净价值 |
| 触发成本 | 高意向核销多，补贴自然单 | 成本用期望核销；看增量毛利与 CPIO |
| 取消退款改签 | 短期订单虚高 | 拉长窗口；Y 用有效/结算后结果 |
| 价与供需时变 | 同人不同天基线差大 | 实时价库存特征；时间切分；强供需可用 switchback |
| 同屏位次偏差 | 左位 CTR 高被当成券更好 | 位次随机或模型区分位次与券效应 |
| 多触点干扰 | 同时有 push/红包/广告 | 实验隔离或记并发；坚持 ITT |
| 库存与 SUTVA | 热门房/舱位有限 | 资源维度实验；策略加库存约束 |
| 疲劳重复触达 | 随机组仍被其他通道发券 | washout、频控特征、用户级固定桶 |

### 4.10 决策要点

```text
{a*,b*} = argmax_{a≠b} [V(x,{a,b}) - λ * ΔC(x,{a,b})]
```

硬过滤：补贴率上限、同品类不重复、场景准入。评估看大盘增量与 DR Policy Value，不要只看单券核销或单业务 AUUC。

---

## 五、项目二：搜索悬浮球活跃度 Uplift



### 5.1 业务定义

```text
T：是否展示悬浮球
Y：分组后7日搜索量
目标：找到展示后会额外增加搜索的人群
```

普通搜索量模型会偏向本来就活跃的人；Uplift 关注展示带来的增量搜索。

### 5.2 样本

优先做用户级固定随机分桶。若历史规则按活跃度决定是否展示，则“历史活跃度”同时影响 treatment 和未来搜索量，必须：

- 回溯完整展示规则；
- 估计 propensity；
- 在历史活跃度、生命周期、渠道等变量上做 IPW/DR 或 Matching；
- 删除没有 overlap 的人群；
- 最终补做随机实验。



### 5.3 特征

- 历史 1/3/7/30 日搜索次数和活跃天数；
- Query 类目、多样性、时段分布；
- 生命周期、会员、设备和渠道；
- 历史活动入口曝光、点击和疲劳度；
- 展示前页面状态。

不能使用本次悬浮球点击和展示后的搜索。

### 5.4 模型

稳妥基线是 XGBoost T-Learner。

7 日搜索量通常零膨胀且长尾，可使用 Hurdle：

```text
mu_t(x) = P(Y(t)>0 | X=x) * E[Y(t) | Y(t)>0, X=x]
```

若将搜索量分成 10 档，**不能拿档序号 0–9 当回归标签完事**，必须用每档真实搜索量代表值还原条件期望：

1. 定边界：如 `(-\infty,0], (0,1], (1,3], …`（业务规则或分位数）；
2. 每档存代表值 `y_bar_b`：该档训练样本真实均值（或中位数）；
3. 模型出 `P(B=b|x,t)`（分类）或先预测档再 softmax；
4. 还原：

```text
mu_hat_t(x) = sum_{b=1..10} P(B(t)=b | X=x) * y_bar_b
```

5. uplift 用还原后的连续期望做差：`mu_hat_1(x) - mu_hat_0(x)`，**不是档编号相减**。

更稳的是 Hurdle（上式零膨胀分解），少一次分档失真。其他可比较方案：`log1p` 回归、Poisson/Tweedie、Huber、winsorize、ZILN（金额类）。

DART 只能缓解树模型过拟合，不能修复 treatment 选择偏差。

### 5.5 评估

- 在独立实验样本上按预测 uplift 分桶；
- 对连续 outcome 计算加权累计增量搜索量；
- 使用 AUUC、分桶校准和 Bootstrap 置信区间；
- 线上验证增量搜索量、搜索 DAU、负反馈和长期活跃。



### 5.6 决策

设一次展示的业务成本为 `c(x)`，每次增量搜索的价值为 `v_s`：

```text
V_show(x) = v_s * tau_hat(x) - c(x)
```

仅对 `V_show(x)>0` 的用户展示，再叠加频控、覆盖人数和用户体验约束。若展示成本可忽略，则按 uplift 排序并由流量配额截断。

---



## 六、项目三：飞猪搜索广告因果效应

参考：[因果推断在阿里飞猪广告算法中的实践](https://zhuanlan.zhihu.com/p/410582804)

### 6.1 业务定义

```text
T：搜索结果页是否投放广告
Y：平台整体GMV、平台营收或转化
```

广告 pCTR/pCVR 高，不代表广告带来高增量。广告与自然结果分布相似，投放广告还可能挤占自然交易，因此必须比较“投广告”和“不投广告”的平台整体结果。

口径要分开：

```text
整体GMV = 广告商品GMV + 自然商品GMV
平台营收 = 广告CPC消耗 + 自然交易抽佣（按原文口径）
```



### 6.2 样本

样本单位是一次搜索请求/结果页，treatment 为该页面是否投放广告。应长期保留小比例随机投放/停投流量作为评估基准；全量随机停投成本高时，飞猪采用 Original Space Matching：

1. 样本单位为一次搜索请求/结果页；
2. 使用广告插入前自然结果队列的 pCTR 分布构造匹配向量；
3. 对无广告 control 样本建立向量索引；
4. 为每个有广告 treatment 样本检索 Top-K 相似 control；
5. 设置 caliper，丢弃距离过远的样本；
6. 对 K 个 control 等权或按距离加权。

匹配后的 **ATT** 估计（Average Treatment Effect on the Treated）：

```text
ATT_hat = (1/N_T) * sum_{i:T_i=1} [Y_i - sum_{j in N_K(i)} w_ij * Y_j]
```

其中：

```text
sum_{j in N_K(i)} w_ij = 1
```

ATT 只回答：**在这些已经投放广告的请求上**，投广告相对不投平均好多少。用于准入/停投解释与当前策略诊断；不能直接当作「从未投过广告的人」的效应，全量策略仍需 ATE、Policy Value 或随机实验。

必须检查匹配前后的 SMD、匹配距离、丢弃率、control 复用次数和有效样本量。Matching 只能平衡已观察变量，无法修复未记录的竞价和人工规则。

### 6.3 特征

- 用户画像、活跃度、消费力；
- Query、行业、城市、设备和时段；
- 广告插入前的 Search Rank Queue；
- 自然 topN 商品的 pCTR/pCVR、价格和相关性分布；
- 广告候选与自然 Top 商品的预估差值；
- 预算、pacing、竞价环境和投放资格。

原生搜索与广告样本分布不同，可采用多任务迁移，并通过 in-batch re-sampling 调整两类样本比例。但重采样用于泛化权衡，不等于因果去偏。

### 6.4 模型



#### S-Learner Baseline

```text
mu_hat(x,t) = f(x,t),  tau_hat(x) = f(x,1) - f(x,0)
```

原文实验中，普通预估模型 AUC 尚可但 Qini 为负；增加 DNN 后 AUC 提升，Uplift 反而更差。这说明 outcome 预测能力不等于效应估计能力。

#### Shortcut

将 treatment effect 分支短接到网络后部，防止单个 treatment 变量被深层网络忽略。

#### 双头模型

```text
Shared Representation
  ├─ Control Head → μ₀(x)
  └─ AD Head      → μ₁(x)
```

这与 TARNet 结构相似。训练只监督 factual head，预测时计算 `mu_hat_1-mu_hat_0`。双头结构不能替代样本去偏，在原始有偏样本上可能直接学习两组分布差异。

#### Effect-Net

若 outcome 是二元变量，不应直接在概率上无约束相加。更稳妥的 logit 残差写法：

```text
mu_hat_0(x) = sigmoid(z_0(x))
```

```text
mu_hat_1(x) = sigmoid(z_0(x) + delta(x))
```

```text
tau_hat(x) = mu_hat_1(x) - mu_hat_0(x)
```

`delta(x)` 是广告在 logit 空间的增量，两个潜在结果始终位于 `[0,1]`。若业务原模型在概率空间做加法，则必须显式 clip 或使用合法链接函数。

### 6.5 评估

- 独立随机或重新匹配的测试集；
- uplift 分桶、AUUC/Qini 和置信区间；
- 平台整体 GMV、平台营收、自然交易损失；
- 广告主 ROI、CTR/CVR 等护栏；
- 在线按相同流量和预算比较旧策略与因果策略。



### 6.6 决策

将广告投放带来的平台增量价值记为 `Delta R(x)`，广告挤占自然交易和体验损失记为 `Delta L(x)`：

```text
V_ad(x) = DeltaR(x) - DeltaLoss(x)
```

可将 `V_ad(x)` 作为广告准入、参竞概率或预算分配的 reward，并把广告主 ROI、平台收入下限和自然流量损失作为 constraints。

---



## 七、项目四：33 个补贴组合与三业务 ABR



### 7.1 业务定义

```text
T∈{0,...,32}：快车/特惠补贴组合
Y_fast：快车应答率（ABR）
Y_bargain：特惠应答率（ABR）
Y_taxi：出租车应答率（ABR）
```

必须同时预测三个业务，因为补贴可能只造成快车、特惠、出租车之间的搬量，而没有创造大盘增量。

### 7.2 样本

- 尽量随机分配 33 个补贴组合；
- 记录每个组合的真实分流概率和可选集合；
- treatment 稀疏时保证每组最小样本量；
- 出行供需存在用户间干扰，优先采用城市×时段、地理网格或 switchback 实验；
- 普通用户级随机实验结果不能无条件外推到供需状态改变后的全量策略。



### 7.3 特征

- 乘客画像、历史呼叫和价格敏感度；
- 起终点、距离、预估时长；
- 城市、时段、天气和供需状态；
- 三业务预估冒泡量、价格和司机供给；
- treatment 前的补贴历史与疲劳度；
- 补贴组合属性。



### 7.4 模型

使用多 treatment、多 outcome TARNet（**可以同时多 treatment 与多 outcome**）：

```text
Shared Representation
  ├─ Treatment 0 Head → [ABR_fast, ABR_bargain, ABR_taxi]
  ├─ ...
  └─ Treatment 32 Head → [ABR_fast, ABR_bargain, ABR_taxi]
```

多 outcome factual loss（只监督实际 treatment 对应的各 outcome 头）：

```text
L = (1/N) * sum_{i=1..N} sum_m omega_m * loss_m(Y_{i,m}, mu_hat_{T_i,m}(X_i))
```

- `omega_m`：业务重要性权重；
- ABR 常用 BCE/MSE；GMV/金额可用 MSE、Hurdle 或 ZILN；
- 预测时对每个 `(t,m)` 都出 `μ̂_{t,m}`，再拼 GMV/毛利做决策。

残差门控可写为：

```text
h_{t,m} = g_{t,m}(h) .* F_{t,m}(h) + (1-g_{t,m}(h)) .* P_{t,m}(h)
```

其中 `P_t,m` 在维度不同时必须做投影。门控增强的是共享与拟合，不是因果识别来源。

**这里的「约束」一般不写进 TARNet 的 loss。** 网络只学潜在结果；预算、供需、折扣在决策层用 `Score_t` + 拉格朗日/硬过滤求解（见 7.6 与第十一章）。若对补贴面额做「越大响应不降」的**单调约束**，那是对 head 输出加软约束/排序损失，与预算约束是两回事。

### 7.5 评估

- 每个补贴组合相对基准组合的 outcome 校准；
- one-vs-control AUUC 仅作局部指标；
- 多 treatment 的 DR Policy Value；
- 三业务搬量、大盘总单量、GMV 和贡献毛利；
- 城市、时段和供需状态分层评估；
- switchback 在线实验。

### 7.6 决策

对每个 treatment 计算：

```text
GMV_hat_t(x) = sum_m B_hat_{t,m}(x) * mu_hat_{t,m}(x) * Price_{t,m}(x)
```

```text
M_hat_t(x) = sum_m B_hat_{t,m}(x) * mu_hat_{t,m}(x) * UnitMargin_{t,m}(x)
```

其中 `B_hat_{t,m}(x)` 是 treatment `t` 下业务 `m` 的预测冒泡量。若补贴会改变业务间流量，就不能把冒泡量固定为与 treatment 无关的 `B_m(x)`。

设 `DeltaC_t(x)` 为补贴组合相对基准组合的增量成本，则增量净价值为：

```text
V_t(x) = [M_hat_t(x) - M_hat_0(x)] - DeltaC_t(x)
```

若业务同时考核 GMV 与净价值，用 `alpha` 表示两者的兑换系数，避免与预算影子价格 `lambda` 混用：

```text
Score_t(x) = [GMV_hat_t(x) - GMV_hat_0(x)] + alpha * V_t(x)
```

最后在预算、供需安全和折扣约束下最大化 `Score_t(x)`（用户级 `argmax` + 调 `λ`，或离线 LP），不能只按逐单最高 ABR 贪心。

---



## 八、项目五：Bidding 流量扶持



### 8.1 业务定义

```text
T：不扶持或不同扶持金/出价档
Y：曝光、转化、GMV、平台收入或消耗
```

普通高 pCTR/pCVR 流量不一定需要扶持；Uplift Bidding 要寻找“加价后才会胜出或转化”的边际流量。

### 8.2 样本

历史出价决定谁能获得曝光，只有胜出样本可见完整后链路，选择偏差极强。推荐：

- 建立随机出价桶或小比例探索流量；
- 记录每个候选出价和选择概率；
- 区分请求级、广告主级和流量桶级 treatment；
- 用 propensity/DR 处理不等比例日志；
- 通过广告主或流量桶实验减少竞价干扰。



### 8.3 特征

- 用户、Query、广告和上下文特征；
- treatment 前 pCTR/pCVR；
- bidding landscape、竞争强度；
- 广告主预算、pacing 和历史消耗；
- 出价档属性和库存状态。

若决策发生在曝光前，当前曝光、点击和转化都是 treatment 后变量，不能作为当前决策特征。

### 8.4 模型

离散出价可用多 treatment T-Learner/TARNet/DR；连续出价可学习剂量响应：

```text
mu(x,b) = E[Y(b) | X=x]
```

对连续 action，必须保证各价格区间有探索覆盖；模型不能在没有支持的高出价区域随意外推。

### 8.5 评估

- 离散出价档的 uplift 分桶和校准；
- 完整策略的 DR Policy Value；
- 不同预算下的消耗、GMV 和平台收入回放；
- 线上评估增量曝光、增量 GMV、增量成本、增量 ROI（iROI，定义见 11.3 节）和广告主 ROI。



### 8.6 决策

单请求边际价值：

```text
V(x,b) = DeltaBenefit(x,b) - DeltaCost(x,b)
```

预算影子价格下：

```text
b_lambda*(x) = argmax_b [V(x,b) - lambda * DeltaCost(x,b)]
```

选择最大值为正的出价档，并叠加广告主 ROI、pacing、流量质量和平台收入约束。

---



## 九、通用模型怎么选



### 9.1 T-Learner

每个 treatment 一个模型：

```text
tau_hat_t(x) = mu_hat_t(x) - mu_hat_0(x)
```

适合 treatment 少、各组样本充足、需要可信基线的场景。缺点是样本被拆散，模型维护成本高。

### 9.2 S-Learner

一个模型输入 treatment：

```text
mu_hat_t(x) = f(x,t)
```

推理时对同一用户遍历所有 treatment。优点是维护简单；缺点是用户特征过强时模型容易忽略 treatment。

### 9.3 TARNet / Shared Bottom + Treatment Heads

共享底层表示，每个 treatment 一个 outcome head。它兼顾样本共享和 treatment 差异，是多干预深度模型的常用主结构。

注意：

- TARNet 仍是“先预测潜在结果，再做差”；
- 它不自动解决混杂偏差；
- treatment head 与多业务 task head 不应混淆；
- 多 outcome 时可在每个 treatment 下再拆 task head（见第七章）；
- **训练只更新 factual head；预测时每个 treatment（及每个 outcome）都要出数。**

### 9.4 DR-Learner

仅预测 `mu_t - mu_0` 容易受到 outcome 模型误差影响。多 treatment 的 Doubly Robust 伪标签：

```text
phi_{i,t} = mu_hat_t(X_i) - mu_hat_0(X_i)
            + I(T_i=t) / e_hat_t(X_i) * [Y_i - mu_hat_t(X_i)]
            - I(T_i=0) / e_hat_0(X_i) * [Y_i - mu_hat_0(X_i)]
```

第二阶段学习：

```text
tau_hat_t(x) = E[phi_{i,t} | X_i=x]
```

“双重稳健”：outcome 模型或 propensity 模型只要有一个估计正确，效应估计仍有机会保持一致。

要求：

- nuisance models 必须 cross-fitting（给某折样本造伪标签时， nuisance 模型不能见过该折）；
- propensity 过小时 clip/trim，避免权重爆炸；
- 样本不足时 DR 高方差，未必优于简单 T-Learner；
- 双重稳健不代表两个模型都可以随意训练。

### 9.5 Uplift Tree/Forest

树的分裂目标直接寻找 treatment-control 响应差异，属于直接效应建模。它适合二元 treatment、解释异质人群，但在高维稀疏特征和多 treatment 下通常不如通用 outcome 模型易扩展。

### 9.6 ZILN Loss（零膨胀对数正态）

面向 **大量为 0 + 正值长尾** 的金额标签（LTV、GMV、贡献毛利）。模型输出 `p`（非零概率）、`μ`、`σ`：

```text
L_ZILN = BCE(1{y>0}, p) + 1{y>0} * LognormalNLL(y; μ, σ)
```

期望还原：

```text
E[Y] = p * exp(μ + σ²/2)
```

比直接 MSE 更抗「全零 + 少数大单」。思想与 Hurdle 接近：先判是否成交/是否有值，再对正值用 lognormal。可作各 treatment head 的 outcome loss。

### 9.7 推荐路线

```text
少量固定 treatment
  → T-Learner 可信基线
  → Shared Bottom + Treatment Heads
  → 非随机数据再加入 DR

大量组合 treatment（如 9 选 2）
  → 单券/品类 head + 组合决策层
  → 或 treatment embedding / conditioned head
  → 必须保留探索覆盖

连续 treatment
  → dose-response
  → 严格限制在实验支持域

金额零膨胀 outcome
  → Hurdle 或 ZILN
```

---



## 十、离线评估：不能只看 AUC



### 10.1 Uplift 分桶

按预测 uplift 从高到低分桶。非等比例实验或观察数据中，桶 `b` 内 treatment `t` 的加权 outcome：

```text
mu_hat_{t,b} =
  sum_{i in bucket b} I(T_i=t) * Y_i / e_hat_t(X_i)
  -------------------------------------------------------
  sum_{i in bucket b} I(T_i=t) / e_hat_t(X_i)
```

相对 control 的真实分桶 uplift：

```text
Uplift_hat_{t,b} = mu_hat_{t,b} - mu_hat_{0,b}
```

等概率随机实验中权重相同，可以退化为两组样本均值之差。

### 10.2 AUUC 与 Qini

- 适合单个 treatment 相对 control 的排序评估；
- 必须使用独立测试集或 OOF 预测；
- 连续 outcome 可以构造累计增量曲线，但需确认工具实现不只支持二元标签；
- 多 treatment 分别计算 one-vs-control AUUC，不代表最终多券策略一定好。



### 10.3 多 Treatment Policy Value

给定策略 `pi(x)`，IPS：

```text
V_hat_IPS(pi) = (1/N) * sum_{i=1..N}
                I(T_i=pi(X_i)) * Y_i / e_hat_{T_i}(X_i)
```

SNIPS：

```text
V_hat_SNIPS(pi) =
  sum_i I(T_i=pi(X_i)) * Y_i / e_hat_{T_i}(X_i)
  ------------------------------------------------
  sum_i I(T_i=pi(X_i)) / e_hat_{T_i}(X_i)
```

DR Policy Value：

```text
V_hat_DR(pi) = (1/N) * sum_{i=1..N} [
  mu_hat_{pi(X_i)}(X_i)
  + I(T_i=pi(X_i)) / e_hat_{T_i}(X_i)
    * (Y_i - mu_hat_{T_i}(X_i))
]
```

策略模型、propensity 和 outcome 模型都应使用与评估样本隔离的数据或交叉拟合，避免策略价值乐观。

### 10.4 其他必查项

- propensity 分布和有效样本量；
- treatment 间 overlap；
- 预测 uplift 与真实分桶 uplift 的校准；
- 时间、人群、城市和渠道稳定性；
- Bootstrap 用户/Query/城市簇置信区间；
- 券档/出价档分配比例和预算回放；
- 长期复购、疲劳、退款及供需干扰。



### 10.5 在线实验

离线指标只能筛模型，最终需要随机实验：

```text
A：当前规则或旧策略
B：Uplift策略
C：随机探索桶
```

应在相同预算、相同准入人群和一致观察窗口下比较增量业务价值。

---



## 十一、从 Uplift 到成本、约束与 ROI

### 11.0 两段式：模型出数，决策层求解约束

国内发券/补贴/bidding **绝大多数不把预算、补贴率、利润率写进 uplift 网络端到端训练**。

```text
模型阶段：μ_t(x)、核销概率、期望成本、多 outcome
    ↓
决策阶段：算 V / Score → 硬过滤（补贴率、频控、准入）
         → 阈值 / 影子价格 λ / 离线 LP·MIP 选动作
```

| 量 | 怎么来 | 用在哪 |
| ---- | ------ | ------ |
| `ΔC` | 核销扣：`P(redeem)×面额`；发放扣：直接面额 | 价值公式、预算 |
| `ΔM` | `μ_t-μ_0` 再结合客单/毛利结构 | 净价值分子 |
| `V` | `ΔM-ΔC` | **优化目标** |
| 补贴率 | 成本/GMV 或成本/成交价上限 | 硬过滤或约束 |
| iROI / 利润率 | `ΔM/ΔC ≥ ρ` | **安全约束**，不是目标 |

可微 knapsack 有研究，生产仍以两段式 + `λ` pacing 为主。

### 11.1 先统一价值符号

对用户 `i` 和 treatment `t`，定义：

- `G_i(t)`：GMV；
- `M_i(t)`：扣除营销成本前的贡献毛利；
- `C_i(t)`：营销成本；
- control 通常满足 `C_i(0)=0`。

条件增量：

```text
DeltaG_{i,t} = E[G_i(t) - G_i(0) | X_i]
```

```text
DeltaM_{i,t} = E[M_i(t) - M_i(0) | X_i]
```

```text
DeltaC_{i,t} = E[C_i(t) - C_i(0) | X_i]
```

增量净价值：

```text
V_{i,t} = DeltaM_{i,t} - DeltaC_{i,t}
```

这是通用公式，不依赖“各 treatment 客单价相同”的假设。

### 11.2 什么时候可以写成转化 uplift × 客单价值

设 `p_i,t=P(Order_i(t)=1 | X_i)`。一般情况下：

```text
DeltaM_{i,t} = p_{i,t} * m_{i,t} - p_{i,0} * m_{i,0}
```

其中 `m_i,t` 是 treatment `t` 下成交后的条件贡献毛利。

只有近似认为 `m_i,t=m_i,0=m_i` 时，才能简化为：

```text
DeltaM_{i,t} ~= (p_{i,t} - p_{i,0}) * m_i = tau_{i,t} * m_i
```

因此原先直接使用 `deltaP_{i,t} * g_{i,t}` 的写法并不通用：券可能改变客单价、酒店星级和订单结构。

若券在核销时才产生成本，且 control 没有营销成本：

```text
DeltaC_{i,t} = P(Redeem_i(t)=1 | X_i) * q_t
```

若发放即产生成本，则直接使用发放成本，不再乘核销概率。

### 11.3 ROI 的三个口径

增量 GMV ROI：

```text
iROI_GMV = DeltaGMV / DeltaMarketingCost
```

增量贡献毛利 ROI：

```text
iROI_margin = DeltaContributionMargin / DeltaMarketingCost
```

净利润 ROI：

```text
ROI_net = (DeltaContributionMargin - DeltaMarketingCost)
          / DeltaMarketingCost
        = iROI_margin - 1
```

因此：

- `iROI_GMV>1` 只代表增量 GMV 大于成本，不等于利润不亏；
- 真正“不亏”应要求 `DeltaContributionMargin >= DeltaMarketingCost`；
- 若 control 也有营销成本，分母必须使用增量成本，而不是 treatment 组总成本。

每增量订单成本：

```text
CPIO = DeltaMarketingCost / DeltaOrders
```

control 无营销成本时，分子等于 treatment 组营销成本；分母始终是相对 control 的增量订单，不能使用 treatment 组总订单。

### 11.4 全局预算优化与求解档位

令 `x_{i,t} in {0,1}` 表示是否给用户 `i` 选择 treatment `t`。

最大化总增量净价值：

```text
maximize_x  sum_{i,t} V_{i,t} * x_{i,t}
```

每个用户最多一个 treatment：

```text
sum_t x_{i,t} <= 1,  for all i
```

预算约束：

```text
sum_{i,t} DeltaC_{i,t} * x_{i,t} <= B
```

最低贡献毛利 ROI 约束：

```text
sum_{i,t} DeltaM_{i,t} * x_{i,t}
  >= rho_min * sum_{i,t} DeltaC_{i,t} * x_{i,t}
```

当 `rho_min=1` 时，对应贡献毛利覆盖营销成本。

**工程上三档求解：**

1. **阈值版（灰度最常见）**：每档要求 `V>0`、`iROI≥ρ`、补贴率≤上限，再取 `V` 最大。好解释，非全局最优。
2. **影子价格 / 拉格朗日（线上实时主力）**：

```text
L(x,lambda) =
  sum_{i,t} [V_{i,t} - lambda * DeltaC_{i,t}] * x_{i,t}
  + lambda * B,  lambda >= 0
```

给定 `lambda` 后，用户级选择：

```text
t_i*(lambda) =
  argmax_{t in {0,...,K}} [V_{i,t} - lambda * DeltaC_{i,t}]
```

将 control 定义为 `V_{i,0}=0, DeltaC_{i,0}=0`，即可自然选择“不干预”。预算打满调高 `λ`，花不完调低（二分或 pacing）。补贴率、频控作选前硬过滤。

3. **离线 LP/MIP**：用户×券矩阵日更/小时更，带城市/券库存；线上用对偶 `λ` 近似。

这里没有重复扣费：

- `V = DeltaM - DeltaC` 中的成本是实际经济成本；
- `lambda * DeltaC` 是预算稀缺产生的影子成本。

### 11.5 半智能阈值与智能定价

第一版可为每档券设置覆盖成本的 uplift 阈值，便于灰度和解释；但券间可能冲突，也无法实现全局最优。

第二版将全部用户—券组合的 `V_{i,t}` 输入 LP/MIP 或拉格朗日求解，同时处理：

- 日预算和券库存；
- 城市/渠道预算；
- 每用户唯一券（或 Top-2 组合约束）；
- 频控和准入；
- 最低 ROI 与补贴率上限；
- 风控和用户体验；
- 保留随机探索流量。

不应直接选择 ROI 最大的券。ROI 是效率比率，净价值是绝对收益；合理做法是把 ROI 作为安全约束，在约束内最大化总净价值。

---



## 十二、高频问题与坑位排查

### 12.1 高频问题速答

| 问题 | 核心回答 |
| ---- | -------- |
| 没有随机实验能否做 | 仅在混杂记录完整且有 overlap 时用 Matching/IPW/DR；国内主流是小流量探索桶 + 大流量去偏，最终仍要随机验收 |
| 场景偏置算不算随机 | 不算。特征与 propensity 必须吃场景/意图；分场景评估；只在可发人群里估效应 |
| 为什么不能把核销当 treatment | 核销是 post-treatment，会造成选择偏差；主分析用 ITT |
| 随机实验是否还需要 propensity | 等概率是已知常数；不等比例、分层随机和 Policy Value 仍需要 |
| T、S、TARNet 怎么选 | 少 treatment 先 T；共享信息强用 Shared Bottom + Heads；非随机再加 DR |
| TARNet 是否属于差分建模 | 是，先输出各 treatment 潜在结果，再在预测阶段做差 |
| 多 head 训练只监督实际 T，其他头干嘛 | 其他头是反事实头，本步不算 loss；预测时每个 T 都要出 μ 再做差 |
| TARNet 能否多 outcome | 能。每个 treatment head 输出多维；factual 多任务加权 loss；约束在决策层 |
| TARNet 里的约束怎么求 | 预算/补贴率/供需一般不在网络 loss 里，用 V−λΔC 或 LP 求 |
| 预算、补贴率、利润率在模型里做吗 | 多数否。模型出 μ/成本；决策层阈值、影子价格或 LP |
| 最大 uplift 为什么不是最优 | 还要客单结构、毛利、成本、预算和 ROI |
| 为什么不直接选 ROI 最大 | ROI 是比率；目标应是总净价值，ROI/补贴率当约束 |
| ATT 估计是干嘛的 | 只对被处理人群估效应；飞猪 Matching 用于已投广告流量诊断，非全人群外推 |
| ATE / ATT / CATE | 人群平均 / 被处理者平均 / 条件个性化；发券主用 CATE |
| 连续值分档后怎么还原 | 每档存真实代表值，`Σ P(档)×代表值`；不要档号相减；优先 Hurdle/ZILN |
| ZILN 是什么 | 零膨胀 + 对数正态损失，适合金额长尾；`E[Y]=p·exp(μ+σ²/2)` |
| PSM 怎么建怎么用 | 多分类倾向分 + 匹配；估 ATT 或洗样本；不当线上 uplift 主分 |
| PSM 和 IPW 区别 | 匹配对齐再比 vs `1/e` 加权；发券评估更常 IPW/DR |
| 9 券选 2 合在一起还是分业务 | 品类内共享估单券价值 + 跨品类组合决策；勿扁平 36 类硬训，勿三业务各取 top 硬拼 |
| 多 treatment 为什么不能只看 Qini | 单 treatment 排序≠最终唯一动作策略；看 DR Policy Value |
| AUC 高为什么 Uplift 可能差 | AUC 看谁会响应，Uplift 看谁会因干预改变 |
| 旅行有随机还会出什么问题 | 意图异质、跨品类搬量、触发成本、退款、价变、位次、多触点、库存干扰、疲劳 |

### 12.2 上线前排查

**样本：**

- treatment 是否为系统动作，control 是否为真实基准；
- 是否保存真实分流概率、实验版本和可选 action 集合；
- propensity 是否极端，treatment 是否有共同支持；
- 是否存在跨用户、跨流量或供需干扰；
- 多品类/组合动作是否记录展示集合而非仅点击券。

**特征与模型：**

- 所有特征是否早于 treatment，是否混入点击、核销等后验行为；
- S-Learner 是否忽略 treatment，各 head 是否输出坍缩；
- 训练是否只回传 factual head，预测是否遍历全部 treatment；
- 是否对训练未覆盖的 treatment/组合做 OOD 外推；
- 分档还原是否用代表值；金额是否考虑零膨胀；
- DR 是否 cross-fitting，权重是否做 clip/trim。

**评估与策略：**

- 是否误用训练集 AUUC/Qini，非等比例数据是否加权；
- 是否评估完整策略、置信区间和分层（含场景）稳定性；
- GMV、贡献毛利、净利润和营销成本口径是否一致；
- control 有成本时是否使用增量成本；
- 预算/ROI/补贴率是否在决策层统一，避免只追单业务 KPI；
- 是否保证每用户唯一动作（或合法 Top-K），并保留探索预算；
- 线上是否在相同预算和观察窗口下比较。

---

## 十三、项目表达速记

| 项目 | 一句话抓手 | 技术关键词 | 决策关键词 |
| ---- | ---------- | ---------- | ---------- |
| 酒店券包与红包 | 避免补贴自然订单 | 多组 RCT、T-Learner、Shared Bottom + Heads、DR | 增量净价值、预算、ROI |
| 旅行多品类 Top-2 | 跨品类选展，防搬量 | 品类内共享、单券价值、组合决策、探索桶 | Top-2、补贴率、λ、大盘 Y |
| 搜索悬浮球 | 找到展示后真正增加搜索的人 | 用户级 RCT、Hurdle/分档还原、连续 AUUC | 展示价值、频控、体验 |
| 飞猪搜索广告 | 衡量广告对平台大盘而非广告自身 | Original Space Matching、ATT、Effect-Net | 准入、参竞概率、多目标约束 |
| 33 个补贴组合 ABR | 区分业务搬量与大盘增量 | 多 treatment×多 outcome TARNet、switchback | GMV、净价值、供需安全 |
| Bidding 流量扶持 | 把资源给加价后才会被撬动的流量 | 随机出价桶、DR、dose-response | 边际价值、pacing、影子价格 |

---

## 十四、一句话收束

> Uplift 的核心不是预测“谁会响应”，而是在可信实验或可识别样本上估计“动作相对基准带来多少增量”，再把增量转换成统一价值，在成本、预算、ROI 和业务约束下选择动作（含不干预或 Top-K 组合）；约束求解放在决策层，模型负责出可信的 `μ`/`τ`。

---
