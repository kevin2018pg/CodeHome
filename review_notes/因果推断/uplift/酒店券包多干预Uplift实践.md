# 多干预 Uplift 与约束决策 · 实践

> 本文用五个项目统一说明 Uplift 的工程落地：酒店券包与红包、搜索悬浮球、飞猪搜索广告、33 个补贴组合 ABR、Bidding 流量扶持。每个项目均按**样本、特征、模型、评估、决策**展开。

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

### 1.2 四类用户

| 人群 | 不干预 | 干预 | 策略 |
|---|---:|---:|---|
| Persuadable | 不响应 | 响应 | 核心干预对象 |
| Sure Thing | 响应 | 响应 | 避免浪费成本 |
| Lost Cause | 不响应 | 不响应 | 无需干预 |
| Sleeping Dog | 响应 | 不响应 | 必须避免干预 |

四类人是概念解释。单个用户的两个潜在结果无法同时观察，不能直接给每个用户打真实四分类标签。

### 1.3 因果识别的三个条件

1. **一致性与 SUTVA**：用户实际接受 `t` 时，观察结果等于 `Y(t)`；一个用户的 treatment 不影响其他用户。
2. **可忽略性**：给定全部干预前混杂因素后，treatment 分配与潜在结果独立。
3. **重叠性**：目标人群中的每种 treatment 都有被分配的可能：

```text
0 < e_t(x) = P(T=t | X=x) < 1,  for all t
```

随机实验主要解决 treatment 分配偏差，但广告竞价、出行供需等场景还可能存在用户间干扰，不能只靠普通用户级随机分桶。

---

## 二、五个项目总览

| 项目 | Treatment | Outcome | 核心偏差 | 推荐起点 | 最终动作 |
|---|---|---|---|---|---|
| 酒店券包与红包 | 不发券 + 多档券 | 订单、GMV、贡献毛利 | 给高意向用户发券，补贴自然订单 | 多组 RCT + T-Learner | 每人选择唯一券档或不发 |
| 搜索悬浮球 | 展示/不展示 | 未来 7 日搜索量 | 高活跃用户更容易被展示 | 用户级 RCT + XGBoost | 圈选真正增加搜索的人群 |
| 飞猪搜索广告 | 投广告/不投广告 | 平台整体 GMV、营收 | 广告由竞价和规则选择 | Matching + Effect-Net | 广告准入、流量与预算分配 |
| 33 个补贴组合 ABR | 33 个补贴组合 | 三业务应答率（ABR）、GMV、贡献毛利 | 多 treatment、多 outcome、供需干扰 | TARNet 多头结构 | 约束下选择补贴组合 |
| Bidding 流量扶持 | 不扶持 + 多档出价 | 曝光、转化、GMV、消耗 | 只有胜出样本可见，选择偏差强 | 随机出价桶 + DR | 预算内选择最优出价 |

共同链路：

```text
实验或去偏样本
  → pre-treatment features
  → 估计 μ_t(x) 或 τ_t(x)
  → 转换成增量 GMV / 毛利 / 成本
  → 预算和业务约束下选 treatment
  → 在线随机实验验证
```

---

## 三、通用数据基础：样本质量先于模型

### 3.1 Treatment 是系统动作，不是用户响应

正确：

```text
发券：系统是否分配某张券
悬浮球：系统是否展示
广告：系统是否投放广告
Bidding：系统采用哪个出价档
```

错误：

```text
是否领券、核销、点击、购买
```

领取、点击、核销均发生在 treatment 之后。把这些行为当 treatment，会将本来就高意向的用户筛进处理组，产生选择偏差。主分析通常估计 ITT（Intention To Treat，意向治疗效应），即系统分配动作带来的整体增量。

### 3.2 优先保留随机探索流量

推荐记录：

- 用户/请求 ID；
- treatment ID；
- 真实分流概率 `e_t(x)`；
- 实验版本、准入条件、分组时间；
- 触达状态；
- 统一观察窗口内的 outcome、GMV、毛利和成本；
- 退款、取消及长期结果。

分层随机、不等比例随机同样可信，但必须保存每条样本的真实分配概率。

### 3.3 非随机数据怎么办

只有在“给定 `X` 后没有遗漏混杂因素”时，观察数据去偏才有因果解释。

常用顺序：

1. 检查 treatment 规则和日志是否完整；
2. 训练多分类 propensity `e_t(x)`；
3. 查看各 treatment 的 overlap；
4. 对极端 propensity 做 trim 或 clip；
5. 使用 IPW、DR 或 Matching；
6. 在独立评估集重复平衡检查；
7. 最终仍用线上随机实验验收。

Propensity 主要用于训练去偏、评估和 overlap 检查，通常不是线上业务模型的最终输出。

### 3.4 特征边界

只允许使用决策时已经存在的特征：

- 用户长期价值和生命周期；
- 历史行为、偏好、价格敏感度；
- 当前请求、场景、时间和渠道；
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

训练时只监督样本实际 treatment 对应的 head：

```text
L_outcome = (1/N) * sum_{i=1..N} loss(Y_i, mu_hat_{T_i}(X_i))
```

预测时遍历全部券档：

```text
tau_hat_t(x) = mu_hat_t(x) - mu_hat_0(x)
```

非随机数据可使用 DR-Learner，详见第九章。

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

将不发券定义为 `V_{i,0}=0`。预算充足时只发放 `V_{i,t}>0` 的最优券；预算受限时再使用第十一章的拉格朗日影子价格、ROI、库存和频控约束。

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

若将搜索量分成 10 档，必须用每档真实搜索量代表值还原条件期望：

```text
mu_hat_t(x) = sum_{b=1..10} P(B(t)=b | X=x) * y_bar_b
```

不能直接把两个类别编号相减。其他可比较方案包括 `log1p` 回归、Poisson/Tweedie、Huber loss 和 winsorize。

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

匹配后的 ATT 估计：

```text
ATT_hat = (1/N_T) * sum_{i:T_i=1} [Y_i - sum_{j in N_K(i)} w_ij * Y_j]
```

其中：

```text
sum_{j in N_K(i)} w_ij = 1
```

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

使用多 treatment、多 outcome TARNet：

```text
Shared Representation
  ├─ Treatment 0 Head → [ABR_fast, ABR_bargain, ABR_taxi]
  ├─ ...
  └─ Treatment 32 Head → [ABR_fast, ABR_bargain, ABR_taxi]
```

多 outcome factual loss：

```text
L = (1/N) * sum_{i=1..N} sum_m omega_m * loss_m(Y_{i,m}, mu_hat_{T_i,m}(X_i))
```

残差门控可写为：

```text
h_{t,m} = g_{t,m}(h) .* F_{t,m}(h) + (1-g_{t,m}(h)) .* P_{t,m}(h)
```

其中 `P_t,m` 在维度不同时必须做投影。门控增强的是共享与拟合，不是因果识别来源。

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

最后在预算、供需安全和折扣约束下最大化 `Score_t(x)`，不能只按逐单最高 ABR 贪心。

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
- 多 outcome 时可在每个 treatment 下再拆 task head。

### 9.4 DR-Learner

对 treatment `t` 相对 control 的 DR 伪标签：

```text
phi_{i,t} = mu_hat_t(X_i) - mu_hat_0(X_i)
            + I(T_i=t) / e_hat_t(X_i) * [Y_i - mu_hat_t(X_i)]
            - I(T_i=0) / e_hat_0(X_i) * [Y_i - mu_hat_0(X_i)]
```

第二阶段学习：

```text
tau_hat_t(x) = E[phi_{i,t} | X_i=x]
```

要求：

- nuisance models 必须 cross-fitting；
- propensity 过小时 clip/trim；
- 样本不足时 DR 高方差，未必优于简单 T-Learner；
- 双重稳健不代表两个模型都可以随意训练。

### 9.5 Uplift Tree/Forest

树的分裂目标直接寻找 treatment-control 响应差异，属于直接效应建模。它适合二元 treatment、解释异质人群，但在高维稀疏特征和多 treatment 下通常不如通用 outcome 模型易扩展。

### 9.6 推荐路线

```text
少量固定 treatment
  → T-Learner 可信基线
  → Shared Bottom + Treatment Heads
  → 非随机数据再加入 DR

大量组合 treatment
  → treatment embedding / conditioned head
  → 必须保留探索覆盖

连续 treatment
  → dose-response
  → 严格限制在实验支持域
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

### 11.4 全局预算优化

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

预算约束的拉格朗日函数：

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

其中将 control 定义为 `V_{i,0}=0, DeltaC_{i,0}=0`，即可自然选择“不干预”。

这里没有重复扣费：

- `V = DeltaM - DeltaC` 中的成本是实际经济成本；
- `lambda * DeltaC` 是预算稀缺产生的影子成本。

通过二分或在线控制调整 `lambda`，使总预算接近 `B`。

### 11.5 半智能阈值与智能定价

第一版可为每档券设置覆盖成本的 uplift 阈值，便于灰度和解释；但券间可能冲突，也无法实现全局最优。

第二版将全部用户—券组合的 `V_{i,t}` 输入 LP/MIP 或拉格朗日求解，同时处理：

- 日预算和券库存；
- 城市/渠道预算；
- 每用户唯一券；
- 频控和准入；
- 最低 ROI；
- 风控和用户体验；
- 保留随机探索流量。

不应直接选择 ROI 最大的券。ROI 是效率比率，净价值是绝对收益；合理做法是把 ROI 作为安全约束，在约束内最大化总净价值。

---

## 十二、高频问题与坑位排查

### 12.1 高频问题速答

| 问题 | 核心回答 |
|---|---|
| 没有随机实验能否做 | 仅在混杂因素记录完整且有 overlap 时使用 Matching/IPW/DR，最终仍需随机实验 |
| 为什么不能把核销当 treatment | 核销是用户选择后的 post-treatment 行为，会造成选择偏差 |
| 随机实验是否还需要 propensity | 等概率实验是已知常数；不等比例、分层随机和 Policy Value 仍需要 |
| T、S、TARNet 怎么选 | 少 treatment 先做 T；共享信息强再做 S/Shared Bottom + Treatment Heads；非随机数据再加 DR |
| TARNet 是否属于差分建模 | 是，先输出各 treatment 潜在结果，再在预测阶段做差 |
| AUC 高为什么 Uplift 可能差 | AUC 识别谁会响应，Uplift 识别谁会因干预而改变 |
| 多 treatment 为什么不能只看 Qini | 单 treatment 排序不代表最终唯一动作策略有效，应看 DR Policy Value |
| 最大 uplift 为什么不是最优动作 | 还要考虑客单结构、贡献毛利、营销成本、预算和 ROI |

### 12.2 上线前排查

**样本：**

- treatment 是否为系统动作，control 是否为真实基准；
- 是否保存真实分流概率、实验版本和可选 action 集合；
- propensity 是否极端，treatment 是否有共同支持；
- 是否存在跨用户、跨流量或供需干扰。

**特征与模型：**

- 所有特征是否早于 treatment，是否混入点击、核销等后验行为；
- S-Learner 是否忽略 treatment，各 head 是否输出坍缩；
- 是否对训练未覆盖的 treatment 做 OOD 外推；
- DR 是否 cross-fitting，权重是否做 clip/trim。

**评估与策略：**

- 是否误用训练集 AUUC/Qini，非等比例数据是否加权；
- 是否评估完整策略、置信区间和分层稳定性；
- GMV、贡献毛利、净利润和营销成本口径是否一致；
- control 有成本时是否使用增量成本；
- 是否保证每用户唯一动作，并保留探索预算；
- 线上是否在相同预算和观察窗口下比较。

---

## 十三、项目表达速记

| 项目 | 一句话抓手 | 技术关键词 | 决策关键词 |
|---|---|---|---|
| 酒店券包与红包 | 避免补贴自然订单 | 多组 RCT、T-Learner、Shared Bottom + Treatment Heads、DR | 增量净价值、预算、ROI |
| 搜索悬浮球 | 找到展示后真正增加搜索的人 | 用户级 RCT、Hurdle、连续 AUUC | 展示价值、频控、体验 |
| 飞猪搜索广告 | 衡量广告对平台大盘而非广告自身的效果 | Original Space Matching、Effect-Net | 准入、参竞概率、多目标约束 |
| 33 个补贴组合 ABR | 区分业务搬量与大盘增量 | 多 treatment、多 outcome TARNet、switchback | GMV、净价值、供需安全 |
| Bidding 流量扶持 | 把资源给加价后才会被撬动的流量 | 随机出价桶、DR、dose-response | 边际价值、pacing、影子价格 |

---

## 十四、附录：现有价敏 Notebook 的升级边界

现有模板主链路是：

```text
XGBoost S-Learner
  → 覆写 trt 得到各档 p_t
  → 形成价敏曲线
  → KMeans 分群
  → OR-Tools MIP 分配
```

可保留“响应曲线 + 运筹分配”的框架，但上线前必须修正：

1. `trt=0` 是高星 96 折，不是真实 control；
2. 不得推理训练集未覆盖的 treatment；
3. AUUC 只能使用独立测试或 OOF 预测；
4. S-Learner 需与 T-Learner/Shared Bottom + Treatment Heads 比较；
5. scaler 训练后保存，预测只调用 `transform`；
6. 人工构造 `favor_0` 只能用于演示；
7. MIP 目标由总订单升级为增量净价值。

---

## 十五、一句话总结

> Uplift 的核心不是预测“谁会响应”，而是在可信实验或可识别样本上估计“动作相对基准带来多少增量”，再把增量转换成统一价值，在成本、预算、ROI 和业务约束下选择唯一动作。
