# 多干预 Uplift · 因果核心与项目速记

> 复习用：只留因果识别、去偏工具、模型选型、项目抓手、约束决策与踩坑。公式一律纯文本，避免预览渲染失败。

---

## 一、Uplift 在问什么

普通 CVR：发券后会不会下单？  
Uplift：相对不发，**多带来多少**订单/GMV/毛利？

```text
mu_t(x) = E[Y(t) | X=x]
tau_t(x) = mu_t(x) - mu_0(x)          # CATE，发券主目标
Y_obs = Y(T)                           # 只观察实际 T；做差在预测阶段
```

| 量 | 含义 | 何时用 |
| -- | ---- | ------ |
| ATE | 人群平均效应 | 全量开关实验 |
| ATT | **仅对被处理者**的平均效应 | Matching 诊断「已投放值不值」 |
| CATE | 给定 x 的个性化效应 | 选人、选券、选价 |
| ITT | 按**系统分配**估效应 | 主分析；不要用领券/核销当 T |

四类人（Persuadable / Sure Thing / Lost Cause / Sleeping Dog）只是解释框架，**无法给每人打真实四分类标签**。

### 识别三条件（不会这三条就不要谈因果）

1. **一致性 + SUTVA**：观察到的 Y 等于 `Y(T)`；一人 treatment 不干扰他人（库存/竞价常破坏）。
2. **可忽略性**：给定干预前混杂 X，T 与潜在结果独立（随机实验直接给；观察数据靠 X 够不够）。
3. **Overlap**：`0 < e_t(x)=P(T=t|X=x) < 1`。某群人几乎必然发/不发 → 不可识别，应 trim。

---

## 二、样本铁律（比模型重要）

**T = 系统动作**（发哪档券、展不展示、投不投广告），不是领券/点击/核销。

**X 只能用决策前特征**；禁止用本次点击、核销、干预后搜索/曝光、T 改变后的排序统计。

优先：**随机探索桶**（1%–10%），日志写死真实 `e_t(x)`。等概率实验 `e` 是常数，**不必再训 propensity**。

观察/规则流量：国内主流 = 大流量 Matching/IPW/DR 筛策略 + **小流量随机验收**；propensity **只服务离线去偏与评估，不是线上打分**。

---

## 三、去偏工具箱（IPW / Matching / DR）

先估或读取倾向分 `e(x)=P(T=1|X)`（多 treatment 则多分类）。X 必须含规则用到的混杂（如历史活跃度）。

### Overlap 是什么

给定 x，处理组与对照组都要有人。`e≈0` 或 `e≈1` → 没有对照，IPW 权重爆炸、Matching 找不到邻居。  
**删没 overlap 的人** = trim 极端 e（如出 `[0.05,0.95]`）；结论只对修剪后人群成立。

### IPW

```text
ATE ≈ mean( I(T=1)Y/e(X) - I(T=0)Y/(1-e(X)) )
```

或样本权重 `1/e_T` 再训 outcome。`e` 极端 → **clip/trim**，或用 SNIPS 归一化。

### Matching（含 PSM）

按 `e(X)` 或原始特征找近邻；caliper 丢远邻；  
`ATT ≈ mean(Y_treated - 加权对照 Y)`。  
飞猪：用广告插入前自然队列向量匹配。用于估 ATT / 洗样本，**不当线上 uplift 分**。

### DR（双重稳健）

```text
phi = mu1(X)-mu0(X)
    + I(T=1)/e(X)*(Y-mu1(X))
    - I(T=0)/(1-e(X))*(Y-mu0(X))
tau(x) = E[phi | X=x]          # 第二阶段再回归
```

outcome **或** propensity 有一个对，效应仍可能对。必须 **cross-fitting**；仍建议 clip e。

### Propensity 要不要每次估？

| 情况 | 做法 |
| ---- | ---- |
| 等概率 RCT | 不用估，e 已知 |
| 不等比例/分层随机 | 用日志概率，不必重训 |
| 观察数据训/评 | 离线训一次 nuisance，定期更新 |
| 线上推理 | **不跑** propensity |

---

## 四、模型怎么选

| 结构 | 要点 | 适用 |
| ---- | ---- | ---- |
| T-Learner | 每 T 一个 `mu_t`，预测做差 | 少 T、基线 |
| S-Learner | `f(X,T)`，推理扫所有 T | 简单；易忽略 T |
| TARNet / 多 head | Shared + 每 T 一个 head | 多 T 主力 |
| DR-Learner | 伪标签再回归 | 非随机加强 |

**多 head**：训练只对 factual `T_i` 算 loss；其他头是反事实，靠该 T 的历史样本更新。**预测必须跑全部 head** 再 `mu_t - mu_0`。多 outcome：每个 head 输出多维，factual 加权 loss；**预算约束不在网络里求**。

### 零膨胀 Y：Hurdle vs ZILN

都不是「神秘优化器」，是 **Y 分布建模**。

```text
Hurdle:  E[Y]=P(Y>0)*E[Y|Y>0]     # 正值分布自选；搜索量常用
ZILN:    E[Y]=p*exp(mu+sig^2/2)   # 正值固定 lognormal；GMV/LTV 常用
L_ZILN = BCE(1{y>0},p) + 1{y>0}*LognormalNLL
```

Y 分档建模时：每档存真实代表值，`E[Y]=Σ P(档)*代表值`，**禁止档号相减**。

### 选型路线

```text
少 T → T-Learner → Shared Heads → 非随机加 DR
组合动作（9选2）→ 单券/品类价值 + 决策层组合，勿扁平几十类硬训
连续出价 → dose-response，禁止无探索外推
```

---

## 五、项目卡片（只记抓手）

| 项目 | 偏差本质 | 做法抓手 | 决策 |
| ---- | -------- | -------- | ---- |
| 酒店多档券 | 高意向被发券=补贴自然单 | RCT + T/多 head；Y 用有效单/毛利 | `argmax V`；预算用 λ |
| 旅行 9 券选 2 | 场景偏置、跨品类搬量、组合稀疏 | 品类内共享估单券 V + 组合层；平台 Y | Top-2 + 品类/补贴率约束 |
| 搜索悬浮球 | 高活跃更易展示 | 活跃度进 propensity；IPW/DR/Matching；trim | `V=vs*τ-c`，频控 |
| 飞猪搜索广告 | 竞价选择；挤占自然 | Matching 估 ATT；双头/Effect-Net；平台 GMV | 准入/参竞看平台增量 |
| 33 补贴×三业务 | 搬量≠大盘增量；供需干扰 | 多 T×多 Y TARNet；switchback | Score=ΔGMV+αV，勿贪 ABR |
| Bidding 扶持 | 只有胜出可见 | 随机出价桶 + DR；dose-response | `argmax(V-λΔC)` |

**旅行 Top-2 首选**：品类内共享估单券，跨品类用平台净价值选对；勿三业务各取 top1 硬拼，勿 36 类扁平多分类。有随机仍要防：意图异质、搬量、触发成本、退款、位次、多触点、库存、疲劳。

**广告 Effect-Net 要点**：二值 Y 用 logit 残差 `sigmoid(z0+δ)`，勿在概率上无界相加。AUC↑ 不等于 Qini↑。

---

## 六、评估：别只看 AUC

### Uplift 分桶（评排序，不是把 Y 切档）

```text
1. 独立集/OOF 算 τ̂(x)
2. 按 τ̂ 从高到低切 10 桶
3. 桶内算加权 μ_T、μ_C（随机等权则均值）
4. 桶 uplift = μ_T - μ_C；高预测桶应真实也高
5. 累计增益 → AUUC/Qini（单 T vs control；多 T 策略看 Policy Value）
```

观察数据桶内：

```text
mu_t,b = Σ_{桶b,T=t} Y/e_t(X)  /  Σ_{桶b,T=t} 1/e_t(X)
```

### Policy Value（评完整策略 π）

```text
IPS:  mean( I(T=π(X)) Y / e_T(X) )
SNIPS: 分子同 IPS，分母再除权重和
DR:   mean( mu_π(X) + I(T=π)/e_T *(Y-mu_T) )
```

策略 / propensity / outcome 要交叉拟合，防乐观。线上：同预算 A/B + 长期挂探索桶。

---

## 七、决策：两段式（模型出数，决策层求解）

```text
模型 → μ_t、核销率、成本
决策 → V = ΔM - ΔC → 硬过滤(补贴率/频控) → 阈值 或 argmax(V-λΔC) 或 LP
```

```text
V_{i,t} = ΔM_{i,t} - ΔC_{i,t}
ΔC ≈ P(redeem)*面额     # 核销才扣；发放即扣则不用乘
仅当客单近似不变: ΔM ≈ τ * m
```

- **目标**：总净价值；**ROI/补贴率是约束不是目标**（勿选 ROI 最大）。
- `iROI_GMV>1` ≠ 赚钱；不亏看 `Δ毛利 ≥ Δ成本`。
- `λ`：预算打满调高、花不完调低；`V` 里已扣经济成本，`λΔC` 是稀缺影子成本，不重复扣。
- Control：`V_0=0, ΔC_0=0` → 可自然选不干预。

---

## 八、踩坑清单（上线前过一遍）

**识别 / 样本**

- [ ] T 是系统动作；control 是真不干预
- [ ] 无 post-treatment 特征进 X
- [ ] 有探索桶或已知 e；观察数据查 overlap 并 trim
- [ ] 供需/竞价场景考虑干扰（勿迷信用户级随机）

**模型**

- [ ] 训练 factual、预测全 T；非随机加 DR/权重且 clip e
- [ ] 零膨胀用 Hurdle/ZILN；分档用代表值还原
- [ ] 未覆盖的 T/组合不外推

**评估 / 决策**

- [ ] 不用训练集 AUUC；多 T 看 DR Policy Value
- [ ] 分场景/分层报；同预算比增量毛利/CPIO
- [ ] 约束在决策层；保留探索；每用户合法唯一动作或 Top-K

---

## 九、30 秒 FAQ

| Q | A |
| - | - |
| 没随机能不能做 | 可去偏筛方案，最终仍要随机验收 |
| 倾向分线上要跑吗 | 一般不用；离线去偏/评估用 |
| IPW 爆炸怎么办 | clip/trim、SNIPS、改 DR/Matching、加探索 |
| 最大 τ 为何不是最优 | 还要毛利、成本、预算、ROI |
| ATT vs CATE | ATT=已处理者平均；发券个性化用 CATE |
| AUC 高 uplift 差 | AUC 看响应，uplift 看增量 |

---

## 十、一句话

> 在可识别样本上估「相对基准的增量」→ 换成净价值 → 在预算/ROI 约束下选动作；模型出 `μ/τ`，约束用 `λ`/LP；随机探索是因果可信的底线。
