# Feature Schema

这个文件解释 `feature_config.json` 里的字段含义、类型和处理方式。

## 处理方式约定

- `hash_bucket_embedding`
  - 适用于 `user_id/item_id/author_id/enum` 这类离散特征
  - 先做字符串化，再做哈希分桶，再查 embedding
  - 即便原始值是数字，也不会按连续数值处理

- `linear_projection`
  - 适用于连续数值、统计值、比率值、二值值
  - 当前实现里会把单标量通过线性层投影到 token 子空间

## 字段分组

### user_profile

- `unionid`
  - 原始类型：`string_or_int64_id`
  - 模型类型：`categorical_id`
  - 处理：`hash_bucket_embedding`

- `membership_tier`
  - 原始类型：`enum_string`
  - 模型类型：`categorical_enum`
  - 处理：`hash_bucket_embedding`

- `device`
  - 原始类型：`enum_string`
  - 模型类型：`categorical_enum`
  - 处理：`hash_bucket_embedding`

- `user_segment`
  - 原始类型：`enum_string`
  - 模型类型：`categorical_enum`
  - 处理：`hash_bucket_embedding`

- `user_age`
- `activity_7d`
- `historical_ctr`
- `historical_cvr`
- `user_pay_tendency_30d`
  - 原始类型：数值
  - 模型类型：`numeric/numeric_stat`
  - 处理：`linear_projection`

### candidate

- `content_id`
  - 原始类型：`string_or_int64_id`
  - 模型类型：`categorical_id`
  - 处理：`hash_bucket_embedding`
  - 说明：当前实现里承担 item 主 ID 的建模角色

- `author_id`
- `brand_tier`
- `content_type`
  - 离散类别，走 `hash_bucket_embedding`

- `price`
- `quality_score`
- `seller_score`
- `candidate_score`
- `item_ctr_7d`
- `item_cvr_7d`
  - 数值/统计特征，走 `linear_projection`

### context

- `traffic_source`
  - 离散特征，走 `hash_bucket_embedding`

- `position`
- `hour_of_day`
- `is_weekend`
- `network_type`
- `session_depth`
  - 数值上下文特征，走 `linear_projection`

### sequence

- `seq_click_rate_7d`
- `seq_watch_depth_7d`
- `seq_skip_rate_7d`
- `seq_ecom_exposure_7d`
- `seq_live_watch_rate_7d`
  - 序列聚合统计特征，走 `linear_projection`

### cross

- `cross_ctr_gap`
- `cross_price_gap`
- `cross_user_author_affinity`
- `cross_user_item_similarity`
- `cross_query_item_intent`
  - 交叉统计/亲和度特征，走 `linear_projection`

## 标签

- `ctr_label`
  - 点击标签

- `cvr_label`
  - 转化标签

当前 RankMixer 是双任务：

- 一个 head 预测 `ctr_pred`
- 一个 head 预测 `cvr_pred`
- 排序分数默认是 `ctr_pred * cvr_pred`
