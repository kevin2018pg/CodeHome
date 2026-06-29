# algo_trainer — 算法自动化训练 & 预测服务

基于 FastAPI + XGBoost 的轻量级算法自动化后端服务，支持模型训练、持久化存储和在线预测。

---

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务（本地）
uvicorn main:app --host 127.0.0.1 --port 8080

# 启动服务（服务器，允许外网访问）
uvicorn main:app --host 0.0.0.0 --port 8080
```

启动后访问以下地址可在浏览器中可视化调用所有接口：
- **本地**：http://127.0.0.1:8080/docs
- **线上**：https://algotrainer.17usoft.com/docs

---

## CSV 文件格式要求

### 训练文件（带标签）

| id | feature_1 | feature_2 | ... | label |
|----|-----------|-----------|-----|-------|
| user_001 | 0.12 | -1.34 | ... | 1 |
| user_002 | 0.85 | 0.67 | ... | 0 |

- **id 列**：必须包含，可以是数字序号、时间戳、字符串等任意格式，**不参与训练**，仅用于结果追踪
- **特征列**：数值型，列名自定义
- **label 列**：分类任务为 0/1 或多类整数，回归任务为连续值

### 预测文件（无标签）

| id | feature_1 | feature_2 | ... |
|----|-----------|-----------|-----|
| pred_001 | 0.33 | 0.91 | ... |
| pred_002 | -0.5 | 1.20 | ... |

- **id 列**：必须包含，会原样透传到预测结果中，用于与原始数据对应
- **特征列**：列名必须与训练时一致（顺序可以不同，服务会自动对齐）
- **不需要 label 列**

---

## 接口总览

> 所有训练接口都支持 `auto_tune` 参数：`false`（默认）用指定超参训练，`true` 自动搜索最优超参。

| 接口 | 方式 | 适用场景 |
|------|------|---------|
| `POST /train` | 文件上传 | 大数据量，本地/脚本调用 |
| `POST /train_text` | JSON文本 | **飞书/OpenClaw 直接粘贴数据** |
| `POST /train_from_path` | 本地路径 | 大数据，文件提前放服务器 |
| `POST /predict` | 文件上传 | 大数据量，返回CSV文件 |
| `POST /predict_text` | JSON文本 | **飞书/OpenClaw 直接粘贴，返回JSON** |
| `POST /predict_from_path` | 本地路径 | 大数据，结果可保存到服务器 |
| `GET /models` | - | 查看所有模型 |
| `GET /models/{id}` | - | 查看模型详情 |
| `DELETE /models/{id}` | - | 删除模型 |

---

## 接口说明

### 1. 训练模型 `POST /train`

**curl 示例：**
```bash
curl -X POST https://algotrainer.17usoft.com/train \
  -F "file=@train.csv" \
  -F "model_id=hotel_ctr_v1" \
  -F "owner=推荐组" \
  -F "task=classification" \
  -F "auto_tune=false"
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| file | 文件 | 必填 | 训练CSV文件 |
| id_col | string | `id` | ID列名 |
| label_col | string | `label` | 标签列名 |
| task | string | `classification` | `classification`（分类）或 `regression`（回归） |
| model_id | string | 自动生成 | 自定义模型ID，**建议填写**，后续预测时使用；已存在则覆盖 |
| owner | string | `unknown` | 归属人/业务标识，用于区分不同人的模型 |
| n_estimators | int | 100 | 树的数量 |
| max_depth | int | 6 | 树的最大深度 |
| learning_rate | float | 0.1 | 学习率 |
| subsample | float | 0.8 | 样本采样比例 |
| test_size | float | 0.2 | 验证集比例 |
| auto_tune | bool | `false` | 是否自动搜索最优超参，`true` 时忽略上方超参设置 |
| n_trials | int | 30 | `auto_tune=true` 时的搜索轮数，建议 20~50 |

**返回示例：**
```json
{
  "model_id": "hotel_ctr_v1",
  "owner": "推荐组",
  "task": "classification",
  "n_features": 20,
  "feature_names": ["f1", "f2", "..."],
  "train_samples": 8000,
  "val_samples": 2000,
  "metrics": {
    "accuracy": 0.9635,
    "auc": 0.9951
  },
  "trained_at": "2026-03-09 11:14:03"
}
```

> **重要**：记住返回的 `model_id`，预测时需要用到。

---

### 2. 预测 `POST /predict`

**curl 示例：**
```bash
curl -X POST https://algotrainer.17usoft.com/predict \
  -F "file=@predict.csv" \
  -F "model_id=hotel_ctr_v1" \
  -F "output_prob=true" \
  -o result.csv
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| file | 文件 | 必填 | 预测CSV文件（含id列，无label列） |
| model_id | string | 必填 | 训练时的模型ID |
| id_col | string | `id` | ID列名 |
| output_prob | bool | `false` | 分类任务是否同时输出预测概率 |

**返回：** CSV文件下载，格式为 `id列 + 特征列 + 预测结果列`

| id | feature_1 | ... | pred_prob | pred_label |
|----|-----------|-----|-----------|------------|
| pred_001 | 0.33 | ... | 0.92 | 1 |
| pred_002 | -0.5 | ... | 0.08 | 0 |

- 分类任务（`output_prob=true`）：输出 `pred_prob`（正类概率）+ `pred_label`（预测标签）
- 分类任务（`output_prob=false`）：只输出 `prediction`（预测标签）
- 回归任务：输出 `prediction`（预测值）

> **注意**：如果模型不存在会返回 404 错误，提示先训练。

---

### 3. 查看模型列表 `GET /models`

**curl 示例：**
```bash
curl https://algotrainer.17usoft.com/models
```

返回所有已训练的模型及其元信息。

```json
{
  "count": 2,
  "models": [
    {
      "model_id": "hotel_ctr_v1",
      "owner": "推荐组",
      "task": "classification",
      "n_features": 20,
      "metrics": {"accuracy": 0.96, "auc": 0.99},
      "trained_at": "2026-03-09 11:14:03"
    }
  ]
}
```

### 4. 查看模型详情 `GET /models/{model_id}`

```bash
curl https://algotrainer.17usoft.com/models/hotel_ctr_v1
```

返回某个模型的完整元信息，包括特征列名、超参数、训练指标等。

### 5. 删除模型 `DELETE /models/{model_id}`

```bash
curl -X DELETE https://algotrainer.17usoft.com/models/hotel_ctr_v1
```

---

### 6. 文本训练 `POST /train_text` （飞书/OpenClaw 专用）

请求体为 JSON，`csv_content` 直接填 CSV 文本，**无需上传文件**：

**curl 示例：**
```bash
curl -X POST https://algotrainer.17usoft.com/train_text \
  -H "Content-Type: application/json" \
  -d '{
    "csv_content": "id,f1,f2,label\n1,0.1,0.2,1\n2,-0.5,1.3,0",
    "model_id": "my_model_v1",
    "owner": "张三",
    "task": "classification"
  }'
```

```json
{
  "csv_content": "id,f1,f2,label\n1,0.1,0.2,1\n2,-0.5,1.3,0",
  "model_id": "my_model_v1",
  "owner": "张三",
  "task": "classification"
}
```

### 7. 文本预测 `POST /predict_text` （飞书/OpenClaw 专用）

请求体为 JSON，结果以 JSON 返回，可直接在飞书消息中查看：

**curl 示例：**
```bash
curl -X POST https://algotrainer.17usoft.com/predict_text \
  -H "Content-Type: application/json" \
  -d '{
    "csv_content": "id,f1,f2\n1,0.1,0.2\n2,-0.5,1.3",
    "model_id": "my_model_v1",
    "output_prob": true
  }'
```

```json
{
  "csv_content": "id,f1,f2\n1,0.1,0.2\n2,-0.5,1.3",
  "model_id": "my_model_v1",
  "output_prob": true
}
```

返回：
```json
{
  "model_id": "my_model_v1",
  "total_rows": 2,
  "results": [
    {"id": 1, "f1": 0.1, "f2": 0.2, "pred_prob": 0.92, "pred_label": 1},
    {"id": 2, "f1": -0.5, "f2": 1.3, "pred_prob": 0.08, "pred_label": 0}
  ]
}
```

### 8. 路径训练 `POST /train_from_path`

文件提前放到服务器，传路径即可：

**curl 示例：**
```bash
curl -X POST https://algotrainer.17usoft.com/train_from_path \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/data/train.csv",
    "model_id": "hotel_ctr_v1",
    "owner": "推荐组",
    "auto_tune": true,
    "n_trials": 30
  }'
```

```json
{
  "file_path": "/data/train.csv",
  "model_id": "hotel_ctr_v1",
  "owner": "推荐组"
}
```

### 9. 路径预测 `POST /predict_from_path`

预测结果可选择保存到服务器，同时返回前10行预览：

**curl 示例：**
```bash
curl -X POST https://algotrainer.17usoft.com/predict_from_path \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/data/predict.csv",
    "model_id": "hotel_ctr_v1",
    "output_prob": true,
    "save_result_path": "/data/result.csv"
  }'
```

```json
{
  "file_path": "/data/predict.csv",
  "model_id": "hotel_ctr_v1",
  "output_prob": true,
  "save_result_path": "/data/result.csv"
}
```

---

## 典型使用流程

### 常规流程（脚本/工具调用）
```
准备训练CSV（含 id 列 + 特征列 + label 列）
    ↓
POST /train  →  拿到 model_id（如 hotel_ctr_v1）
    ↓
准备预测CSV（含 id 列 + 特征列，无 label 列）
    ↓
POST /predict（带上 model_id）  →  下载预测结果CSV
```

### 飞书 OpenClaw 流程（直接对话）
```
在飞书中通过 OpenClaw 下达命令，粘贴CSV数据
    ↓
POST /train_text（JSON body，csv_content 填训练数据）
    →  返回 model_id 和评估指标
    ↓
POST /predict_text（JSON body，csv_content 填预测数据）
    →  返回 JSON 结果，直接在飞书消息中查看
```

### 大数据量流程（提前上传到服务器）
```
把 CSV 文件上传到服务器指定目录
    ↓
POST /train_from_path（传文件路径）  →  拿到 model_id
    ↓
POST /predict_from_path（传文件路径 + save_result_path）
    →  结果自动保存到服务器，返回预览摘要
```

---

## 模型隔离说明

不同人/不同业务场景使用不同的 `model_id` 来隔离：

```
hotel_ctr_v1     # 酒店CTR模型（推荐组）
ad_cvr_model     # 广告CVR模型（广告组）
recall_score_v2  # 召回排序模型 v2
```

每个模型独立保存，互不干扰。预测时服务会自动校验特征列名是否与该模型训练时一致。

---

## 错误说明

| 状态码 | 原因 | 解决方法 |
|--------|------|---------|
| 400 | CSV缺少id列 | 确保CSV包含id列 |
| 400 | CSV缺少label列 | 确保训练CSV包含label列 |
| 400 | 预测数据缺少特征列 | 特征列名需与训练时一致 |
| 404 | 模型不存在 | 先调用 /train 训练模型 |

---

## 注意事项

### 模型持久化

训练好的模型保存在容器内的 `/app/models/` 目录。**容器重启后模型文件会丢失**，需在部署平台上为该目录挂载持久化存储（NFS 或对象存储），否则每次重启都需要重新训练。

---

## 文件结构

```
algo_trainer/
├── main.py              # 服务主程序
├── test_service.py      # 自动化测试脚本
├── requirements.txt     # 依赖
├── README.md            # 本文档
└── models/              # 模型持久化目录
    ├── hotel_ctr_v1.pkl     # 模型权重
    └── hotel_ctr_v1.json    # 模型元信息
```
