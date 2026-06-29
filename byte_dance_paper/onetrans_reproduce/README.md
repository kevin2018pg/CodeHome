# OneTrans 复现 —— 酒店推荐排序模型

> 论文：**OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer in Industrial Recommender**（WWW 2026，字节跳动 & 南洋理工大学）

---

## 论文核心思想

传统排序模型将**序列建模**（用户行为历史）和**特征交互**（用户/Item/上下文特征）分成两个独立模块。OneTrans 用**单一 Transformer 骨干**统一处理两者：

```
传统方式：序列编码器 → 压缩表示 → 特征交互模块 → 预测
OneTrans：统一 Tokenizer → 单一 Transformer Stack → 预测
```

三大创新点：
- **Mixed 参数化**：S-token（行为序列）共享参数，NS-token（非序列特征）独立参数
- **Pyramid Stack**：逐层裁剪 S-token 数量，将序列信息蒸馏到 NS-token，降低计算量
- **Cross-Request KV Cache**：同一用户多个候选复用 S-side KV，推理延迟降低 ~30%

---

## 项目结构

```
onetrans_reproduce/
├── 核心模块
│   ├── model.py                  # OneTrans 模型（MixedMHA / MixedFFN / Pyramid / Calibration）
│   ├── tokenizer.py              # Tokenizer（NSFeatureEncoder / AutoSplit / Sequential）
│   └── data.py                   # 数据模块（mock 模拟数据 + Parquet 流式加载）
│
├── 训练
│   ├── train.py                  # 全量训练（每天一次，更新所有参数）
│   └── train_embedding.py        # 增量训练（高频更新，支持 emb / head / all 三种模式）
│
├── 推理 & 部署
│   ├── infer.py                  # 离线批量推理 + TorchScript 导出
│   ├── serving/
│   │   └── model_loader.py       # 线程安全热加载（轮询 HDFS latest.json，无需重启服务）
│   └── torchserve/
│       ├── onetrans_handler.py   # TorchServe Handler
│       ├── build_mar.py          # 打包 .mar 文件
│       ├── setup_config_onetrans.json
│       ├── config.properties
│       └── sample_request.json
│
├── 工具
│   ├── check_parquet.py          # 训练前数据格式验证
│   └── feature_config.yaml       # 特征配置（字段名 / 类型 / 归一化方式）
│
├── requirements.txt
├── INTERVIEW.md                  # 面试准备文档（酒店推荐场景）
└── .gitignore
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install torch scikit-learn tqdm pandas pyarrow pyyaml
```

### 2. 用模拟数据验证模型（无需任何文件）

```bash
python train.py --data_mode mock --epochs 3
```

### 3. 用真实数据训练（Parquet 格式）

**Step 1**：修改 `feature_config.yaml`，填入你的 Hive 字段名

**Step 2**：验证数据格式

```bash
python check_parquet.py \
    --data_dir ./data/parquet/train \
    --feature_config feature_config.yaml
```

**Step 3**：全量训练

```bash
python train.py \
    --data_mode parquet \
    --train_dir ./data/parquet/train \
    --val_dir   ./data/parquet/val \
    --feature_config feature_config.yaml \
    --d_model 256 --n_layers 6 --batch_size 2048 --epochs 5 \
    --save_path ./checkpoints/best_model.pt
```

---

## 增量训练（高频更新）

三种模式对应不同的更新频率和更新目标：

| 模式 | 更新内容 | 建议频率 | 优化器 |
|------|---------|---------|--------|
| `emb` | 指定 ID Embedding（如 hotel_id） | 每3小时 | Adagrad（稀疏友好） |
| `head` | Task Head + Calibration 位置偏置 | 每1小时 | Adam |
| `all` | emb + head 同时更新 | 按需 | Adagrad(emb) + Adam(head) |

```bash
# 每3小时：更新 hotel_id Embedding（处理新酒店冷启动）
python train_embedding.py \
    --mode emb \
    --base_checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/parquet/recent_3h \
    --feature_config feature_config.yaml \
    --emb_features hotel_id \
    --save_path ./checkpoints/emb_update_latest.pt \
    --hdfs_base hdfs://namenode:9000/models/onetrans   # 可选，不填则只保存本地

# 每1小时：更新 Task Head + Calibration（应对节假日/促销分布漂移）
python train_embedding.py \
    --mode head \
    --base_checkpoint ./checkpoints/best_model.pt \
    --data_dir ./data/parquet/recent_1h \
    --feature_config feature_config.yaml \
    --save_path ./checkpoints/head_update_latest.pt
```

---

## 推理服务热加载

训练完成后写入 HDFS，推理服务自动轮询 `latest.json` 热替换模型，无需重启：

```python
# 推理服务 main.py
from serving.model_loader import ModelLoader   # HDFS 模式
# from serving.model_loader import LocalModelLoader  # 本地测试模式

loader = ModelLoader(
    hdfs_base="hdfs://namenode:9000/models/onetrans",
    poll_interval=60,   # 每60秒检查一次新版本
    device="cuda",
)
loader.start()

# 每次推理
model = loader.get_model()
preds, _ = model(ns_inputs, sequences, timestamps, seq_masks, position_ids)
```

---

## 离线批量推理

```bash
# 批量推理，结果写入 CSV
python infer.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data_dir   ./data/parquet/test \
    --feature_config feature_config.yaml \
    --output ./predictions.csv

# 导出 TorchScript（C++ 服务部署）
python infer.py \
    --checkpoint ./checkpoints/best_model.pt \
    --data_dir   ./data/parquet/test \
    --feature_config feature_config.yaml \
    --export_ts ./model_deploy.pt
```

---

## TorchServe 在线服务

```bash
# 1. 打包 .mar
python torchserve/build_mar.py \
    --checkpoint ./checkpoints/best_model.pt \
    --output_dir ./mar_output \
    --version 1.0

# 2. 启动服务
torchserve --start \
    --model-store ./mar_output \
    --models onetrans=OneTrans_ranking.mar \
    --ts-config torchserve/config.properties

# 3. 测试
curl -X POST http://localhost:8080/predictions/onetrans \
     -H "Content-Type: application/json" \
     -d @torchserve/sample_request.json
```

---

## 特征配置说明（feature_config.yaml）

支持 4 种 NS 特征类型：

| type | 说明 | 关键参数 |
|------|------|---------|
| `continuous` | 连续数值（年龄、价格） | `scale`：归一化分母；`-1` 表示 log(x+1) |
| `discrete_id` | 离散整数 ID（城市、品牌） | `vocab_size`、`emb_dim` |
| `discrete_str` | 字符串类别（需 Hive 预处理为整数） | `vocab_size`、`emb_dim` |
| `multihot` | 多值离散（兴趣标签，逗号分隔） | `vocab_size`、`emb_dim`、`max_len`、`pooling` |

序列特征支持任意字段组合（通过 `fields` 列表声明），不同序列可以有不同字段：

```yaml
sequences:
  - name: click
    timestamps: click_seq_ts
    max_len: 50
    fields:
      - col: click_seq_ids    # 第1个字段，请求 JSON 里用 "ids" key
        scale: 1000000.0
      - col: click_seq_cats
        scale: 100.0
      - col: click_seq_prices
        scale: 1000.0

  - name: purchase
    timestamps: buy_seq_ts
    max_len: 10
    fields:
      - col: buy_seq_ids
        scale: 1000000.0
      - col: buy_seq_cats
        scale: 100.0
      - col: buy_seq_brands   # 购买序列有品牌字段，点击序列没有
        scale: 50000.0
```

> **注意**：OneTrans 的 S-token 共享参数，要求所有序列的 `len(fields)` 相同（seq_dim 一致）。字段数不同时，在 Hive 建表时补 0 列对齐。

---

## 工业部署架构

```
召回层（双塔 + Milvus）
    ↓ 召回 300 个候选酒店
排序层（OneTrans）
    ↓ 精排 Top 20

三个独立 Docker 训练容器：
  容器1  每天 02:00   train.py          全量训练，写 HDFS/full/
  容器2  每3小时      train_embedding.py --mode emb   更新 hotel_id Embedding，写 HDFS/emb/
  容器3  每1小时      train_embedding.py --mode head  更新 Head+Calibration，写 HDFS/head/

HDFS latest.json（active 字段优先级：head > emb > full）
    ↓ 推理服务每60秒轮询
serving/model_loader.py 热替换模型（不重启服务）
```

---

## 与论文的差异说明

| 论文 | 本实现 | 说明 |
|------|--------|------|
| 百亿参数规模 | 默认 128d × 4层 | 可通过 `--d_model 256 --n_layers 6` 扩大 |
| FlashAttention-2 | 标准 PyTorch attention | 需要 `pip install flash-attn` 后手动替换 |
| 混合精度训练 | FP32 | 可在 `train.py` 加 `torch.cuda.amp.autocast()` |
| 完整 KV Cache 推理 | 接口预留，未完整实现 | `forward_with_kv_cache` 为占位实现 |
| 位置偏置 Calibration | 已实现 | `model.position_bias`，通过 `position_ids` 传入 |
