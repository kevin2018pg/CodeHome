# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Multi-project Python ML research & engineering monorepo with four independent subprojects. All documentation and comments are in Chinese. There is no shared dependency management or monorepo tooling — each project is self-contained.

## Projects

### algo_trainer/ — ML Training & Prediction Microservice
FastAPI + XGBoost backend for automated model training and inference. Supports CSV data via file upload, JSON text body (for Feishu/OpenClaw integration), and server-side file paths.

```bash
cd algo_trainer
pip install -r requirements.txt
uvicorn main:app --host 127.0.0.1 --port 8080  # Swagger at /docs
```

Tests:
```bash
python test_service.py   # endpoint integration tests
python test_local.py     # local validation
```

Key details:
- Single entry point: `main.py` (all routes defined here)
- Models persist as pickle + JSON metadata in `models/`
- Optuna auto-tune available via `auto_tune=true` parameter
- CSV must contain `id` column (passthrough, not trained on) and `label` column (training only)

### byte_dance_paper/onetrans_reproduce/ — OneTrans Paper Reproduction
PyTorch reproduction of "OneTrans: Unified Feature Interaction and Sequence Modeling with One Transformer" (WWW 2026). Hotel recommendation ranking model.

```bash
cd byte_dance_paper/onetrans_reproduce
pip install -r requirements.txt

# Mock data (no files needed):
python train.py --data_mode mock --epochs 3

# Real data (Parquet):
python check_parquet.py --data_dir ./data/parquet/train --feature_config feature_config.yaml
python train.py --data_mode parquet --train_dir ./data/parquet/train --val_dir ./data/parquet/val --feature_config feature_config.yaml

# Resume from checkpoint:
python train.py --resume checkpoints/global_stepXXXX/ckpt.pt
```

Architecture:
- `model.py` — MixedMHA, MixedFFN, Pyramid Stack, RMSNorm, position bias calibration
- `tokenizer.py` — NSFeatureEncoder, AutoSplit, sequential tokenization
- `data.py` — mock data generation + Parquet streaming DataLoader
- `train.py` — full training; `train_embedding.py` — incremental (emb/head/all modes)
- `infer.py` — batch inference + TorchScript export
- `feature_config.yaml` — YAML-driven feature definitions (change features without code changes)
- `serving/model_loader.py` — HDFS-based hot reload for production serving
- `torchserve/` — TorchServe deployment packaging

Key concepts:
- **Mixed parameterization**: S-tokens (sequences) share params; NS-tokens (non-sequence features) have independent params
- **Pyramid Stack**: progressive S-token pruning across layers
- Incremental training: 3 containers at different frequencies (daily full, 3h embedding, 1h head)

### tencent_advertising_gr/ — Tencent Advertising Competition (2025 Finals)
Generative next-item recommendation using HSTU model. Achieved score 0.129843 (advanced to finals).

```bash
cd tencent_advertising_gr
pip install "torch==2.7.1" polars numpy tqdm tensorboard

# Full pipeline:
./run.sh

# Or manually:
python -u get_stat.py
python -u main.py

# Inference:
./infer.sh checkpoints/global_step136000 10  # TOPK=10
```

All I/O configured via environment variables (set by `run.sh`):
- `TRAIN_DATA_PATH`, `TRAIN_CKPT_PATH`, `TRAIN_LOG_PATH`, `TRAIN_TF_EVENTS_PATH`, `USER_CACHE_PATH`

Architecture:
- `model.py` — HSTU model with mixed negative sampling
- `dataset.py` — DataLoader with dense embeddings
- `main.py` — training & evaluation loop
- `infer.py` — Top-K retrieval inference
- Data format: `seq.jsonl` lines of `[user_id, item_id, user_feat, item_feat, action, ts]`

Key techniques: in-batch + global negative sampling, log-q correction, hard negative mining with curriculum learning, action-conditioned generation.

### test/ — Data Integration Utilities
Spark/Hive chunked merge scripts for large-scale offline data processing (500K rows per chunk, memory-safe pandas joins).

## Hardware & Runtime

- Tested on NVIDIA H20 (96GB), CUDA 12.3
- Python 3.10+ required for ML projects
- algo_trainer works on CPU (XGBoost)
