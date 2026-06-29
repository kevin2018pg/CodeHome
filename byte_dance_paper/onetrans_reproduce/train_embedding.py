"""
OneTrans 增量训练脚本（支持三种模式）

三个独立 Docker 容器各自定时触发，训练完写入 HDFS，推理服务轮询热加载。

模式说明：
  --mode emb   只更新 ID Embedding（hotel_id 等），用 Adagrad，每3小时跑一次
  --mode head  只更新 Task Head + Calibration 位置偏置，用 Adam，每1小时跑一次
  --mode all   同时更新 emb + head（数据量充足时使用）

HDFS 目录约定：
  hdfs_base/
    full/   best_model_YYYYMMDD.pt         ← train.py 写入
    emb/    emb_update_YYYYMMDD_HH.pt      ← 本脚本 mode=emb 写入
    head/   head_update_YYYYMMDD_HH.pt     ← 本脚本 mode=head 写入
    latest.json                            ← 记录各层最新路径，推理服务轮询此文件

latest.json 格式：
  {
    "full":       "hdfs://...full/best_model_20240115.pt",
    "emb":        "hdfs://...emb/emb_update_20240115_09.pt",
    "head":       "hdfs://...head/head_update_20240115_10.pt",
    "active":     "hdfs://...head/head_update_20240115_10.pt",
    "updated_at": "2024-01-15T10:05:00"
  }
  active 字段 = 推理服务应该加载的最新模型（优先级：head > emb > full）

Docker 定时任务配置示例（三个容器各自的 crontab）：

  # 容器1：每天 02:00 全量训练
  0 2 * * * python train.py --data_mode parquet \
      --train_dir /data/parquet/train --val_dir /data/parquet/val \
      --feature_config /app/feature_config.yaml \
      --save_path /tmp/best_model.pt
  # 训练完后上传 HDFS（train.py 里调用或外部脚本）

  # 容器2：每3小时更新 Embedding（06/09/12/15/18/21）
  0 6,9,12,15,18,21 * * * python train_embedding.py \
      --mode emb \
      --base_checkpoint /tmp/best_model.pt \
      --data_dir /data/parquet/recent_3h \
      --feature_config /app/feature_config.yaml \
      --emb_features hotel_id \
      --hdfs_base hdfs://namenode:9000/models/onetrans \
      --save_path /tmp/emb_update_latest.pt

  # 容器3：每小时更新 Head + Calibration
  0 * * * * python train_embedding.py \
      --mode head \
      --base_checkpoint /tmp/best_model.pt \
      --data_dir /data/parquet/recent_1h \
      --feature_config /app/feature_config.yaml \
      --hdfs_base hdfs://namenode:9000/models/onetrans \
      --save_path /tmp/head_update_latest.pt
"""

import argparse
import json
import os
import subprocess
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from data import LABEL_COLS, get_dataloader, load_feature_config
from train import load_checkpoint, save_checkpoint, multitask_loss, compute_auc


# ---------------------------------------------------------------------------
# 冻结策略
# ---------------------------------------------------------------------------

FREEZE_RULES = {
    # mode -> 解冻哪些参数（按参数名前缀匹配）
    "emb": [
        "ns_encoder.embeddings",     # 所有 NS Embedding（由 --emb_features 进一步过滤）
    ],
    "head": [
        "task_heads",                # 多任务预测头
        "position_bias",             # Calibration 位置偏置
    ],
    "all": [
        "ns_encoder.embeddings",
        "task_heads",
        "position_bias",
    ],
}


def apply_freeze(model: nn.Module, mode: str, emb_features: List[str]) -> Dict:
    """
    按 mode 冻结模型参数。
    emb_features 在 mode=emb/all 时进一步限定只更新哪些 Embedding。
    """
    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    prefixes = FREEZE_RULES[mode]
    unfrozen = []

    for name, param in model.named_parameters():
        for prefix in prefixes:
            if name.startswith(prefix):
                # emb 模式：只解冻 emb_features 指定的特征
                if prefix == "ns_encoder.embeddings":
                    feat_name = name.split(".")[2]   # ns_encoder.embeddings.<feat>.weight
                    if feat_name not in emb_features:
                        continue
                param.requires_grad = True
                unfrozen.append(name)
                break

    if not unfrozen:
        raise ValueError(
            f"mode={mode!r} 下没有找到可训练参数。\n"
            f"emb_features={emb_features}\n"
            f"模型中的 Embedding: {list(model.ns_encoder.embeddings.keys()) if model.ns_encoder else '无'}"
        )

    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n[冻结策略] mode={mode}")
    print(f"  解冻的参数: {unfrozen}")
    print(f"  冻结: {frozen/1e6:.2f}M 参数  |  可训练: {trainable/1e3:.1f}K 参数")
    return {"frozen": frozen, "trainable": trainable, "unfrozen": unfrozen}


def get_optimizer(model: nn.Module, mode: str, lr: float):
    """
    不同 mode 用不同优化器：
    - emb:  Adagrad，对稀疏 Embedding 更新友好（每个 ID 独立累积梯度历史）
    - head: Adam，参数量小，Adam 收敛更快
    - all:  分组优化，emb 用 Adagrad，head 用 Adam
    """
    trainable = [p for p in model.parameters() if p.requires_grad]

    if mode == "emb":
        return torch.optim.Adagrad(trainable, lr=lr)

    if mode == "head":
        return torch.optim.Adam(trainable, lr=lr)

    # mode == "all"：emb 用 Adagrad，head 用 Adam，分组传入各自优化器
    emb_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "embeddings" in name:
            emb_params.append(param)
        else:
            head_params.append(param)

    # 返回一个 optimizer 列表，train_one_epoch 里会依次 step
    optimizers = []
    if emb_params:
        optimizers.append(torch.optim.Adagrad(emb_params, lr=lr))
    if head_params:
        optimizers.append(torch.optim.Adam(head_params, lr=lr * 0.1))
    return optimizers


# ---------------------------------------------------------------------------
# 训练 & 评估
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, grad_clip=1.0) -> Dict:
    """
    optimizer 支持单个 optimizer 或 list（mode=all 时传入 [Adagrad, Adam]）。
    """
    model.train()
    # 统一为列表，方便统一 zero_grad / step
    optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
    total_loss, n = 0.0, 0

    for batch in tqdm(loader, desc="训练", leave=False):
        ns_inputs = {k: v.to(device) for k, v in batch["ns_inputs"].items()}
        seqs      = [s.to(device) for s in batch["sequences"]]
        ts        = [t.to(device) for t in batch["timestamps"]]
        masks     = [m.to(device) for m in batch["seq_masks"]]
        pos_ids   = batch.get("position_id")
        if pos_ids is not None:
            pos_ids = pos_ids.to(device)

        batch_labels = {k: v.to(device) for k, v in batch.items()
                        if k not in ("ns_inputs", "sequences", "timestamps", "seq_masks", "position_id")}

        for opt in optimizers:
            opt.zero_grad()
        preds, _ = model(ns_inputs, seqs, ts, masks, pos_ids)
        loss = multitask_loss(preds, batch_labels)
        loss.backward()

        trainable = [p for p in model.parameters() if p.requires_grad]
        nn.utils.clip_grad_norm_(trainable, grad_clip)
        for opt in optimizers:
            opt.step()

        total_loss += loss.item()
        n += 1

    return {"loss": total_loss / max(n, 1)}


@torch.no_grad()
def quick_eval(model, loader, device, task_names, max_batches=30) -> Dict:
    model.eval()
    preds_all  = {t: [] for t in task_names}
    labels_all = {t: [] for t in task_names}
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ns_inputs = {k: v.to(device) for k, v in batch["ns_inputs"].items()}
        seqs      = [s.to(device) for s in batch["sequences"]]
        ts        = [t.to(device) for t in batch["timestamps"]]
        masks     = [m.to(device) for m in batch["seq_masks"]]
        preds, _  = model(ns_inputs, seqs, ts, masks)
        for name in task_names:
            if name in preds:
                # model 返回 logit，评估时转为概率
                preds_all[name].extend(torch.sigmoid(preds[name]).cpu().tolist())
            lk = f"{name}_label"
            if lk in batch:
                labels_all[name].extend(batch[lk].tolist())
    return {
        f"{name}_auc": compute_auc(preds_all[name], labels_all[name])
        for name in task_names if preds_all[name]
    }


# ---------------------------------------------------------------------------
# HDFS 工具
# ---------------------------------------------------------------------------

def hdfs_put(local_path: str, hdfs_path: str):
    """上传文件到 HDFS，hdfs_path 为完整路径如 hdfs://host:9000/models/..."""
    # 先确保 HDFS 目录存在
    hdfs_dir = os.path.dirname(hdfs_path)
    subprocess.run(["hdfs", "dfs", "-mkdir", "-p", hdfs_dir], check=True)
    subprocess.run(["hdfs", "dfs", "-put", "-f", local_path, hdfs_path], check=True)
    print(f"[HDFS] 上传完成: {local_path} → {hdfs_path}")


def hdfs_get(hdfs_path: str, local_path: str):
    """从 HDFS 下载文件到本地"""
    os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
    subprocess.run(["hdfs", "dfs", "-get", "-f", hdfs_path, local_path], check=True)
    print(f"[HDFS] 下载完成: {hdfs_path} → {local_path}")


def update_latest_json(hdfs_base: str, mode: str, new_hdfs_path: str, run_time: str):
    """
    更新 HDFS 上的 latest.json：
    1. 先下载当前版本
    2. 更新对应字段
    3. 重新上传
    """
    latest_hdfs = f"{hdfs_base}/latest.json"
    latest_local = "/tmp/onetrans_latest.json"

    # 尝试下载现有 latest.json，不存在则用空模板
    try:
        hdfs_get(latest_hdfs, latest_local)
        with open(latest_local) as f:
            meta = json.load(f)
    except Exception:
        meta = {"full": "", "emb": "", "head": ""}

    # 更新对应字段
    if mode in ("emb", "head", "full"):
        meta[mode] = new_hdfs_path
    elif mode == "all":
        meta["emb"]  = new_hdfs_path
        meta["head"] = new_hdfs_path

    meta["updated_at"] = run_time

    # active = 优先级最高的可用模型（head > emb > full）
    meta["active"] = meta.get("head") or meta.get("emb") or meta.get("full") or ""

    with open(latest_local, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    hdfs_put(latest_local, latest_hdfs)
    print(f"[HDFS] latest.json 已更新: active={meta['active']}")
    return meta


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(args):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_time = datetime.now().strftime("%Y%m%d_%H%M")
    mode     = args.mode
    emb_features = [n.strip() for n in args.emb_features.split(",")]

    print(f"\n{'='*60}")
    print(f"增量训练  mode={mode}  时间={run_time}  设备={device}")
    print(f"基础模型: {args.base_checkpoint}")
    if mode in ("emb", "all"):
        print(f"更新 Embedding: {emb_features}")
    if mode in ("head", "all"):
        print(f"更新 Task Head + Calibration")
    print(f"{'='*60}\n")

    # ── 加载基础模型 ──
    # 优先从 HDFS 拉取最新的 base_checkpoint
    local_base = args.base_checkpoint
    if args.hdfs_base and args.base_checkpoint.startswith("hdfs://"):
        local_base = f"/tmp/onetrans_base_{run_time}.pt"
        hdfs_get(args.base_checkpoint, local_base)

    model = load_checkpoint(local_base, str(device))
    model.train()

    # ── 冻结策略 ──
    apply_freeze(model, mode, emb_features)
    optimizer = get_optimizer(model, mode, args.lr)

    # ── 加载特征配置 ──
    if args.feature_config:
        _, _, label_cols = load_feature_config(args.feature_config)
    else:
        label_cols = LABEL_COLS
    task_names = [c.replace("_label", "") for c in label_cols]

    # ── 加载数据 ──
    train_loader, _ = get_dataloader(
        mode="parquet", shuffle=True,
        data_dir=args.data_dir,
        feature_config=args.feature_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    val_loader = None
    if args.val_dir:
        val_loader, _ = get_dataloader(
            mode="parquet", shuffle=False,
            data_dir=args.val_dir,
            feature_config=args.feature_config,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )

    # ── 训练循环 ──
    best_auc = 0.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, args.grad_clip)
        elapsed = time.time() - t0
        log = f"Epoch {epoch}/{args.epochs} | Loss={train_metrics['loss']:.4f} | {elapsed:.1f}s"

        if val_loader:
            val_metrics = quick_eval(model, val_loader, device, task_names)
            ctr_auc = val_metrics.get("ctr_auc", 0.0)
            log += f" | CTR AUC={ctr_auc:.4f}"
            if ctr_auc > best_auc:
                best_auc = ctr_auc
                _do_save(model, local_base, args, run_time, epoch, val_metrics)
        else:
            _do_save(model, local_base, args, run_time, epoch, train_metrics)

        print(log)

    # ── 上传 HDFS + 更新 latest.json ──
    if args.hdfs_base:
        # mode=all 时同时更新 emb 和 head 两个子目录
        upload_modes = ["emb", "head"] if mode == "all" else [mode]
        hdfs_path = None
        for m in upload_modes:
            fname     = f"{m}_update_{run_time}.pt"
            hdfs_path = f"{args.hdfs_base}/{m}/{fname}"
            hdfs_put(args.save_path, hdfs_path)
            meta = update_latest_json(args.hdfs_base, m, hdfs_path, run_time)
        meta = update_latest_json(args.hdfs_base, upload_modes[-1], hdfs_path, run_time)
        print(f"\n[完成] 模型已上传 HDFS，推理服务将在下次轮询时自动加载")
        print(f"  active model: {meta['active']}")
    else:
        print(f"\n[完成] 模型已保存到本地: {args.save_path}")
        print(f"  （未配置 --hdfs_base，跳过 HDFS 上传）")


def _do_save(model, local_base, args, run_time, epoch, metrics):
    base_ckpt = torch.load(local_base, map_location="cpu", weights_only=False)
    save_checkpoint(
        model,
        model_kwargs=base_ckpt["model_kwargs"],
        epoch=base_ckpt.get("epoch", 0),
        metrics={**base_ckpt.get("metrics", {}), f"{args.mode}_update": run_time, **metrics},
        path=args.save_path,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OneTrans 增量训练（emb / head / all）")

    p.add_argument("--mode", type=str, required=True, choices=["emb", "head", "all"],
                   help="emb=只更新Embedding  head=只更新TaskHead+Calibration  all=两者都更新")
    p.add_argument("--base_checkpoint", type=str, required=True,
                   help="基础模型路径，支持本地路径或 hdfs:// 路径")
    p.add_argument("--data_dir",        type=str, required=True,
                   help="训练数据目录（Parquet）")
    p.add_argument("--feature_config",  type=str, default=None)
    p.add_argument("--emb_features",    type=str, default="hotel_id",
                   help="mode=emb/all 时要更新的 Embedding 特征，逗号分隔")
    p.add_argument("--save_path",       type=str, default="/tmp/incremental_latest.pt",
                   help="本地保存路径（上传 HDFS 前的临时文件）")
    p.add_argument("--hdfs_base",       type=str, default=None,
                   help="HDFS 根目录，如 hdfs://namenode:9000/models/onetrans")
    p.add_argument("--val_dir",         type=str, default=None)
    p.add_argument("--epochs",          type=int,   default=1)
    p.add_argument("--batch_size",      type=int,   default=512)
    p.add_argument("--lr",              type=float, default=None,
                   help="学习率（不填则按 mode 自动选择：emb=0.01, head=0.001）")
    p.add_argument("--grad_clip",       type=float, default=1.0)
    p.add_argument("--num_workers",     type=int,   default=0)

    args = p.parse_args()

    # 按 mode 自动选择默认学习率
    if args.lr is None:
        args.lr = 0.01 if args.mode == "emb" else 0.001

    main(args)
