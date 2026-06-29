"""
OneTrans 训练脚本

通过 --data_mode 切换数据来源：
  --data_mode mock     → 随机生成模拟数据，无需任何文件（默认，快速验证）
  --data_mode parquet  → 从 Hive 导出的 Parquet 文件加载（正式训练）

用法示例：
  # 模拟数据验证模型
  python train.py --data_mode mock --epochs 3

  # 真实数据训练（使用 YAML 配置文件定义特征字段）
  python train.py --data_mode parquet \
      --train_dir ./data/parquet/train \
      --val_dir   ./data/parquet/val   \
      --feature_config feature_config.yaml \
      --d_model 256 --n_layers 6 --batch_size 2048 --epochs 5
"""

import argparse
import math
import os
import time
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data import LABEL_COLS, get_dataloader, load_feature_config
from model import OneTrans


# ---------------------------------------------------------------------------
# 损失 & 评估
# ---------------------------------------------------------------------------

def multitask_loss(
    predictions: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    task_weights: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    # 使用 BCEWithLogitsLoss：内部 log-sum-exp 技巧，数值稳定，不受 sigmoid 饱和影响
    task_weights = task_weights or {}
    losses = []
    for name, logit in predictions.items():
        label_key = f"{name}_label"
        if label_key in batch:
            loss = nn.functional.binary_cross_entropy_with_logits(
                logit, batch[label_key].float()
            )
            losses.append(task_weights.get(name, 1.0) * loss)
    return torch.stack(losses).sum() if losses else torch.tensor(0.0)


def compute_auc(preds: list, labels: list) -> float:
    try:
        return roc_auc_score(labels, preds)
    except ValueError:
        return 0.5


# ---------------------------------------------------------------------------
# 训练 & 评估循环
# ---------------------------------------------------------------------------

def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """
    Warmup + Cosine Decay 学习率调度（P1-4）。
    前 warmup_steps 步线性升温（从 1/warmup_steps 开始，不从 0 开始），
    之后余弦衰减到 lr * 0.1。
    """
    def lr_lambda(step: int) -> float:
        # step 从 0 开始，+1 保证第一步 LR 不为 0
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, device, grad_clip=1.0) -> Dict:
    model.train()
    total_loss, n_batches, n_skipped = 0.0, 0, 0

    for batch in tqdm(loader, desc="Train", leave=False):
        ns_inputs = {k: v.to(device) for k, v in batch["ns_inputs"].items()}
        seqs      = [s.to(device) for s in batch["sequences"]]
        ts        = [t.to(device) for t in batch["timestamps"]]
        masks     = [m.to(device) for m in batch["seq_masks"]]
        batch_d   = {k: v.to(device) for k, v in batch.items()
                     if k not in ("ns_inputs", "sequences", "timestamps", "seq_masks")}

        pos_ids = batch.get("position_id")
        if pos_ids is not None:
            pos_ids = pos_ids.to(device)

        optimizer.zero_grad()
        preds, _ = model(ns_inputs, seqs, ts, masks, pos_ids)
        loss = multitask_loss(preds, batch_d)

        if not torch.isfinite(loss):
            n_skipped += 1
            optimizer.zero_grad()
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches  += 1

    if n_skipped > 0:
        print(f"  [警告] 本 epoch 跳过了 {n_skipped} 个 NaN/Inf batch（共 {n_batches + n_skipped} 个）")

    return {"loss": total_loss / max(n_batches, 1)}


@torch.no_grad()
def evaluate(model, loader, device, task_names) -> Dict:
    model.eval()
    all_preds  = {name: [] for name in task_names}
    all_labels = {name: [] for name in task_names}

    for batch in tqdm(loader, desc="Eval", leave=False):
        ns_inputs = {k: v.to(device) for k, v in batch["ns_inputs"].items()}
        seqs      = [s.to(device) for s in batch["sequences"]]
        ts        = [t.to(device) for t in batch["timestamps"]]
        masks     = [m.to(device) for m in batch["seq_masks"]]
        pos_ids = batch.get("position_id")
        if pos_ids is not None:
            pos_ids = pos_ids.to(device)
        preds, _ = model(ns_inputs, seqs, ts, masks, pos_ids)

        for name in task_names:
            if name in preds:
                # model 返回 logit，评估时转为概率
                all_preds[name].extend(torch.sigmoid(preds[name]).cpu().tolist())
            label_key = f"{name}_label"
            if label_key in batch:
                all_labels[name].extend(batch[label_key].tolist())

    return {
        f"{name}_auc": compute_auc(all_preds[name], all_labels[name])
        for name in task_names
        if all_preds[name] and all_labels[name]
    }


# ---------------------------------------------------------------------------
# Checkpoint 保存 / 加载
# ---------------------------------------------------------------------------

def save_checkpoint(model, model_kwargs: dict, epoch: int, metrics: dict, path: str):
    """
    保存完整 checkpoint，包含：
      - state_dict：模型权重
      - model_kwargs：重建模型所需的全部超参
      - epoch / metrics：训练状态，便于比较和筛选
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save({
        "state_dict":   model.state_dict(),
        "model_kwargs": model_kwargs,
        "epoch":        epoch,
        "metrics":      metrics,
    }, path)


def load_checkpoint(path: str, device: str = "cpu"):
    """
    从 checkpoint 文件恢复模型，返回 eval 模式的模型实例。

    用法：
        model = load_checkpoint("best_model.pt")
        predictions, _ = model(ns_inputs, sequences, timestamps, seq_masks)
    """
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = OneTrans(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    print(f"[Checkpoint] 加载成功: epoch={ckpt['epoch']}, metrics={ckpt['metrics']}")
    return model


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}  |  数据模式: {args.data_mode}")

    # 如果指定了 YAML 配置，从中读取标签字段；否则用代码默认值
    ns_feature_specs = None
    if args.feature_config:
        ns_feature_specs, _, label_cols = load_feature_config(args.feature_config)
    else:
        label_cols = LABEL_COLS

    task_names = [col.replace("_label", "") for col in label_cols]

    # ── 加载数据（mock / parquet 统一入口）──
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature_config=args.feature_config,
    )

    if args.data_mode == "mock":
        seq_dims = args.seq_dims or [32] * len(args.seq_lengths)
        train_loader, train_ds = get_dataloader(
            mode="mock", shuffle=True, seed=42,
            n_samples=args.n_train, ns_dim=args.ns_dim,
            seq_lengths=args.seq_lengths, seq_dims=seq_dims,
            label_cols=label_cols,
            **common,
        )
        val_loader, _ = get_dataloader(
            mode="mock", shuffle=False, seed=43,
            n_samples=args.n_val, ns_dim=args.ns_dim,
            seq_lengths=args.seq_lengths, seq_dims=seq_dims,
            label_cols=label_cols,
            **common,
        )
    else:
        assert args.train_dir and args.val_dir, \
            "parquet 模式需要 --train_dir 和 --val_dir"
        train_loader, train_ds = get_dataloader(
            mode="parquet", shuffle=True,
            data_dir=args.train_dir, **common,
        )
        val_loader, _ = get_dataloader(
            mode="parquet", shuffle=False,
            data_dir=args.val_dir, **common,
        )

    # 从 dataset 自动读取模型输入维度
    seq_dims    = train_ds.seq_dims
    seq_lengths = train_ds.seq_lengths
    # parquet 模式用 spec 里的 ns_feature_specs；mock 模式用 dataset 上的虚拟 spec
    if ns_feature_specs is None:
        ns_feature_specs = getattr(train_ds, "ns_feature_specs", None)

    # ── 构建模型 ──
    model_kwargs = dict(
        ns_tokenizer_type="auto_split",
        ns_feature_specs=ns_feature_specs,
        total_ns_dim=train_ds.ns_dim,
        L_NS=args.L_NS,
        seq_input_dims=seq_dims,
        timestamp_aware=True,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        task_names=task_names,
        L_S_init=sum(seq_lengths),
    )
    model = OneTrans(**model_kwargs).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{'='*60}")
    print(f"模型参数量: {n_params/1e6:.2f}M")
    print(f"序列维度: {seq_dims}  |  序列长度: {seq_lengths}")
    print(f"d_model={args.d_model}, n_heads={args.n_heads}, n_layers={args.n_layers}, L_NS={args.L_NS}")
    print(f"{'='*60}\n")

    # AdamW 对 Transformer 更稳定，weight_decay 防止过拟合
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8
    )

    # P1-4：Warmup + Cosine Decay LR Scheduler
    steps_per_epoch = args.n_train // args.batch_size if args.data_mode == "mock" else None
    if steps_per_epoch:
        total_steps  = steps_per_epoch * args.epochs
        warmup_steps = min(steps_per_epoch, total_steps // 10)  # 前 10% warmup
        scheduler = get_lr_scheduler(optimizer, warmup_steps, total_steps)
        print(f"LR Scheduler: warmup={warmup_steps} steps, total={total_steps} steps")
    else:
        # parquet IterableDataset 不知道总步数，用 epoch 级别的 cosine
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.1
        )

    # ── 训练循环 ──
    best_ctr_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer,
                                    scheduler if steps_per_epoch else None,
                                    device, args.grad_clip)
        # parquet 模式：epoch 结束后 step scheduler
        if not steps_per_epoch:
            scheduler.step()

        val_metrics = evaluate(model, val_loader, device, task_names)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        # 用第一个任务的 AUC 作为 best model 判断依据（通常是 CTR）
        primary_task = task_names[0]
        primary_auc  = val_metrics.get(f"{primary_task}_auc", 0.0)
        auc_str = "  ".join(
            f"{n.upper()} AUC: {val_metrics.get(f'{n}_auc', 0.0):.4f}"
            for n in task_names
        )
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Loss: {train_metrics['loss']:.4f} | "
            f"{auc_str} | "
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        if primary_auc > best_ctr_auc:
            best_ctr_auc = primary_auc
            if args.save_path:
                save_checkpoint(model, model_kwargs, epoch, val_metrics, args.save_path)
                print(f"  -> 保存最佳 checkpoint  (epoch={epoch}, {task_names[0].upper()} AUC={primary_auc:.4f})")

    print(f"\n训练完成！最佳 {task_names[0].upper()} AUC: {best_ctr_auc:.4f}")
    if args.save_path:
        print(f"\n加载并部署示例：")
        print(f"  python infer.py --checkpoint {args.save_path} --feature_config feature_config.yaml")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="OneTrans 训练脚本")

    # 数据模式
    p.add_argument("--data_mode",  type=str, default="mock",
                   choices=["mock", "parquet"], help="mock=模拟数据  parquet=真实数据")
    p.add_argument("--num_workers", type=int, default=0)

    # mock 模式专用
    p.add_argument("--n_train",     type=int,         default=20000)
    p.add_argument("--n_val",       type=int,         default=4000)
    p.add_argument("--ns_dim",      type=int,         default=160)
    p.add_argument("--seq_lengths", type=int, nargs="+", default=[50, 10, 20])
    p.add_argument("--seq_dims",    type=int, nargs="+", default=None)

    # parquet 模式专用
    p.add_argument("--train_dir",       type=str, default=None, help="训练集 Parquet 目录")
    p.add_argument("--val_dir",         type=str, default=None, help="验证集 Parquet 目录")
    p.add_argument("--feature_config",  type=str, default=None,
                   help="YAML 特征配置文件路径（不指定则用代码内默认字段）")

    # 模型超参
    p.add_argument("--d_model",  type=int,   default=128)
    p.add_argument("--n_heads",  type=int,   default=4)
    p.add_argument("--n_layers", type=int,   default=4)
    p.add_argument("--L_NS",     type=int,   default=8)
    p.add_argument("--dropout",  type=float, default=0.1)

    # 训练超参
    p.add_argument("--epochs",     type=int,   default=5)
    p.add_argument("--batch_size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=0.001)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--save_path",  type=str,   default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
