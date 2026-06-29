"""
OneTrans 推理脚本

功能：
  1. 从 checkpoint 恢复最佳模型
  2. 批量推理 Parquet 数据，输出各任务预测分数
  3. 可选：导出 TorchScript（.pt）用于 C++/Java 服务部署

用法：
  # 批量推理，结果写入 CSV
  python infer.py \
      --checkpoint  ./best_model.pt \
      --data_dir    ./data/parquet/test \
      --feature_config feature_config.yaml \
      --output      ./predictions.csv

  # 导出 TorchScript（部署用）
  python infer.py \
      --checkpoint  ./best_model.pt \
      --feature_config feature_config.yaml \
      --export_ts   ./model_deploy.pt
"""

import argparse
import os
from typing import Dict, List

import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from data import get_dataloader, load_feature_config, LABEL_COLS
from train import load_checkpoint


# ---------------------------------------------------------------------------
# 推理主逻辑
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader,
    device: str,
    task_names: List[str],
    label_cols: List[str],
) -> pd.DataFrame:
    """
    批量推理，返回包含预测分数（和标签，如果有）的 DataFrame。
    """
    model.eval()
    records = []

    for batch in tqdm(loader, desc="推理中"):
        ns_inputs = {k: v.to(device) for k, v in batch["ns_inputs"].items()}
        seqs      = [s.to(device) for s in batch["sequences"]]
        ts        = [t.to(device) for t in batch["timestamps"]]
        masks     = [m.to(device) for m in batch["seq_masks"]]

        preds, _ = model(ns_inputs, seqs, ts, masks)

        B = seqs[0].size(0)
        for i in range(B):
            row = {}
            for name in task_names:
                # model 返回 logit，推理时转为概率
                row[f"{name}_score"] = torch.sigmoid(preds[name][i]).item()
            # 如果 batch 里有标签，一并记录（方便离线评估）
            for col in label_cols:
                if col in batch:
                    row[col] = batch[col][i].item()
            records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# TorchScript 导出（用于 C++/Java 服务）
# ---------------------------------------------------------------------------

def export_torchscript(model: torch.nn.Module, export_path: str, example_batch: Dict):
    """
    将模型导出为 TorchScript（traced 模式）。

    注意：TorchScript 不支持 Python dict 输入，这里先把 ns_inputs 拼成 tensor 再 trace。
    实际部署时推荐在服务层做特征拼接，或用 ONNX 导出。
    """
    model.eval()
    device = next(model.parameters()).device

    ns_inputs = {k: v.to(device) for k, v in example_batch["ns_inputs"].items()}
    seqs      = [s.to(device) for s in example_batch["sequences"]]
    ts        = [t.to(device) for t in example_batch["timestamps"]]
    masks     = [m.to(device) for m in example_batch["seq_masks"]]

    try:
        traced = torch.jit.trace(
            model,
            (ns_inputs, seqs, ts, masks),
            strict=False,
        )
        traced.save(export_path)
        print(f"[TorchScript] 导出成功: {export_path}")
        print(f"  C++ 加载示例：")
        print(f"    auto model = torch::jit::load(\"{export_path}\");")
    except Exception as e:
        print(f"[TorchScript] 导出失败（dict 输入限制）: {e}")
        print("  建议改用 ONNX 导出，或在服务层将特征拼成 tensor 后再 trace")
        _suggest_onnx_export()


def _suggest_onnx_export():
    print("""
  ONNX 导出参考（需要 pip install onnx onnxruntime）：
    import torch.onnx
    # 先把 ns_inputs dict 改成 tensor 输入的 wrapper 模型
    torch.onnx.export(
        model_wrapper,
        (ns_tensor, seq_tensor, ts_tensor, mask_tensor),
        "model.onnx",
        opset_version=17,
        input_names=["ns", "seq", "ts", "mask"],
        output_names=["ctr_score", "cvr_score"],
        dynamic_axes={"ns": {0: "batch"}, "seq": {0: "batch"}},
    )
""")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")

    # ── 加载模型 ──
    model = load_checkpoint(args.checkpoint, device)
    # task_names 直接从模型属性读取，无需重复加载 checkpoint
    task_names = model.task_names

    # ── 导出 TorchScript（可选）──
    if args.export_ts:
        if args.output:
            print("[提示] 同时指定了 --export_ts 和 --output，导出完成后不会执行批量推理")
            print("       如需批量推理，请单独运行不带 --export_ts 的命令")
        if not args.data_dir:
            print("导出 TorchScript 需要提供 --data_dir 以获取示例输入")
        else:
            loader, _ = get_dataloader(
                mode="parquet", shuffle=False,
                data_dir=args.data_dir,
                feature_config=args.feature_config,
                batch_size=4, num_workers=0,
            )
            example_batch = next(iter(loader))
            export_torchscript(model, args.export_ts, example_batch)
        return

    # ── 批量推理 ──
    assert args.data_dir, "批量推理需要 --data_dir"

    if args.feature_config:
        _, _, label_cols = load_feature_config(args.feature_config)
    else:
        label_cols = LABEL_COLS

    loader, _ = get_dataloader(
        mode="parquet", shuffle=False,
        data_dir=args.data_dir,
        feature_config=args.feature_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    df = run_inference(model, loader, device, task_names, label_cols)

    # ── 输出结果 ──
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\n预测结果已保存到: {args.output}")
        print(f"共 {len(df)} 条，列: {df.columns.tolist()}")
    else:
        print(df.head(10).to_string())

    # ── 如果有标签，顺便计算 AUC ──
    for name in task_names:
        label_col = f"{name}_label"
        score_col = f"{name}_score"
        if label_col in df.columns and score_col in df.columns:
            try:
                auc = roc_auc_score(df[label_col], df[score_col])
                print(f"{name} AUC: {auc:.4f}")
            except ValueError:
                pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OneTrans 推理脚本")
    p.add_argument("--checkpoint",     type=str, required=True,  help="checkpoint 文件路径（.pt）")
    p.add_argument("--data_dir",       type=str, default=None,   help="推理数据目录（Parquet 文件）")
    p.add_argument("--feature_config", type=str, default=None,   help="YAML 特征配置文件")
    p.add_argument("--output",         type=str, default=None,   help="预测结果输出路径（.csv）")
    p.add_argument("--export_ts",      type=str, default=None,   help="导出 TorchScript 到此路径（.pt）")
    p.add_argument("--batch_size",     type=int, default=1024)
    p.add_argument("--num_workers",    type=int, default=0)
    args = p.parse_args()
    main(args)
