import argparse
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

from rank_mixer.data import build_dataloader, generate_dataset, load_rows
from rank_mixer.features import RankMixerFeatureConfig, RankMixerFeatureEncoder
from rank_mixer.model import (
    RankMixerModel,
    RankMixerModelConfig,
    compute_auc,
    compute_ndcg_at_k,
    dump_metrics,
    move_batch_to_device,
)


def evaluate(
    model: RankMixerModel,
    dataloader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    ctr_labels: List[int] = []
    cvr_labels: List[int] = []
    ctr_predictions: List[float] = []
    cvr_predictions: List[float] = []
    request_ids: List[int] = []
    criterion = nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            ctr_logits = outputs["ctr_logits"]
            cvr_logits = outputs["cvr_logits"]
            ctr_loss = criterion(ctr_logits, batch["labels"]["ctr"])
            cvr_loss = criterion(cvr_logits, batch["labels"]["cvr"])
            loss = ctr_loss + cvr_loss + outputs["aux_loss"]
            total_loss += float(loss.item()) * len(batch["labels"]["ctr"])
            ctr_labels.extend(batch["labels"]["ctr"].long().cpu().tolist())
            cvr_labels.extend(batch["labels"]["cvr"].long().cpu().tolist())
            ctr_predictions.extend(torch.sigmoid(ctr_logits).cpu().tolist())
            cvr_predictions.extend(torch.sigmoid(cvr_logits).cpu().tolist())
            request_ids.extend(batch["request_id"].cpu().tolist())

    row_count = max(1, len(ctr_labels))
    rank_scores = [ctr * cvr for ctr, cvr in zip(ctr_predictions, cvr_predictions)]
    return {
        "logloss": total_loss / row_count,
        "ctr_auc": compute_auc(ctr_labels, ctr_predictions),
        "cvr_auc": compute_auc(cvr_labels, cvr_predictions),
        "ndcg_at_10": compute_ndcg_at_k(request_ids, cvr_labels, rank_scores, k=10),
    }


def train(
    train_path: Path,
    valid_path: Path,
    feature_config_path: Path,
    model_path: Path,
    metrics_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    model_config: RankMixerModelConfig,
    device_name: str,
) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    train_rows = load_rows(train_path)
    valid_rows = load_rows(valid_path)
    feature_config = RankMixerFeatureConfig.load(feature_config_path)
    encoder = RankMixerFeatureEncoder(feature_config)

    train_loader = build_dataloader(train_rows, encoder=encoder, batch_size=batch_size, shuffle=True)
    valid_loader = build_dataloader(valid_rows, encoder=encoder, batch_size=batch_size, shuffle=False)

    model = RankMixerModel(feature_config=feature_config, model_config=model_config)
    device = torch.device(device_name)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch in train_loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(batch)
            ctr_loss = criterion(outputs["ctr_logits"], batch["labels"]["ctr"])
            cvr_loss = criterion(outputs["cvr_logits"], batch["labels"]["cvr"])
            loss = ctr_loss + cvr_loss + outputs["aux_loss"]
            loss.backward()
            optimizer.step()
            batch_rows = len(batch["labels"]["ctr"])
            total_loss += float(loss.item()) * batch_rows
            total_rows += batch_rows

        train_metrics = {"logloss": total_loss / max(1, total_rows)}
        valid_metrics = evaluate(model, valid_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_logloss": train_metrics["logloss"],
                "valid_logloss": valid_metrics["logloss"],
                "valid_ctr_auc": valid_metrics["ctr_auc"],
                "valid_cvr_auc": valid_metrics["cvr_auc"],
                "valid_ndcg_at_10": valid_metrics["ndcg_at_10"],
            }
        )

    model.save_artifact(model_path)
    metrics = {
        "train_rows": len(train_rows),
        "valid_rows": len(valid_rows),
        "feature_tokens": feature_config.num_tokens,
        "model_config": model_config.to_dict(),
        "history": history,
        "final_valid_metrics": history[-1],
        "paper_alignment": {
            "multi_head_token_mixing": True,
            "per_token_ffn": True,
            "sparse_moe": model_config.use_sparse_moe,
            "mean_pooling": True,
        },
    }
    dump_metrics(metrics_path, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ByteDance RankMixer style model.")
    parser.add_argument("--train-path", type=Path, default=Path("artifacts/data/train.csv"))
    parser.add_argument("--valid-path", type=Path, default=Path("artifacts/data/valid.csv"))
    parser.add_argument("--feature-config", type=Path, default=Path("feature_config.json"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model/rankmixer.pt"))
    parser.add_argument("--metrics-path", type=Path, default=Path("artifacts/model/metrics.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--use-sparse-moe", action="store_true")
    parser.add_argument("--model-dim", type=int, default=320)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--ff-multiplier", type=int, default=4)
    parser.add_argument("--numeric-embedding-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--sparse-topk", type=int, default=2)
    parser.add_argument("--moe-l1-weight", type=float, default=1e-4)
    parser.add_argument("--disable-dtsi", action="store_true")
    parser.add_argument("--generate-data", action="store_true")
    args = parser.parse_args()

    if args.generate_data:
        generate_dataset(args.train_path, num_requests=900, seed=42)
        generate_dataset(args.valid_path, num_requests=220, seed=99)

    model_config = RankMixerModelConfig(
        model_dim=args.model_dim,
        num_layers=args.num_layers,
        ff_multiplier=args.ff_multiplier,
        dropout=args.dropout,
        numeric_embedding_dim=args.numeric_embedding_dim,
        use_sparse_moe=args.use_sparse_moe,
        num_experts=args.num_experts,
        sparse_topk=args.sparse_topk,
        moe_l1_weight=args.moe_l1_weight,
        dense_training_sparse_inference=not args.disable_dtsi,
    )

    train(
        train_path=args.train_path,
        valid_path=args.valid_path,
        feature_config_path=args.feature_config,
        model_path=args.model_path,
        metrics_path=args.metrics_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_config=model_config,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
