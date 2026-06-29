import csv
import json
from pathlib import Path

from infer import infer
from rank_mixer.data import generate_dataset, group_by_request, load_rows
from rank_mixer.model import RankMixerModel, RankMixerModelConfig, compute_auc, compute_ndcg_at_k
from train import train


def verify_predictions(prediction_path: Path) -> None:
    rows = load_rows(prediction_path)
    grouped = group_by_request(rows)
    assert grouped, "Prediction file is empty."
    for request_id, items in grouped.items():
        scores = [float(item["rank_mixer_score"]) for item in items]
        assert scores == sorted(scores, reverse=True), f"Scores are not sorted for request {request_id}"
        assert all(0.0 <= score <= 1.0 for score in scores), "Predicted score must be a probability."
        assert all(0.0 <= float(item["ctr_pred"]) <= 1.0 for item in items), "CTR prediction must be a probability."
        assert all(0.0 <= float(item["cvr_pred"]) <= 1.0 for item in items), "CVR prediction must be a probability."


def main() -> None:
    root = Path("artifacts")
    data_dir = root / "data"
    model_dir = root / "model"
    pred_dir = root / "predictions"

    train_path = data_dir / "train.csv"
    valid_path = data_dir / "valid.csv"
    test_path = data_dir / "test.csv"
    model_path = model_dir / "rankmixer.pt"
    metrics_path = model_dir / "metrics.json"
    sparse_model_path = model_dir / "rankmixer_sparse.pt"
    sparse_metrics_path = model_dir / "metrics_sparse.json"
    prediction_path = pred_dir / "test_predictions.csv"
    sparse_prediction_path = pred_dir / "test_predictions_sparse.csv"
    feature_config_path = Path("feature_config.json")

    generate_dataset(train_path, num_requests=180, seed=42)
    generate_dataset(valid_path, num_requests=64, seed=99)
    generate_dataset(test_path, num_requests=48, seed=123)

    train(
        train_path=train_path,
        valid_path=valid_path,
        feature_config_path=feature_config_path,
        model_path=model_path,
        metrics_path=metrics_path,
        epochs=2,
        batch_size=256,
        learning_rate=1e-3,
        model_config=RankMixerModelConfig(),
        device_name="cpu",
    )
    infer(test_path, model_path, prediction_path, batch_size=512, device_name="cpu")
    verify_predictions(prediction_path)

    train(
        train_path=train_path,
        valid_path=valid_path,
        feature_config_path=feature_config_path,
        model_path=sparse_model_path,
        metrics_path=sparse_metrics_path,
        epochs=1,
        batch_size=256,
        learning_rate=1e-3,
        model_config=RankMixerModelConfig(use_sparse_moe=True),
        device_name="cpu",
    )
    infer(test_path, sparse_model_path, sparse_prediction_path, batch_size=512, device_name="cpu")
    verify_predictions(sparse_prediction_path)

    model = RankMixerModel.load_artifact(model_path)
    test_rows = load_rows(test_path)
    prediction_rows = load_rows(prediction_path)
    ctr_labels = [int(row["ctr_label"]) for row in prediction_rows]
    cvr_labels = [int(row["cvr_label"]) for row in prediction_rows]
    ctr_predictions = [float(row["ctr_pred"]) for row in prediction_rows]
    cvr_predictions = [float(row["cvr_pred"]) for row in prediction_rows]
    predictions = [float(row["rank_mixer_score"]) for row in prediction_rows]
    request_ids = [int(row["request_id"]) for row in prediction_rows]
    test_metrics = {
        "test_rows": len(test_rows),
        "ctr_auc": compute_auc(ctr_labels, ctr_predictions),
        "cvr_auc": compute_auc(cvr_labels, cvr_predictions),
        "ndcg_at_10": compute_ndcg_at_k(request_ids, cvr_labels, predictions, k=10),
        "loaded_model_tokens": model.feature_config.num_tokens,
    }

    summary_path = root / "verification_summary.json"
    summary = {
        "model_metrics": json.loads(metrics_path.read_text(encoding="utf-8")),
        "sparse_model_metrics": json.loads(sparse_metrics_path.read_text(encoding="utf-8")),
        "test_metrics": test_metrics,
        "prediction_file": str(prediction_path),
        "sparse_prediction_file": str(sparse_prediction_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    top_preview_path = root / "top_predictions_preview.csv"
    with top_preview_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "request_id",
            "item_id",
            "ctr_label",
            "cvr_label",
            "position",
            "ctr_pred",
            "cvr_pred",
            "rank_mixer_score",
            "candidate_score",
            "historical_ctr",
            "seq_click_rate_7d",
            "content_type",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in prediction_rows[:50]:
            writer.writerow({key: row[key] for key in fieldnames})

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
