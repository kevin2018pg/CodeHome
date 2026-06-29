import argparse
import csv
from pathlib import Path
from typing import Dict, List

import torch

from rank_mixer.data import build_dataloader, load_rows
from rank_mixer.features import RankMixerFeatureEncoder
from rank_mixer.model import RankMixerModel, move_batch_to_device


def infer(
    input_path: Path,
    model_path: Path,
    output_path: Path,
    batch_size: int = 1024,
    device_name: str = "cpu",
) -> List[Dict[str, object]]:
    rows = load_rows(input_path)
    if not rows:
        raise ValueError(f"No rows found in input file: {input_path}")

    model = RankMixerModel.load_artifact(model_path, map_location=device_name)
    encoder = RankMixerFeatureEncoder(model.feature_config)
    dataloader = build_dataloader(rows, encoder=encoder, batch_size=batch_size, shuffle=False)
    device = torch.device(device_name)
    model.to(device)
    model.eval()

    predictions: List[float] = []
    ctr_predictions: List[float] = []
    cvr_predictions: List[float] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            outputs = model(batch)
            ctr_probs = torch.sigmoid(outputs["ctr_logits"]).cpu().tolist()
            cvr_probs = torch.sigmoid(outputs["cvr_logits"]).cpu().tolist()
            ctr_predictions.extend(ctr_probs)
            cvr_predictions.extend(cvr_probs)
            predictions.extend([ctr * cvr for ctr, cvr in zip(ctr_probs, cvr_probs)])

    enriched_rows: List[Dict[str, object]] = []
    for row, ctr_score, cvr_score, score in zip(rows, ctr_predictions, cvr_predictions, predictions):
        enriched = dict(row)
        enriched["ctr_pred"] = round(ctr_score, 6)
        enriched["cvr_pred"] = round(cvr_score, 6)
        enriched["rank_mixer_score"] = round(score, 6)
        enriched_rows.append(enriched)

    enriched_rows.sort(
        key=lambda row: (
            int(row["request_id"]),
            -float(row["rank_mixer_score"]),
            int(row.get("position", 0)),
            int(row["item_id"]),
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) + ["ctr_pred", "cvr_pred", "rank_mixer_score"]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched_rows)
    return enriched_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ByteDance RankMixer style batch inference.")
    parser.add_argument("--input-path", type=Path, default=Path("artifacts/data/test.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("artifacts/model/rankmixer.pt"))
    parser.add_argument("--output-path", type=Path, default=Path("artifacts/predictions/test_predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    infer(
        input_path=args.input_path,
        model_path=args.model_path,
        output_path=args.output_path,
        batch_size=args.batch_size,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
