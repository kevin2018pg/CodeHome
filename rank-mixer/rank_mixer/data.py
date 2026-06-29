import csv
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from rank_mixer.features import RankMixerFeatureEncoder


USER_SEGMENTS = ["new", "active", "high_value", "churn_risk"]
MEMBERSHIP_TIERS = ["free", "silver", "gold", "vip"]
DEVICES = ["ios", "android", "mobile_web", "desktop"]
TRAFFIC_SOURCES = ["search", "push", "homepage", "ads", "affiliate"]
BRAND_TIERS = ["mass", "premium", "luxury"]
CONTENT_TYPES = ["short_video", "live", "mall", "ad"]


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def build_record(rng: random.Random, request_id: int, position: int, item_id: int) -> Dict[str, object]:
    unionid = rng.randint(1, 70000)
    author_id = rng.randint(1, 15000)
    content_id = item_id
    user_segment = rng.choices(USER_SEGMENTS, weights=[0.18, 0.47, 0.20, 0.15])[0]
    membership_tier = rng.choices(MEMBERSHIP_TIERS, weights=[0.50, 0.25, 0.18, 0.07])[0]
    device = rng.choices(DEVICES, weights=[0.30, 0.35, 0.20, 0.15])[0]
    traffic_source = rng.choices(TRAFFIC_SOURCES, weights=[0.28, 0.18, 0.27, 0.14, 0.13])[0]
    brand_tier = rng.choices(BRAND_TIERS, weights=[0.65, 0.25, 0.10])[0]
    content_type = rng.choices(CONTENT_TYPES, weights=[0.62, 0.10, 0.16, 0.12])[0]

    user_age = rng.randint(18, 60)
    activity_7d = max(1, int(rng.gauss(14, 7)))
    historical_ctr = min(max(rng.gauss(0.065, 0.022), 0.003), 0.35)
    historical_cvr = min(max(rng.gauss(0.015, 0.008), 0.001), 0.10)
    user_pay_tendency_30d = round(min(max(rng.gauss(0.22, 0.15), 0.0), 1.0), 4)
    price = round(max(0.0, rng.lognormvariate(3.2, 0.7)), 2)
    quality_score = round(min(max(rng.gauss(0.70, 0.12), 0.05), 0.99), 4)
    seller_score = round(min(max(rng.gauss(4.50, 0.25), 3.2), 5.0), 4)
    hour_of_day = rng.randint(0, 23)
    is_weekend = 1 if rng.random() < 0.28 else 0
    network_type = rng.choices([2, 3, 4, 5], weights=[0.05, 0.12, 0.60, 0.23])[0]
    session_depth = rng.randint(1, 40)
    seq_click_rate_7d = round(min(max(rng.gauss(0.12, 0.05), 0.0), 0.9), 4)
    seq_watch_depth_7d = round(min(max(rng.gauss(0.34, 0.16), 0.0), 1.0), 4)
    seq_skip_rate_7d = round(min(max(rng.gauss(0.41, 0.16), 0.0), 1.0), 4)
    seq_ecom_exposure_7d = round(min(max(rng.gauss(18, 10), 0.0), 120.0), 4)
    seq_live_watch_rate_7d = round(min(max(rng.gauss(0.18, 0.12), 0.0), 1.0), 4)
    cross_ctr_gap = round(min(max(rng.gauss(0.0, 0.22), -1.0), 1.0), 4)
    cross_price_gap = round(min(max(rng.gauss(0.0, 0.35), -2.0), 2.0), 4)
    cross_user_author_affinity = round(min(max(rng.gauss(0.18, 0.20), 0.0), 1.0), 4)
    cross_user_item_similarity = round(min(max(rng.gauss(0.20, 0.18), 0.0), 1.0), 4)
    cross_query_item_intent = round(min(max(rng.gauss(0.16, 0.16), 0.0), 1.0), 4)
    candidate_score = round(min(max(rng.gauss(0.52, 0.18), 0.0), 1.0), 4)
    item_ctr_7d = round(min(max(rng.gauss(0.08, 0.03), 0.001), 0.4), 4)
    item_cvr_7d = round(min(max(rng.gauss(0.012, 0.007), 0.0005), 0.12), 4)

    segment_bias = {"new": -0.10, "active": 0.03, "high_value": 0.16, "churn_risk": -0.08}[user_segment]
    membership_bias = {"free": -0.05, "silver": 0.02, "gold": 0.08, "vip": 0.15}[membership_tier]
    source_bias = {"search": 0.08, "push": 0.03, "homepage": 0.02, "ads": -0.07, "affiliate": -0.04}[traffic_source]
    content_bias = {"short_video": 0.06, "live": 0.03, "mall": -0.02, "ad": -0.08}[content_type]
    device_bias = 0.04 if device in ("ios", "android") else -0.01
    hour_bias = 0.05 if hour_of_day in (12, 13, 20, 21, 22) else -0.02
    weekend_bias = 0.04 if is_weekend and content_type in ("live", "mall") else 0.0

    latent_relevance = (
        1.40 * historical_ctr
        + 1.65 * historical_cvr
        + 0.80 * quality_score
        + 0.18 * (seller_score - 4.0)
        + 0.05 * math.log1p(activity_7d)
        + 0.90 * seq_click_rate_7d
        + 0.40 * seq_watch_depth_7d
        - 0.35 * seq_skip_rate_7d
        + 0.12 * math.log1p(seq_ecom_exposure_7d)
        + 0.25 * seq_live_watch_rate_7d
        + 0.60 * cross_user_author_affinity
        + 0.45 * cross_user_item_similarity
        + 0.35 * cross_query_item_intent
        + 0.35 * candidate_score
        + 0.90 * item_ctr_7d
        + 0.40 * item_cvr_7d
        - 0.06 * position
        + segment_bias
        + membership_bias
        + source_bias
        + content_bias
        + device_bias
        + hour_bias
        + weekend_bias
        + rng.gauss(0.0, 0.15)
    )

    ctr_logit = (
        -2.60
        + 2.20 * sigmoid(latent_relevance)
        + 0.45 * candidate_score
        + 0.30 * cross_ctr_gap
        + 0.22 * cross_query_item_intent
        - 0.06 * abs(cross_price_gap)
        - 0.04 * math.log1p(session_depth)
        + rng.gauss(0.0, 0.14)
    )
    ctr_label = 1 if rng.random() < sigmoid(ctr_logit) else 0

    cvr_logit = (
        -3.60
        + 1.00 * sigmoid(latent_relevance)
        + 0.55 * historical_cvr
        + 0.70 * user_pay_tendency_30d
        + 0.45 * item_cvr_7d
        + 0.18 * cross_user_item_similarity
        - 0.020 * price
        + rng.gauss(0.0, 0.12)
    )
    cvr_prob = sigmoid(cvr_logit) * ctr_label
    cvr_label = 1 if rng.random() < cvr_prob else 0

    return {
        "request_id": request_id,
        "item_id": item_id,
        "ctr_label": ctr_label,
        "cvr_label": cvr_label,
        "position": position,
        "unionid": unionid,
        "membership_tier": membership_tier,
        "device": device,
        "user_segment": user_segment,
        "user_age": user_age,
        "activity_7d": activity_7d,
        "historical_ctr": round(historical_ctr, 6),
        "historical_cvr": round(historical_cvr, 6),
        "user_pay_tendency_30d": user_pay_tendency_30d,
        "author_id": author_id,
        "content_id": content_id,
        "brand_tier": brand_tier,
        "content_type": content_type,
        "price": price,
        "quality_score": quality_score,
        "seller_score": seller_score,
        "item_ctr_7d": item_ctr_7d,
        "item_cvr_7d": item_cvr_7d,
        "hour_of_day": hour_of_day,
        "is_weekend": is_weekend,
        "network_type": network_type,
        "session_depth": session_depth,
        "traffic_source": traffic_source,
        "seq_click_rate_7d": seq_click_rate_7d,
        "seq_watch_depth_7d": seq_watch_depth_7d,
        "seq_skip_rate_7d": seq_skip_rate_7d,
        "seq_ecom_exposure_7d": seq_ecom_exposure_7d,
        "seq_live_watch_rate_7d": seq_live_watch_rate_7d,
        "cross_ctr_gap": cross_ctr_gap,
        "cross_price_gap": cross_price_gap,
        "cross_user_author_affinity": cross_user_author_affinity,
        "cross_user_item_similarity": cross_user_item_similarity,
        "cross_query_item_intent": cross_query_item_intent,
        "candidate_score": candidate_score,
    }


def generate_dataset(
    output_path: Path,
    num_requests: int,
    min_items_per_request: int = 20,
    max_items_per_request: int = 50,
    seed: int = 42,
) -> Tuple[int, int]:
    if num_requests <= 0:
        raise ValueError("num_requests must be positive.")
    if min_items_per_request <= 0 or max_items_per_request < min_items_per_request:
        raise ValueError("Invalid item count range.")

    ensure_dir(output_path.parent)
    rng = random.Random(seed)
    total_rows = 0
    next_item_id = 1
    fieldnames = [
        "request_id",
        "item_id",
        "ctr_label",
        "cvr_label",
        "position",
        "unionid",
        "membership_tier",
        "device",
        "user_segment",
        "user_age",
        "activity_7d",
        "historical_ctr",
        "historical_cvr",
        "user_pay_tendency_30d",
        "author_id",
        "content_id",
        "brand_tier",
        "content_type",
        "price",
        "quality_score",
        "seller_score",
        "item_ctr_7d",
        "item_cvr_7d",
        "hour_of_day",
        "is_weekend",
        "network_type",
        "session_depth",
        "traffic_source",
        "seq_click_rate_7d",
        "seq_watch_depth_7d",
        "seq_skip_rate_7d",
        "seq_ecom_exposure_7d",
        "seq_live_watch_rate_7d",
        "cross_ctr_gap",
        "cross_price_gap",
        "cross_user_author_affinity",
        "cross_user_item_similarity",
        "cross_query_item_intent",
        "candidate_score",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for request_id in range(1, num_requests + 1):
            item_count = rng.randint(min_items_per_request, max_items_per_request)
            rows = [build_record(rng, request_id, position, next_item_id + position - 1) for position in range(1, item_count + 1)]
            next_item_id += item_count
            if not any(int(row["ctr_label"]) == 1 for row in rows):
                best_row = max(rows, key=lambda row: float(row["candidate_score"]) + float(row["historical_ctr"]))
                best_row["ctr_label"] = 1
            for row in rows:
                writer.writerow(row)
            total_rows += len(rows)
    return num_requests, total_rows


def load_rows(csv_path: Path) -> List[Dict[str, object]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {csv_path}")

    rows: List[Dict[str, object]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is missing a header row: {csv_path}")
        for raw in reader:
            row: Dict[str, object] = {}
            for key, value in raw.items():
                if key in {
                    "request_id",
                    "item_id",
                    "ctr_label",
                    "cvr_label",
                    "position",
                    "unionid",
                    "user_age",
                    "activity_7d",
                    "author_id",
                    "content_id",
                    "hour_of_day",
                    "is_weekend",
                    "network_type",
                    "session_depth",
                }:
                    row[key] = int(value)
                elif key in {
                    "historical_ctr",
                    "historical_cvr",
                    "user_pay_tendency_30d",
                    "price",
                    "quality_score",
                    "seller_score",
                    "item_ctr_7d",
                    "item_cvr_7d",
                    "seq_click_rate_7d",
                    "seq_watch_depth_7d",
                    "seq_skip_rate_7d",
                    "seq_ecom_exposure_7d",
                    "seq_live_watch_rate_7d",
                    "cross_ctr_gap",
                    "cross_price_gap",
                    "cross_user_author_affinity",
                    "cross_user_item_similarity",
                    "cross_query_item_intent",
                    "candidate_score",
                    "ctr_pred",
                    "cvr_pred",
                    "rank_mixer_score",
                }:
                    row[key] = float(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def group_by_request(rows: Sequence[Dict[str, object]]) -> Dict[int, List[Dict[str, object]]]:
    grouped: Dict[int, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["request_id"])].append(dict(row))
    return grouped


class RankDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, object]]):
        self.rows = list(rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        return self.rows[index]


def build_dataloader(
    rows: Sequence[Dict[str, object]],
    encoder: RankMixerFeatureEncoder,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = RankDataset(rows)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=encoder.collate,
    )
