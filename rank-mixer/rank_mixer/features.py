import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch


@dataclass
class CategoricalFieldSpec:
    name: str
    num_buckets: int
    embedding_dim: int
    raw_type: str = "string"
    model_type: str = "categorical_id"
    processor: str = "hash_bucket_embedding"
    description: str = ""


@dataclass
class NumericFieldSpec:
    name: str
    raw_type: str = "float"
    model_type: str = "numeric"
    processor: str = "linear_projection"
    description: str = ""


@dataclass
class FeatureGroupSpec:
    name: str
    categorical: List[CategoricalFieldSpec]
    numeric: List[NumericFieldSpec]


@dataclass
class RankMixerFeatureConfig:
    request_id_field: str
    item_id_field: str
    label_fields: Dict[str, str]
    groups: List[FeatureGroupSpec]

    @property
    def num_tokens(self) -> int:
        return len(self.groups)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "RankMixerFeatureConfig":
        groups: List[FeatureGroupSpec] = []
        for group_payload in payload["groups"]:
            groups.append(
                FeatureGroupSpec(
                    name=str(group_payload["name"]),
                    categorical=[
                        CategoricalFieldSpec(
                            name=str(field["name"]),
                            num_buckets=int(field["num_buckets"]),
                            embedding_dim=int(field["embedding_dim"]),
                            raw_type=str(field.get("raw_type", "string")),
                            model_type=str(field.get("model_type", "categorical_id")),
                            processor=str(field.get("processor", "hash_bucket_embedding")),
                            description=str(field.get("description", "")),
                        )
                        for field in group_payload.get("categorical", [])
                    ],
                    numeric=[
                        NumericFieldSpec(
                            name=str(field["name"]) if isinstance(field, dict) else str(field),
                            raw_type=str(field.get("raw_type", "float")) if isinstance(field, dict) else "float",
                            model_type=str(field.get("model_type", "numeric")) if isinstance(field, dict) else "numeric",
                            processor=str(field.get("processor", "linear_projection"))
                            if isinstance(field, dict)
                            else "linear_projection",
                            description=str(field.get("description", "")) if isinstance(field, dict) else "",
                        )
                        for field in group_payload.get("numeric", [])
                    ],
                )
            )
        return cls(
            request_id_field=str(payload["request_id_field"]),
            item_id_field=str(payload["item_id_field"]),
            label_fields={str(key): str(value) for key, value in payload["label_fields"].items()},
            groups=groups,
        )

    @classmethod
    def load(cls, path: Path) -> "RankMixerFeatureConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def stable_hash_bucket(text: str, num_buckets: int) -> int:
    if num_buckets <= 0:
        raise ValueError("num_buckets must be positive.")
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % num_buckets


class RankMixerFeatureEncoder:
    def __init__(self, config: RankMixerFeatureConfig):
        self.config = config

    def encode_row(self, row: Dict[str, object]) -> Dict[str, object]:
        categorical: Dict[str, int] = {}
        numeric: Dict[str, float] = {}
        for group in self.config.groups:
            for field in group.categorical:
                value = str(row[field.name])
                categorical[field.name] = stable_hash_bucket(f"{field.name}={value}", field.num_buckets)
            for field in group.numeric:
                numeric[field.name] = float(row[field.name])

        return {
            "request_id": int(row[self.config.request_id_field]),
            "item_id": int(row[self.config.item_id_field]),
            "labels": {
                task_name: float(row[field_name]) for task_name, field_name in self.config.label_fields.items()
            },
            "categorical": categorical,
            "numeric": numeric,
        }

    def encode_rows(self, rows: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
        return [self.encode_row(row) for row in rows]

    def collate(self, rows: List[Dict[str, object]]) -> Dict[str, object]:
        encoded_rows = self.encode_rows(rows)
        categorical_batch: Dict[str, torch.Tensor] = {}
        numeric_batch: Dict[str, torch.Tensor] = {}

        for group in self.config.groups:
            for field in group.categorical:
                categorical_batch[field.name] = torch.tensor(
                    [int(row["categorical"][field.name]) for row in encoded_rows],
                    dtype=torch.long,
                )
            for field in group.numeric:
                numeric_batch[field.name] = torch.tensor(
                    [float(row["numeric"][field.name]) for row in encoded_rows],
                    dtype=torch.float32,
                )

        return {
            "request_id": torch.tensor([int(row["request_id"]) for row in encoded_rows], dtype=torch.long),
            "item_id": torch.tensor([int(row["item_id"]) for row in encoded_rows], dtype=torch.long),
            "labels": {
                task_name: torch.tensor(
                    [float(row["labels"][task_name]) for row in encoded_rows],
                    dtype=torch.float32,
                )
                for task_name in self.config.label_fields
            },
            "categorical": categorical_batch,
            "numeric": numeric_batch,
        }
