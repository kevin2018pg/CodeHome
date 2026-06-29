import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn

from rank_mixer.features import RankMixerFeatureConfig


@dataclass
class RankMixerModelConfig:
    model_dim: int = 320
    num_layers: int = 2
    ff_multiplier: int = 4
    dropout: float = 0.1
    numeric_embedding_dim: int = 16
    use_sparse_moe: bool = False
    num_experts: int = 4
    sparse_topk: int = 2
    moe_l1_weight: float = 1e-4
    dense_training_sparse_inference: bool = True

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, object]) -> "RankMixerModelConfig":
        return cls(
            model_dim=int(payload["model_dim"]),
            num_layers=int(payload["num_layers"]),
            ff_multiplier=int(payload["ff_multiplier"]),
            dropout=float(payload["dropout"]),
            numeric_embedding_dim=int(payload["numeric_embedding_dim"]),
            use_sparse_moe=bool(payload["use_sparse_moe"]),
            num_experts=int(payload["num_experts"]),
            sparse_topk=int(payload["sparse_topk"]),
            moe_l1_weight=float(payload["moe_l1_weight"]),
            dense_training_sparse_inference=bool(payload.get("dense_training_sparse_inference", True)),
        )


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


class MultiHeadTokenMixing(nn.Module):
    def __init__(self, num_tokens: int, model_dim: int):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive.")
        if model_dim % num_tokens != 0:
            raise ValueError("model_dim must be divisible by num_tokens.")
        self.num_tokens = num_tokens
        self.model_dim = model_dim
        self.head_dim = model_dim // num_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_tokens, model_dim = x.shape
        if num_tokens != self.num_tokens or model_dim != self.model_dim:
            raise ValueError("Unexpected token shape for MultiHeadTokenMixing.")
        mixed = x.view(batch_size, num_tokens, self.num_tokens, self.head_dim)
        mixed = mixed.permute(0, 2, 1, 3).contiguous()
        return mixed.view(batch_size, self.num_tokens, self.model_dim)


class DensePerTokenFFN(nn.Module):
    def __init__(self, num_tokens: int, model_dim: int, ff_multiplier: int, dropout: float):
        super().__init__()
        hidden_dim = model_dim * ff_multiplier
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(model_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, model_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_tokens)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = [ffn(x[:, index, :]) for index, ffn in enumerate(self.ffns)]
        return torch.stack(outputs, dim=1), x.new_tensor(0.0)


class SparsePerTokenMoE(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        model_dim: int,
        ff_multiplier: int,
        dropout: float,
        num_experts: int,
        sparse_topk: int,
        l1_weight: float,
        dense_training_sparse_inference: bool,
    ):
        super().__init__()
        hidden_dim = model_dim * ff_multiplier
        self.num_experts = num_experts
        self.sparse_topk = sparse_topk
        self.l1_weight = l1_weight
        self.dense_training_sparse_inference = dense_training_sparse_inference
        self.routers = nn.ModuleList([nn.Linear(model_dim, num_experts) for _ in range(num_tokens)])
        self.experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(model_dim, hidden_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim, model_dim),
                            nn.Dropout(dropout),
                        )
                        for _ in range(num_experts)
                    ]
                )
                for _ in range(num_tokens)
            ]
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        token_outputs: List[torch.Tensor] = []
        total_penalty = x.new_tensor(0.0)
        for token_index, (router, experts) in enumerate(zip(self.routers, self.experts)):
            token_state = x[:, token_index, :]
            gates = torch.relu(router(token_state))
            use_sparse_routing = (
                self.sparse_topk < self.num_experts
                and (not self.training or not self.dense_training_sparse_inference)
            )
            if use_sparse_routing:
                topk_values, topk_indices = torch.topk(gates, k=self.sparse_topk, dim=-1)
                sparse_gates = torch.zeros_like(gates)
                sparse_gates.scatter_(1, topk_indices, topk_values)
                gates = sparse_gates
            total_penalty = total_penalty + gates.mean()
            gate_sums = gates.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            normalized_gates = gates / gate_sums
            expert_outputs = torch.stack([expert(token_state) for expert in experts], dim=1)
            token_outputs.append(torch.sum(expert_outputs * normalized_gates.unsqueeze(-1), dim=1))
        return torch.stack(token_outputs, dim=1), total_penalty * self.l1_weight


class RankMixerBlock(nn.Module):
    def __init__(self, num_tokens: int, config: RankMixerModelConfig):
        super().__init__()
        self.token_mixing = MultiHeadTokenMixing(num_tokens=num_tokens, model_dim=config.model_dim)
        self.norm1 = nn.LayerNorm(config.model_dim)
        self.norm2 = nn.LayerNorm(config.model_dim)
        if config.use_sparse_moe:
            self.pffn = SparsePerTokenMoE(
                num_tokens=num_tokens,
                model_dim=config.model_dim,
                ff_multiplier=config.ff_multiplier,
                dropout=config.dropout,
                num_experts=config.num_experts,
                sparse_topk=config.sparse_topk,
                l1_weight=config.moe_l1_weight,
                dense_training_sparse_inference=config.dense_training_sparse_inference,
            )
        else:
            self.pffn = DensePerTokenFFN(
                num_tokens=num_tokens,
                model_dim=config.model_dim,
                ff_multiplier=config.ff_multiplier,
                dropout=config.dropout,
            )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed = self.token_mixing(x)
        x = self.norm1(mixed + x)
        ffn_out, aux_loss = self.pffn(x)
        x = self.norm2(ffn_out + x)
        return x, aux_loss


class RankMixerModel(nn.Module):
    def __init__(self, feature_config: RankMixerFeatureConfig, model_config: RankMixerModelConfig):
        super().__init__()
        if feature_config.num_tokens <= 0:
            raise ValueError("RankMixer requires at least one feature group.")
        self.feature_config = feature_config
        self.model_config = model_config

        self.categorical_embeddings = nn.ModuleDict()
        self.numeric_projections = nn.ModuleDict()
        self.group_projectors = nn.ModuleDict()

        for group in feature_config.groups:
            input_dim = 0
            for field in group.categorical:
                self.categorical_embeddings[field.name] = nn.Embedding(field.num_buckets, field.embedding_dim)
                input_dim += field.embedding_dim
            for field in group.numeric:
                self.numeric_projections[field.name] = nn.Linear(1, model_config.numeric_embedding_dim)
                input_dim += model_config.numeric_embedding_dim
            if input_dim == 0:
                raise ValueError(f"Feature group '{group.name}' must contain at least one field.")
            self.group_projectors[group.name] = nn.Sequential(
                nn.Linear(input_dim, model_config.model_dim),
                nn.GELU(),
                nn.LayerNorm(model_config.model_dim),
            )

        self.blocks = nn.ModuleList(
            [RankMixerBlock(num_tokens=feature_config.num_tokens, config=model_config) for _ in range(model_config.num_layers)]
        )
        self.output_head = nn.Sequential(
            nn.Linear(model_config.model_dim, model_config.model_dim),
            nn.GELU(),
            nn.Dropout(model_config.dropout),
        )
        self.ctr_head = nn.Linear(model_config.model_dim, 1)
        self.cvr_head = nn.Linear(model_config.model_dim, 1)

    def tokenize(self, batch: Dict[str, object]) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        categorical_batch: Dict[str, torch.Tensor] = batch["categorical"]
        numeric_batch: Dict[str, torch.Tensor] = batch["numeric"]

        for group in self.feature_config.groups:
            parts: List[torch.Tensor] = []
            for field in group.categorical:
                parts.append(self.categorical_embeddings[field.name](categorical_batch[field.name]))
            for field in group.numeric:
                parts.append(self.numeric_projections[field.name](numeric_batch[field.name].unsqueeze(-1)))
            token = torch.cat(parts, dim=-1)
            tokens.append(self.group_projectors[group.name](token))
        return torch.stack(tokens, dim=1)

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        x = self.tokenize(batch)
        aux_loss = x.new_tensor(0.0)
        for block in self.blocks:
            x, block_aux = block(x)
            aux_loss = aux_loss + block_aux
        pooled = x.mean(dim=1)
        shared = self.output_head(pooled)
        ctr_logits = self.ctr_head(shared).squeeze(-1)
        cvr_logits = self.cvr_head(shared).squeeze(-1)
        return {"ctr_logits": ctr_logits, "cvr_logits": cvr_logits, "aux_loss": aux_loss}

    def predict_proba(self, batches: Iterable[Dict[str, object]], device: Optional[torch.device] = None) -> List[float]:
        self.eval()
        outputs: List[float] = []
        with torch.no_grad():
            for batch in batches:
                batch = move_batch_to_device(batch, device or next(self.parameters()).device)
                forward_outputs = self.forward(batch)
                ctr_probs = torch.sigmoid(forward_outputs["ctr_logits"])
                cvr_probs = torch.sigmoid(forward_outputs["cvr_logits"])
                outputs.extend((ctr_probs * cvr_probs).cpu().tolist())
        return outputs

    def save_artifact(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "feature_config": self.feature_config.to_dict(),
                "model_config": self.model_config.to_dict(),
                "state_dict": self.state_dict(),
            },
            path,
        )

    @classmethod
    def load_artifact(cls, path: Path, map_location: str | torch.device = "cpu") -> "RankMixerModel":
        payload = torch.load(path, map_location=map_location)
        feature_config = RankMixerFeatureConfig.from_dict(payload["feature_config"])
        model_config = RankMixerModelConfig.from_dict(payload["model_config"])
        model = cls(feature_config=feature_config, model_config=model_config)
        model.load_state_dict(payload["state_dict"])
        return model


def move_batch_to_device(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    categorical = {name: tensor.to(device) for name, tensor in batch["categorical"].items()}
    numeric = {name: tensor.to(device) for name, tensor in batch["numeric"].items()}
    return {
        "request_id": batch["request_id"].to(device),
        "item_id": batch["item_id"].to(device),
        "labels": {name: tensor.to(device) for name, tensor in batch["labels"].items()},
        "categorical": categorical,
        "numeric": numeric,
    }


def compute_auc(labels: List[int], predictions: List[float]) -> float:
    pairs = sorted(zip(predictions, labels), key=lambda pair: pair[0])
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return 0.5
    rank_sum = 0.0
    rank = 1
    index = 0
    while index < len(pairs):
        next_index = index
        while next_index < len(pairs) and pairs[next_index][0] == pairs[index][0]:
            next_index += 1
        avg_rank = (rank + next_index) / 2.0
        positives = sum(pairs[i][1] for i in range(index, next_index))
        rank_sum += positives * avg_rank
        rank = next_index + 1
        index = next_index
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def compute_ndcg_at_k(request_ids: List[int], labels: List[int], predictions: List[float], k: int = 10) -> float:
    grouped: Dict[int, List[Tuple[float, int]]] = {}
    for request_id, label, score in zip(request_ids, labels, predictions):
        grouped.setdefault(request_id, []).append((score, label))

    ndcgs: List[float] = []
    for pairs in grouped.values():
        ranked = sorted(pairs, key=lambda pair: pair[0], reverse=True)[:k]
        ideal = sorted(pairs, key=lambda pair: pair[1], reverse=True)[:k]
        dcg = 0.0
        idcg = 0.0
        for index, (_, label) in enumerate(ranked, start=1):
            dcg += (2 ** label - 1) / math.log2(index + 1)
        for index, (_, label) in enumerate(ideal, start=1):
            idcg += (2 ** label - 1) / math.log2(index + 1)
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return sum(ndcgs) / len(ndcgs) if ndcgs else 0.0


def dump_metrics(path: Path, metrics: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
