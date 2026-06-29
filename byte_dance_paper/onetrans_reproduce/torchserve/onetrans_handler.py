# coding: utf-8
"""
OneTrans TorchServe Handler

对应 BERT 相似度 handler 的三个方法：
  preprocess  → 解析 JSON 请求，构造模型输入张量
  inference   → 调用 OneTrans 前向推理
  postprocess → 格式化输出为 JSON

请求格式（HTTP POST body，JSON）：
{
  "ns_features": {
    "user_age":      25.0,
    "item_price":    199.0,
    "user_city_id":  42,
    "user_gender":   1,
    "user_interest_tags": [3, 17, 42]
  },
  "sequences": [
    {
      "name": "click",
      "ids":        [1001, 2003, 5566],
      "cats":       [10,   20,   30],
      "prices":     [99.0, 199.0, 49.0],
      "timestamps": [1700000100, 1700000200, 1700000300]
    },
    {
      "name": "purchase",
      "ids":        [1001],
      "timestamps": [1700000050]
    }
  ]
}

响应格式：
{
  "ctr_score": 0.1234,
  "cvr_score": 0.0567
}
"""

import json
import logging
import math
import os
import sys
from typing import Dict, List

import torch
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class OneTrans_Handler(BaseHandler):

    def __init__(self):
        super().__init__()
        self.initialized  = False
        self.model        = None
        self.setup_config = None
        self.ns_specs     = None   # 从 feature_config.yaml 加载的特征描述
        self.seq_configs  = None
        self.label_cols   = None

    # ──────────────────────────────────────────────────────────────────────
    # 1. initialize：加载模型和配置（TorchServe 启动时调用一次）
    # ──────────────────────────────────────────────────────────────────────

    def initialize(self, ctx):
        self.manifest   = ctx.manifest
        properties      = ctx.system_properties
        model_dir       = properties.get("model_dir")

        # 确定设备
        gpu_id = properties.get("gpu_id")
        self.device = torch.device(
            f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None
            else "cpu"
        )
        logger.info("OneTrans handler: device=%s", self.device)

        # 读取 setup_config
        cfg_path = os.path.join(model_dir, "setup_config_onetrans.json")
        with open(cfg_path, "r", encoding="utf-8") as f:
            self.setup_config = json.load(f)

        # 把项目代码目录加入 sys.path，使 model / tokenizer / data 可以 import
        sys.path.insert(0, model_dir)
        from model import OneTrans
        from data  import load_feature_config

        # 加载特征配置
        feat_cfg_path = os.path.join(model_dir, self.setup_config["feature_config"])
        self.ns_specs, self.seq_configs, self.label_cols = load_feature_config(feat_cfg_path)
        self.task_names = self.setup_config.get("task_names", ["ctr", "cvr"])

        # 预先计算各序列的 max_len，供 preprocess 使用
        # seq_configs 格式：{"name", "timestamps", "max_len", "fields": [{"col","scale"},...]}
        self.seq_max_lens = [cfg["max_len"] for cfg in self.seq_configs]

        # 加载 checkpoint
        serialized_file = self.manifest["model"]["serializedFile"]
        ckpt_path       = os.path.join(model_dir, serialized_file)
        ckpt            = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model = OneTrans(**ckpt["model_kwargs"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        logger.info(
            "OneTrans 加载成功: epoch=%s, metrics=%s",
            ckpt.get("epoch"), ckpt.get("metrics"),
        )
        self.initialized = True

    # ──────────────────────────────────────────────────────────────────────
    # 2. preprocess：JSON 请求 → 模型输入张量
    # ──────────────────────────────────────────────────────────────────────

    def preprocess(self, requests):
        """
        将 HTTP 请求体解析为模型输入。
        支持单条请求（batch_size=1）和批量请求（TorchServe 动态 batching）。
        """
        batch_ns     = []   # List[Dict[str, tensor(scalar)]]
        batch_seqs   = [[] for _ in self.seq_configs]   # List[List[tensor(L,d)]]
        batch_ts     = [[] for _ in self.seq_configs]
        batch_masks  = [[] for _ in self.seq_configs]

        for req in requests:
            body = req.get("body") or req.get("data")
            if isinstance(body, (bytes, bytearray)):
                body = body.decode("utf-8")
            data = json.loads(body) if isinstance(body, str) else body

            # ── NS 特征 ──
            ns_raw = data.get("ns_features", {})
            ns_inputs = self._parse_ns(ns_raw)
            batch_ns.append(ns_inputs)

            # ── 行为序列 ──
            seq_raw = data.get("sequences", [])
            # 按 name 字段匹配（来自 feature_config.yaml 中 sequences[].name）
            seq_by_name = {s.get("name", f"seq_{i}"): s for i, s in enumerate(seq_raw)}

            for i, cfg in enumerate(self.seq_configs):
                # seq_configs 为 dict 格式：{"name", "timestamps", "max_len", "fields":[...]}
                seq_name = cfg.get("name", f"seq_{i}")
                seq_data = seq_by_name.get(seq_name, {})

                seq_tensor, ts_tensor, mask_tensor = self._parse_seq(seq_data, cfg)
                batch_seqs[i].append(seq_tensor)
                batch_ts[i].append(ts_tensor)
                batch_masks[i].append(mask_tensor)

        # ── 组装 batch ──
        # ns_inputs：每个特征 stack 成 (B,) 或 (B, max_len)
        ns_keys = batch_ns[0].keys()
        ns_batch = {
            k: torch.stack([item[k] for item in batch_ns]).to(self.device)
            for k in ns_keys
        }
        seqs_batch  = [torch.stack(s).to(self.device) for s in batch_seqs]
        ts_batch    = [torch.stack(t).to(self.device) for t in batch_ts]
        masks_batch = [torch.stack(m).to(self.device) for m in batch_masks]

        return ns_batch, seqs_batch, ts_batch, masks_batch

    def _parse_ns(self, ns_raw: dict) -> Dict[str, torch.Tensor]:
        """将 JSON 里的 ns_features 转为 {特征名: tensor} 字典"""
        result = {}
        for spec in self.ns_specs:
            name  = spec["name"]
            ftype = spec["type"]
            val   = ns_raw.get(name, 0)

            if ftype == "continuous":
                scale = spec["scale"]
                v = float(val)
                v = math.log1p(max(v, 0.0)) if scale == -1 else v / scale
                result[name] = torch.tensor(v, dtype=torch.float32)

            elif ftype in ("discrete_id", "discrete_str"):
                idx = int(val) if val else 0
                idx = max(0, min(idx, spec["vocab_size"] - 1))
                result[name] = torch.tensor(idx, dtype=torch.long)

            elif ftype == "multihot":
                max_len    = spec["max_len"]
                vocab_size = spec["vocab_size"]
                ids = [int(x) for x in (val if isinstance(val, list) else [])][:max_len]
                ids = [max(0, min(x, vocab_size - 1)) for x in ids]
                ids += [0] * (max_len - len(ids))   # padding
                result[name] = torch.tensor(ids, dtype=torch.long)

        return result

    def _parse_seq(self, seq_data: dict, cfg: dict):
        """
        将单条序列 JSON 转为 (seq_tensor, ts_tensor, mask_tensor)。
        按 cfg["fields"] 顺序动态拼向量，与 data.py 的 _build_seqs_row 逻辑保持一致。

        请求 JSON 中序列字段命名约定：
          fields 中第 1 个字段 → "ids"
          fields 中第 2 个字段 → "field_1"
          fields 中第 3 个字段 → "field_2"
          ...（或直接用 col 名，handler 会同时尝试两种 key）
        """
        fields  = cfg["fields"]
        max_len = cfg["max_len"]
        col_ts  = cfg.get("timestamps")

        # 请求 JSON 里的字段 key 约定：第 1 个 field 用 "ids"，其余用 col 名或 "field_N"
        def get_field_vals(field_idx: int, col_name: str, scale: float) -> list:
            key = "ids" if field_idx == 0 else col_name
            raw = seq_data.get(key, seq_data.get(f"field_{field_idx}", []))
            return [float(x) / scale for x in raw][-max_len:]

        field_vals = [
            get_field_vals(idx, f["col"], f["scale"])
            for idx, f in enumerate(fields)
        ]
        actual_len = len(field_vals[0]) if field_vals else 0
        pad        = max_len - actual_len

        # 时间戳转相对时间差（与 data.py 保持一致）
        ts_raw = [float(x) for x in seq_data.get("timestamps", [])][-max_len:]
        if ts_raw:
            ref = max(ts_raw)
            ts_vals = [(ref - t) / ref if ref > 0 else 0.0 for t in ts_raw]
        else:
            ts_vals = []
        ts_vals += [0.0] * (max_len - len(ts_vals))

        mask  = [True] * actual_len + [False] * pad
        parts = []
        for vals in field_vals:
            vals_padded = vals + [0.0] * (max_len - len(vals))
            parts.append(torch.tensor(vals_padded, dtype=torch.float32).unsqueeze(1))

        return (
            torch.cat(parts, dim=1),                        # (max_len, num_fields)
            torch.tensor(ts_vals, dtype=torch.float32),     # (max_len,)
            torch.tensor(mask,    dtype=torch.bool),        # (max_len,)
        )

    # ──────────────────────────────────────────────────────────────────────
    # 3. inference：调用模型前向
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def inference(self, input_batch):
        ns_batch, seqs_batch, ts_batch, masks_batch = input_batch
        predictions, _ = self.model(ns_batch, seqs_batch, ts_batch, masks_batch)
        return predictions   # Dict[task_name, (B,) tensor]

    # ──────────────────────────────────────────────────────────────────────
    # 4. postprocess：预测结果 → JSON 响应
    # ──────────────────────────────────────────────────────────────────────

    def postprocess(self, inference_output):
        """
        将 (B,) 的预测 tensor 转为 JSON 列表，每条对应一个请求。
        响应示例：{"ctr_score": 0.1234, "cvr_score": 0.0567}
        """
        B = next(iter(inference_output.values())).size(0)
        results = []
        for i in range(B):
            row = {
                # model 返回 logit，postprocess 时转为概率
                f"{name}_score": round(torch.sigmoid(inference_output[name][i]).item(), 6)
                for name in self.task_names
                if name in inference_output
            }
            results.append(json.dumps(row, ensure_ascii=False))
        return results
