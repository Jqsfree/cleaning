"""Ettin CrossEncoder reranker loader (ST-incompatible HF layouts).

cross-encoder/ettin-reranker-{32m,68m}-v1 ship as modular ST packages
(encoder + Pooling + Dense + LayerNorm + Dense) whose safetensors cannot be
loaded by sentence-transformers 5.1 / transformers 4.57 AutoModel-for-sequence-
classification. This module assembles the same stack manually and scores
(query, document) pairs → relevance logits.

Only pass input_ids / attention_mask to the ModernBERT encoder (no token_type_ids).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, PreTrainedTokenizerFast

DEFAULT_MODEL_IDS = {
    "32m": "cross-encoder/ettin-reranker-32m-v1",
    "68m": "cross-encoder/ettin-reranker-68m-v1",
}

# Entertainment live-performance retention query (gold AUC≈0.815 on 68m).
ENTERTAINMENT_QUERY = (
    "street busker performing for crowd; dance studio choreography practice; "
    "circus acrobat live show; college band concert; talent show audition stage"
)


class _Dense(nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool, activation: str):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = nn.GELU() if activation == "gelu" else nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.linear(features))


class _LayerNorm(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.norm = nn.LayerNorm(dimension)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.norm(features)


class EttinReranker(nn.Module):
    """Manual assembly of Ettin ST modular CrossEncoder heads."""

    def __init__(
        self,
        model_id: str,
        *,
        device: str | None = None,
        cache_dir: str | Path | None = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.device = device or _default_device()
        allow = [
            "config.json",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "1_Pooling/*",
            "2_Dense/*",
            "3_LayerNorm/*",
            "4_Dense/*",
            "modules.json",
        ]
        snap = Path(
            snapshot_download(
                model_id,
                allow_patterns=allow,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
        )
        tok_cfg = json.loads((snap / "tokenizer_config.json").read_text(encoding="utf-8"))
        self.tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(snap / "tokenizer.json"),
            unk_token=tok_cfg.get("unk_token", "[UNK]"),
            sep_token=tok_cfg.get("sep_token", "[SEP]"),
            pad_token=tok_cfg.get("pad_token", "[PAD]"),
            cls_token=tok_cfg.get("cls_token", "[CLS]"),
            mask_token=tok_cfg.get("mask_token", "[MASK]"),
            model_max_length=int(tok_cfg.get("model_max_length", 7999)),
        )
        self.encoder = AutoModel.from_pretrained(str(snap))
        d2 = json.loads((snap / "2_Dense/config.json").read_text(encoding="utf-8"))
        d4 = json.loads((snap / "4_Dense/config.json").read_text(encoding="utf-8"))
        ln = json.loads((snap / "3_LayerNorm/config.json").read_text(encoding="utf-8"))
        self.dense2 = _Dense(
            d2["in_features"],
            d2["out_features"],
            bias=bool(d2.get("bias", False)),
            activation="gelu",
        )
        self.ln = _LayerNorm(int(ln["dimension"]))
        self.dense4 = _Dense(
            d4["in_features"],
            d4["out_features"],
            bias=bool(d4.get("bias", True)),
            activation="identity",
        )
        self.dense2.load_state_dict(load_file(str(snap / "2_Dense/model.safetensors")))
        self.ln.load_state_dict(load_file(str(snap / "3_LayerNorm/model.safetensors")))
        self.dense4.load_state_dict(load_file(str(snap / "4_Dense/model.safetensors")))
        self.to(self.device).eval()

    @torch.inference_mode()
    def predict(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        batch_size: int = 64,
        max_length: int = 256,
    ) -> np.ndarray:
        if not pairs:
            return np.asarray([], dtype=np.float32)
        scores: list[np.ndarray] = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            enc = self.tokenizer(
                [p[0] for p in batch],
                [p[1] for p in batch],
                padding=True,
                truncation="longest_first",
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {
                k: v.to(self.device)
                for k, v in enc.items()
                if k in ("input_ids", "attention_mask")
            }
            cls = self.encoder(**enc).last_hidden_state[:, 0]
            logit = self.dense4(self.ln(self.dense2(cls))).squeeze(-1)
            scores.append(logit.float().cpu().numpy())
        return np.concatenate(scores).astype(np.float32)


def resolve_model_id(size_or_id: str) -> str:
    key = size_or_id.strip().lower()
    if key in DEFAULT_MODEL_IDS:
        return DEFAULT_MODEL_IDS[key]
    return size_or_id


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value != value:  # NaN
        return ""
    text = str(value)
    if text.lower() in {"nan", "none"}:
        return ""
    return text.replace("\x00", "").strip()


def build_doc(title: object, channel: object | None = None) -> str:
    """Document text for reranking — title only (channel ignored)."""
    del channel  # API compat; feature contract is title-only
    return _as_text(title)


def _default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
