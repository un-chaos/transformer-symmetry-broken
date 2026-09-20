"""
Shared helpers for the scripts in this directory.

Kept deliberately small: the data flags and the checkpoint plumbing are the only
things every script needs, and duplicating twenty argparse options in three
files is how flag names silently drift apart.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symbreak_transformer.config import (
    DATASET_PRESETS,
    BiasConfig,
    DataConfig,
    Seq2SeqConfig,
)
from symbreak_transformer.data import PAD_ID, Tokenizer, build_tokenizers
from symbreak_transformer.model import Seq2SeqTransformer
from symbreak_transformer.utils import resolve_device

__all__ = [
    "add_data_args",
    "build_data_config",
    "load_tokenizers",
    "load_checkpoint",
    "build_model",
    "zeroed_biases",
    "finite_or_none",
]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def add_data_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the data-source flags shared by every script that reads a corpus."""
    group = ap.add_argument_group("data")
    group.add_argument("--dataset", choices=["hf", "synthetic", "local"], default=None)
    group.add_argument(
        "--dataset_preset",
        choices=sorted(DATASET_PRESETS),
        default=None,
        help="shorthand for a dataset flag set, see scripts/train.py --list_models",
    )
    group.add_argument("--data_dir", type=str, default=None)
    group.add_argument("--hf_repo", type=str, default=None)
    group.add_argument("--hf_endpoint", type=str, default=None)
    group.add_argument("--src_field", type=str, default=None)
    group.add_argument("--tgt_field", type=str, default=None)
    group.add_argument("--max_train_samples", type=int, default=None)
    group.add_argument("--max_val_samples", type=int, default=None)
    group.add_argument(
        "--synthetic_task", choices=["copy", "reverse", "sort"], default=None
    )
    group.add_argument("--synthetic_train_size", type=int, default=None)
    group.add_argument("--synthetic_val_size", type=int, default=None)
    group.add_argument("--synthetic_test_size", type=int, default=None)
    group.add_argument("--synthetic_vocab", type=int, default=None)
    group.add_argument("--synthetic_len", type=int, default=None)
    group.add_argument("--tokenizer", choices=["word", "char"], default=None)
    group.add_argument("--min_freq", type=int, default=None)
    group.add_argument("--max_vocab", type=int, default=None)
    group.add_argument("--max_src_len", type=int, default=None)
    group.add_argument("--max_tgt_len", type=int, default=None)
    group.add_argument("--no_lowercase", action="store_true")
    group.add_argument("--no_fallback_to_synthetic", action="store_true")
    return ap


def build_data_config(args: argparse.Namespace) -> DataConfig:
    """
    Assemble a validated :class:`DataConfig` from the data flags.

    A ``--dataset_preset`` provides the baseline and every explicitly-passed flag
    overrides it (the argparse defaults are ``None`` precisely so that "not
    passed" is distinguishable from "passed the default value").
    """
    kwargs: Dict[str, object] = {}
    if getattr(args, "dataset_preset", None):
        kwargs.update(DATASET_PRESETS[args.dataset_preset])
    for name in (
        "data_dir",
        "hf_repo",
        "hf_endpoint",
        "src_field",
        "tgt_field",
        "max_train_samples",
        "max_val_samples",
        "synthetic_task",
        "synthetic_train_size",
        "synthetic_val_size",
        "synthetic_test_size",
        "synthetic_vocab",
        "synthetic_len",
        "tokenizer",
        "min_freq",
        "max_vocab",
        "max_src_len",
        "max_tgt_len",
    ):
        value = getattr(args, name, None)
        if value is not None:
            kwargs[name] = value
    if getattr(args, "dataset", None) is not None:
        kwargs["source"] = args.dataset
    if getattr(args, "no_lowercase", False):
        kwargs["lowercase"] = False
    if getattr(args, "no_fallback_to_synthetic", False):
        kwargs["fallback_to_synthetic"] = False
    return DataConfig(**kwargs).validate()


def load_tokenizers(
    ckpt_path: Path,
    data_cfg: DataConfig,
    train_pairs: Sequence[Tuple[str, str]],
) -> Tuple[Tokenizer, Tokenizer]:
    """
    Prefer the tokenizers saved beside the checkpoint, else rebuild them.

    Tokenizer construction is deterministic, so rebuilding from the same data
    flags reproduces the same vocabulary.
    """
    src_path = ckpt_path.parent / "tokenizer_src.json"
    tgt_path = ckpt_path.parent / "tokenizer_tgt.json"
    if src_path.exists() and tgt_path.exists():
        print(f"[data] reusing tokenizers from {ckpt_path.parent}")
        return Tokenizer.load(src_path), Tokenizer.load(tgt_path)
    print("[data] no saved tokenizers found next to the checkpoint, rebuilding them")
    return build_tokenizers(data_cfg, {"train": list(train_pairs)})


# --------------------------------------------------------------------------- #
# Checkpoints
# --------------------------------------------------------------------------- #
def load_checkpoint(path, device: Optional[torch.device] = None):
    """
    Load a training checkpoint written by ``scripts/train.py``.

    Returns:
        ``(cfg, bias_cfg, raw_checkpoint)`` where the raw dict still holds the
        ``model`` state dict and the bookkeeping fields.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(
        ckpt_path, map_location=device if device is not None else "cpu"
    )
    if "config" not in ckpt:
        raise SystemExit(f"{ckpt_path} has no 'config' entry; not a training checkpoint")
    cfg = Seq2SeqConfig.from_dict(ckpt["config"])
    bias_cfg = BiasConfig.from_dict(ckpt.get("bias") or {})
    return cfg, bias_cfg, ckpt


def build_model(
    cfg: Seq2SeqConfig,
    bias_cfg: BiasConfig,
    src_vocab_size: int,
    tgt_vocab_size: int,
    device: Optional[torch.device] = None,
) -> Seq2SeqTransformer:
    """Instantiate the model with the special-token ids used throughout."""
    model = Seq2SeqTransformer(
        cfg,
        bias_cfg,
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        pad_id=PAD_ID,
        bos_id=Tokenizer.bos_id,
        eos_id=Tokenizer.eos_id,
    )
    if device is not None:
        model = model.to(device)
    return model


# --------------------------------------------------------------------------- #
# Bias manipulation
# --------------------------------------------------------------------------- #
def _bias_tensors(model: Seq2SeqTransformer) -> List[torch.Tensor]:
    """Every live bias tensor of the model (embedding bias + attention sectors)."""
    tensors: List[torch.Tensor] = []
    for bias in model.embed_biases():
        if getattr(bias, "b", None) is not None:
            tensors.append(bias.b)
    for attn in model.attention_biases():
        for sector in attn.sectors().values():
            if getattr(sector, "b", None) is not None:
                tensors.append(sector.b)
    return tensors


@contextlib.contextmanager
def zeroed_biases(model: Seq2SeqTransformer) -> Iterator[None]:
    """
    Temporarily set every symmetry-breaking bias to exactly zero.

    This is how the analysis measures what the bias actually does: run the model
    with and without it on the same batch and compare. The originals are always
    restored, even if the body raises.
    """
    saved = [(tensor, tensor.detach().clone()) for tensor in _bias_tensors(model)]
    try:
        with torch.no_grad():
            for tensor, _ in saved:
                tensor.zero_()
        yield
    finally:
        with torch.no_grad():
            for tensor, original in saved:
                tensor.copy_(original)


def finite_or_none(value) -> Optional[float]:
    """Return ``float(value)`` when it is finite, else ``None`` (for JSON output)."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result and abs(result) != float("inf") else None
