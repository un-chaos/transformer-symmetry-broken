#!/usr/bin/env python3
"""
Unified training script for the symmetry-breaking encoder-decoder Transformer.

The network shape, the symmetry-breaking bias ``b`` and the optimizer are three
independent command-line choices, plus per-field overrides on top of them:

* ``--model`` picks an entry of :data:`symbreak_transformer.config.PRESETS`;
* ``--bias_preset`` picks an entry of
  :data:`symbreak_transformer.config.BiasPresets` (``symmetric``, ``b-gaussian``,
  ``b-const``, ``attn-bQbV``, ...) and every individual ``--bias_*`` /
  ``--mean_Q`` / ``--use_q_bias`` flag overrides a single field of it;
* ``--optimizer`` selects ``egd`` (energy-conserving descent, driven through a
  closure), ``adamw`` or ``sgdm`` -- all three run through *one* training loop.

Everything the run needs is written into ``runs/<run_name>/``: a header-first
``training_log.csv`` (loss, throughput, the EGD internals and the bias norms),
``model_best.pt`` / ``model_final.pt``, the resolved ``args.json`` /
``config.json`` / ``bias.json``, the two tokenizers and ``training_curve.png``.
Checkpoints store the configs as **plain dicts**, so an old checkpoint stays
loadable even after a dataclass changes.

The one thing to get right is EGD's ``F0``: the step is only taken while
``loss - F0 > eps2``.  Leave ``--egd_F0`` unset (``None``) and it resolves to
``initial_loss - auto_F0_margin``, which is always safe.

Usage Examples:
    # symmetric control run (b = 0, no attention biases)
    python scripts/train.py --model small --bias_preset symmetric \\
        --dataset_preset multi30k-tiny --optimizer egd

    # the same model with the embedding symmetry broken by a fixed Gaussian b
    python scripts/train.py --model small --bias_preset b-gaussian \\
        --dataset_preset multi30k-tiny --optimizer egd

    # reference-style per-head biases bQ + bV, biases redrawn every step
    python scripts/train.py --model small --bias_preset attn-bQbV \\
        --attn_resample per_step --dataset_preset multi30k-tiny

    # AdamW comparison run
    python scripts/train.py --model small --bias_preset b-gaussian \\
        --dataset_preset multi30k-tiny --optimizer adamw --adam_lr 1e-4

    # offline smoke test: synthetic task, no network, well under a minute
    python scripts/train.py --model smoke --dataset_preset synthetic-reverse \\
        --max_steps 60

    # what is available?
    python scripts/train.py --list_models
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add the repository root to the path so ``python scripts/train.py`` works from
# anywhere (upstream does the same).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symbreak_transformer.config import (  # noqa: E402
    DATASET_PRESETS,
    PRESETS,
    BiasConfig,
    BiasPresets,
    DataConfig,
    resolve_bias_config,
    preset_table,
    resolve_config,
)
from symbreak_transformer.data import (  # noqa: E402
    PAD_ID,
    Tokenizer,
    build_dataloaders,
    build_tokenizers,
    load_all_splits,
)
from symbreak_transformer.evaluate import evaluate_bleu, evaluate_loss  # noqa: E402
from symbreak_transformer.model import Seq2SeqTransformer  # noqa: E402
from symbreak_transformer.optimizer import build_optimizer as _build_optimizer  # noqa: E402
from symbreak_transformer.utils import (  # noqa: E402
    CSVLogger,
    StepWatchdog,
    Timer,
    configure_console_encoding,
    count_parameters,
    environment_report,
    format_seconds,
    resolve_device,
    seed_everything,
)

# Optional dependency, exactly like upstream: without wandb the flag is a no-op.
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

__all__ = ["build_optimizer", "main", "bias_tag", "run_name_for", "load_plot_curve",
           "data_config_from_args", "model_config_from_args", "bias_config_from_args",
           "clip_lengths", "model_parameter_counts"]

#: Columns of ``runs/<run>/training_log.csv``, in this exact order.  Rows are
#: written twice per eval cycle (a train row at ``--log_every`` and a val row at
#: ``--valid_every_updates``); a field a row does not carry is left blank.  The
#: first column is ``step`` -- one optimizer step -- which the README and
#: ``scripts/plot_curve.py`` both read; the ``*_every_updates`` flags are what
#: control its cadence.
LOG_FIELDS: List[str] = [
    "step",
    "epoch",
    "train_loss",
    "train_ppl",
    "val_loss",
    "val_ppl",
    "elapsed_sec",
    "sec_per_step",
    "tokens_per_sec",
    "egd_iteration",
    "egd_momentum_norm",
    "egd_skipped",
    "embed_b_norm",
    "bQ_norm",
    "bK_norm",
    "bV_norm",
    "best_val_loss",
]


# --------------------------------------------------------------------------- #
# Optimizer
# --------------------------------------------------------------------------- #
def build_optimizer(model, kind, egd_kwargs=None, adamw_kwargs=None, sgdm_kwargs=None):
    """Build the optimizer selected by ``kind`` (``egd`` / ``adamw`` / ``sgdm``).

    Thin wrapper around :func:`symbreak_transformer.optimizer.build_optimizer`
    with the upstream signature, so the CLI can hand it plain kwargs dicts.
    Only parameters with ``requires_grad=True`` reach the optimizer.

    Args:
        model: the model whose parameters are optimised.
        kind: ``"egd"``, ``"adamw"`` or ``"sgdm"``.
        egd_kwargs: keyword arguments for
            :class:`~symbreak_transformer.optimizer.EGD`.
        adamw_kwargs: keyword arguments for :class:`torch.optim.AdamW`.
        sgdm_kwargs: keyword arguments for :class:`torch.optim.SGD`.

    Returns:
        The constructed optimizer.

    Raises:
        ValueError: unknown ``kind`` or a model without trainable parameters.
    """
    return _build_optimizer(
        model,
        kind,
        egd_kwargs=egd_kwargs,
        adamw_kwargs=adamw_kwargs,
        sgdm_kwargs=sgdm_kwargs,
    )


# --------------------------------------------------------------------------- #
# Run identity
# --------------------------------------------------------------------------- #
def bias_tag(bias_cfg: BiasConfig) -> str:
    """Short slug naming the symmetry-breaking setting of a run.

    ``symmetric`` / ``bgaussian`` / ``bgaussian-rs`` / ``bconst`` / ``blearn``
    for the embedding bias, ``attn-bQ`` / ``attn-bQbV`` / ``attn-bQbKbV`` for the
    per-head ones; an embedding bias and an attention bias can both be on, in
    which case the two slugs are joined with ``+``.
    """
    parts: List[str] = []
    if bias_cfg.embed_mode != "zero" or bias_cfg.embed_learnable:
        parts.append(f"b{bias_cfg.embed_mode}")
        if bias_cfg.embed_resample == "per_step":
            parts.append("rs")
        if bias_cfg.embed_learnable:
            parts.append("learn")
    if bias_cfg.attention_enabled:
        sectors = "".join(
            name
            for name, on in (
                ("Q", bias_cfg.use_q_bias),
                ("K", bias_cfg.use_k_bias),
                ("V", bias_cfg.use_v_bias),
            )
            if on
        )
        attn = f"attn-b{sectors}"
        if bias_cfg.attn_resample == "per_step":
            attn += "-rs"
        if bias_cfg.attn_learnable:
            attn += "-learn"
        parts.append(attn)
    return "+".join(parts) if parts else "symmetric"


def run_name_for(model: str, optimizer: str, bias_cfg: BiasConfig, seed: int) -> str:
    """``<model>-<optimizer>-<bias slug>-seed<seed>``, as ``--name`` would."""
    return f"{model}-{optimizer}-{bias_tag(bias_cfg)}-seed{seed}"


# --------------------------------------------------------------------------- #
# Logging helpers
# --------------------------------------------------------------------------- #
def _bias_norms(report: Dict[str, Any]) -> Dict[str, float]:
    """Collapse ``model.bias_report()`` into the four CSV norm columns.

    The report names its entries ``embed.bias_norm`` and ``<site>.bQ_norm`` with
    one entry per attention site (``encoder``, ``decoder_self``, ...) and mode
    strings next to the numbers.  Keys are normalised (non-alphanumerics dropped,
    lower-cased), non-numeric values are ignored, and each column takes the
    **largest** norm seen for that sector -- so the CSV keeps exactly one column
    per bias however many sites or layers carry it.
    """
    wanted = {
        "embed_b_norm": ("embedbiasnorm", "embedbnorm", "embednorm"),
        "bQ_norm": ("bqnorm",),
        "bK_norm": ("bknnorm",),
        "bV_norm": ("bvnorm",),
    }
    normalised: Dict[str, float] = {}
    for key, value in (report or {}).items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue  # modes and other strings carry no norm
        normalised["".join(ch for ch in str(key).lower() if ch.isalnum())] = numeric

    out: Dict[str, float] = {}
    for field, candidates in wanted.items():
        matches = [
            value
            for key, value in normalised.items()
            if any(candidate in key for candidate in candidates)
        ]
        out[field] = max(matches) if matches else 0.0
    return out


def _bias_report(model, note: Dict[str, Any]) -> Dict[str, Any]:
    """``model.bias_report()``, never fatal.

    The bias norms are diagnostics: a run that has already spent minutes on the
    optimizer must not die because the reporting helper choked.  A failure is
    reported once per run and the norm columns stay at ``0.0`` (the caller
    zero-fills), so the CSV keeps its shape.

    Args:
        model: the model to ask.
        note: mutable per-run bookkeeping (``{"warned": bool, "error": str}``).
    """
    try:
        return model.bias_report() or {}
    except Exception as exc:
        note["error"] = f"{type(exc).__name__}: {exc}"
        if not note["warned"]:
            note["warned"] = True
            print(
                f"[bias] model.bias_report() failed ({note['error']}); "
                f"the bias-norm columns will stay 0.0"
            )
        return {}


def _egd_log_fields(optimizer) -> Dict[str, Any]:
    """EGD internals for the CSV; all ``nan``/absent for AdamW and SGDM."""
    stats: Dict[str, Any] = {}
    if hasattr(optimizer, "stats"):
        try:
            stats = optimizer.stats() or {}
        except Exception:  # pragma: no cover - stats() is best-effort only
            stats = {}
    return {
        "egd_iteration": stats.get("iteration", ""),
        "egd_momentum_norm": stats.get("momentum_norm", ""),
        "egd_skipped": stats.get("skipped_updates", ""),
    }


def _row_extra(model, optimizer, elapsed: float, note: Dict[str, Any]) -> Dict[str, Any]:
    """Bias norms + EGD internals + elapsed time, shared by both row kinds."""
    row: Dict[str, Any] = {"elapsed_sec": elapsed}
    row.update(_bias_norms(_bias_report(model, note)))
    row.update(_egd_log_fields(optimizer))
    return row


def _record_loss(path: Optional[Path], loss: float, tag: str = "train_loss") -> None:
    """Append one logged loss value to ``losses.csv`` beside the run's log.

    **This file is part of the contract, not a debug leftover.**  A run writes two
    CSVs with deliberately different jobs:

    * ``training_log.csv`` -- the rich, *interleaved* log.  A train row is written
      every ``--log_every`` updates and a val row every ``--valid_every_updates``,
      so a field a given row does not carry is blank and there is **no** one row
      per optimizer step.
    * ``losses.csv`` -- exactly ``tag,loss``, one row per optimizer update.  It is
      the only place the raw per-update loss trajectory exists, which is what a
      test asserting "the loss went down" (or a curve against update index)
      needs without knowing the logging cadence.

    ``tests/test_evaluate.py::read_losses`` reads this file, so dropping it would
    break the end-to-end suite.
    """
    if path is None:
        return
    try:
        with open(path, "a") as fh:
            fh.write(f"{tag},{float(loss):.6f}\n")
    except OSError:  # pragma: no cover - logging must never kill a run
        pass


def _open_csv_logger(path: Path, fieldnames: List[str], append: bool):
    """Open a :class:`CSVLogger`, appending to an existing log when asked.

    ``CSVLogger`` truncates on construction so a re-run never appends to stale
    rows.  A **resumed** run is the one case where appending is right: the rows
    already in the CSV are this same run's history, and truncating them would
    lose the early part of the curve.  The existing header is kept verbatim, so a
    log written by an older revision still has its own column order.
    """
    if not append or not path.exists():
        return CSVLogger(path, fieldnames)

    rows = path.read_text(encoding="utf-8").splitlines()
    header = rows[0].split(",") if rows else []
    logger = CSVLogger(path, header or fieldnames)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows[1:]:
            fh.write(row + "\n")
    print(f"[log] appending to the existing log ({len(rows) - 1} row(s) kept)")
    return logger


# --------------------------------------------------------------------------- #
# Checkpointing
# --------------------------------------------------------------------------- #
def _restore_checkpoint(path: Path, model, optimizer, device) -> "tuple[int, float]":
    """Load a checkpoint into ``model``/``optimizer``; return ``(start_step, best_val)``.

    The optimizer state is restored inside a ``try`` because a checkpoint may
    carry an AdamW-shaped dict for an EGD run (or the reverse), and that must
    degrade to "continue with fresh optimizer state", never abort the run.

    Returns:
        ``(int(ckpt.get("update", 0)), ckpt.get("val_loss", inf))`` -- the step
        to continue from (so training reaches ``total_steps`` rather than
        restarting) and the best validation loss already achieved, so a resumed
        run cannot overwrite a better ``model_best.pt`` with a worse one.
    """
    if not path.exists():
        print(f"[resume] WARNING: {path} does not exist; starting from scratch")
        return 0, float("inf")
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception as exc:
        print(
            f"[resume] WARNING: could not read {path} "
            f"({type(exc).__name__}: {exc}); starting from scratch"
        )
        return 0, float("inf")
    if not isinstance(ckpt, dict) or "model" not in ckpt:
        print(f"[resume] WARNING: {path} is not a training checkpoint; starting fresh")
        return 0, float("inf")

    model.load_state_dict(ckpt["model"])
    if "optimizer" in ckpt:
        try:
            optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as exc:
            print(
                f"[resume] could not restore optimizer state "
                f"({type(exc).__name__}: {exc}); continuing with fresh optimizer state"
            )
    start = int(ckpt.get("update", 0))
    prior_best = ckpt.get("val_loss")
    prior_best = float(prior_best) if prior_best is not None else float("inf")
    print(
        f"[resume] loaded {path} at update {start} "
        f"(val_loss={ckpt.get('val_loss')}, consumed_tokens={ckpt.get('consumed_tokens')})"
    )
    return start, prior_best


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer,
    cfg,
    bias_cfg: BiasConfig,
    update: int,
    epoch: int,
    val_loss: float,
    consumed_tokens: int,
) -> Path:
    """Save a checkpoint as **plain dicts** (never pickled dataclasses).

    ``config``/``bias`` go through ``to_dict()`` so the payload is a pure
    ``dict``/``float``/``Tensor`` tree: an old checkpoint stays loadable after a
    dataclass gains a field, and ``torch.load``'s ``weights_only`` default keeps
    working.
    """
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "config": cfg.to_dict(),
        "bias": bias_cfg.to_dict(),
        "update": int(update),
        "epoch": int(epoch),
        "consumed_tokens": int(consumed_tokens),
        "val_loss": float(val_loss),
        "symmetric": bool(bias_cfg.symmetric),
        "symmetry_breaking": _symmetry_breaking(bias_cfg),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic-ish: never leave a half-written checkpoint
    return path


def _symmetry_breaking(bias_cfg: BiasConfig) -> Dict[str, Any]:
    """The ``symmetry_breaking`` entry every checkpoint carries."""
    return {
        "symmetric": bool(bias_cfg.symmetric),
        "embed_mode": bias_cfg.embed_mode,
        "embed_std": bias_cfg.embed_std,
        "embed_const_value": bias_cfg.embed_const_value,
        "embed_resample": bias_cfg.embed_resample,
        "embed_learnable": bias_cfg.embed_learnable,
        "bQ": bool(bias_cfg.use_q_bias),
        "bK": bool(bias_cfg.use_k_bias),
        "bV": bool(bias_cfg.use_v_bias),
        "attn_mode": bias_cfg.attn_mode,
        "attn_resample": bias_cfg.attn_resample,
        "attn_learnable": bias_cfg.attn_learnable,
        "mean_Q": bias_cfg.mean_Q,
        "std_Q": bias_cfg.std_Q,
        "mean_K": bias_cfg.mean_K,
        "std_K": bias_cfg.std_K,
        "mean_V": bias_cfg.mean_V,
        "std_V": bias_cfg.std_V,
        "share_across_heads": bool(bias_cfg.share_across_heads),
        "share_across_layers": bool(bias_cfg.share_across_layers),
    }


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """The full flag list.

    Every flag that stands for a config field uses ``argparse.SUPPRESS`` as its
    default, so the namespace tells "flag not given" apart from "flag given the
    field's default value".  That distinction is what makes
    ``--bias_preset`` + single-field overrides work: only the flags actually
    present on the command line are applied on top of the preset.
    """
    S = argparse.SUPPRESS
    ap = argparse.ArgumentParser(
        description=(
            "Train the symmetry-breaking encoder-decoder Transformer "
            "(EGD / AdamW / SGDM)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- model shape ------------------------------------------------------- #
    g = ap.add_argument_group("model shape")
    g.add_argument("--model", choices=sorted(PRESETS), default="small",
                   help="model size preset (default: small)")
    g.add_argument("--ctx", type=int, default=S, help="context length")
    g.add_argument("--n_embd", type=int, default=S)
    g.add_argument("--n_head", type=int, default=S)
    g.add_argument("--n_encoder_layer", type=int, default=S)
    g.add_argument("--n_decoder_layer", type=int, default=S)
    g.add_argument("--d_ff", type=int, default=S)
    g.add_argument("--dropout", type=float, default=S)
    g.add_argument("--attention_dropout", type=float, default=S)
    g.add_argument("--activation", choices=["gelu", "relu", "prelu"], default=S)
    g.add_argument("--use_prelu", action="store_true", default=S,
                   help="shorthand for --activation prelu")
    g.add_argument("--prelu_random_init", action="store_true", default=S)
    g.add_argument("--norm_first", action="store_true", default=S)
    g.add_argument("--no_scale_embedding", action="store_false", dest="scale_embedding",
                   default=S, help="do not scale the embedding by sqrt(n_embd)")
    g.add_argument("--no_tie_output_embedding", action="store_false",
                   dest="tie_output_embedding", default=S,
                   help="use a separate output projection")
    g.add_argument("--share_embeddings", action="store_true", default=S,
                   help="one embedding matrix for source and target (equal vocabs)")
    g.add_argument("--init_std", type=float, default=S)
    g.add_argument("--vocab", type=int, default=12000,
                   help="vocabulary size used for the --list_models parameter counts")
    g.add_argument("--list_models", action="store_true",
                   help="print the model / bias / dataset presets and exit")

    # --- bias (the point of the project) ----------------------------------- #
    g = ap.add_argument_group("symmetry-breaking bias")
    g.add_argument("--bias_preset", choices=sorted(BiasPresets), default=S,
                   help="named bias setting; single flags below override its fields")
    g.add_argument("--symmetric", action="store_true", default=S,
                   help="force b = 0 and every attention bias off (control run)")
    g.add_argument("--bias_mode", choices=["zero", "gaussian", "const"], default=S,
                   help="how the embedding bias b is drawn")
    g.add_argument("--bias_mean", type=float, default=S)
    g.add_argument("--bias_std", type=float, default=S)
    g.add_argument("--bias_const", type=float, default=S, help="value used by const mode")
    g.add_argument("--bias_resample", choices=["fixed", "per_step"], default=S)
    g.add_argument("--bias_learnable", action="store_true", default=S,
                   help="train b instead of keeping it a fixed buffer")
    g.add_argument("--use_q_bias", action="store_true", default=S)
    g.add_argument("--use_k_bias", action="store_true", default=S)
    g.add_argument("--use_v_bias", action="store_true", default=S)
    g.add_argument("--attn_mode", choices=["zero", "gaussian", "const"], default=S)
    g.add_argument("--attn_resample", choices=["fixed", "per_step"], default=S)
    g.add_argument("--attn_learnable", action="store_true", default=S)
    g.add_argument("--mean_Q", type=float, default=S)
    g.add_argument("--std_Q", type=float, default=S)
    g.add_argument("--mean_K", type=float, default=S)
    g.add_argument("--std_K", type=float, default=S)
    g.add_argument("--mean_V", type=float, default=S)
    g.add_argument("--std_V", type=float, default=S)
    g.add_argument("--no_share_across_heads", action="store_false",
                   dest="share_across_heads", default=S,
                   help="give every head its own bias vector")
    g.add_argument("--no_share_across_layers", action="store_false",
                   dest="share_across_layers", default=S,
                   help="give every layer its own bias object")
    g.add_argument("--bias_seed", type=int, default=S, help="dedicated bias RNG seed")

    # --- data -------------------------------------------------------------- #
    g = ap.add_argument_group("data")
    g.add_argument("--dataset", choices=["hf", "synthetic", "local"], default=S,
                   help="where the parallel text comes from")
    g.add_argument("--dataset_preset", choices=sorted(DATASET_PRESETS), default=S)
    g.add_argument("--data_dir", type=str, default=S)
    g.add_argument("--hf_repo", type=str, default=S)
    g.add_argument("--hf_endpoint", type=str, default=S)
    g.add_argument("--src_field", type=str, default=S)
    g.add_argument("--tgt_field", type=str, default=S)
    g.add_argument("--max_train_samples", type=int, default=S, help="0 = no limit")
    g.add_argument("--max_val_samples", type=int, default=S, help="0 = no limit")
    g.add_argument("--synthetic_task", choices=["copy", "reverse", "sort"], default=S)
    g.add_argument("--synthetic_train_size", type=int, default=S)
    g.add_argument("--synthetic_val_size", type=int, default=S)
    g.add_argument("--synthetic_test_size", type=int, default=S)
    g.add_argument("--synthetic_vocab", type=int, default=S)
    g.add_argument("--synthetic_len", type=int, default=S)
    g.add_argument("--tokenizer", choices=["word", "char"], default=S)
    g.add_argument("--min_freq", type=int, default=S)
    g.add_argument("--max_vocab", type=int, default=S)
    g.add_argument("--no_lowercase", action="store_false", dest="lowercase", default=S)
    g.add_argument("--max_src_len", type=int, default=S, help="0 = context length")
    g.add_argument("--max_tgt_len", type=int, default=S, help="0 = context length")
    g.add_argument("--no_fallback_to_synthetic", action="store_false",
                   dest="fallback_to_synthetic", default=S,
                   help="fail loudly when a HuggingFace download fails")

    # --- optimizer --------------------------------------------------------- #
    g = ap.add_argument_group("optimizer")
    g.add_argument("--optimizer", choices=["egd", "adamw", "sgdm"], default="egd")
    g.add_argument("--egd_lr", type=float, default=0.1)
    g.add_argument("--egd_eta", type=float, default=100.0)
    g.add_argument("--egd_F0", type=float, default=None,
                   help="loss offset; omit for initial_loss - auto_F0_margin")
    g.add_argument("--egd_nu", type=float, default=0.0)
    g.add_argument("--egd_eps1", type=float, default=1e-10)
    g.add_argument("--egd_eps2", type=float, default=1e-40)
    g.add_argument("--egd_wd", type=float, default=0.0)
    g.add_argument("--egd_consEn", action="store_true", default=True)
    g.add_argument("--no_egd_consEn", action="store_false", dest="egd_consEn")
    g.add_argument("--egd_seed", type=int, default=None)
    g.add_argument("--adam_lr", type=float, default=1e-4)
    g.add_argument("--adam_wd", type=float, default=0.01)
    g.add_argument("--adam_beta1", type=float, default=0.9)
    g.add_argument("--adam_beta2", type=float, default=0.95)
    g.add_argument("--sgdm_lr", type=float, default=0.03)
    g.add_argument("--sgdm_momentum", type=float, default=0.95)

    # --- train ------------------------------------------------------------- #
    g = ap.add_argument_group("training")
    g.add_argument("--batch_size", type=int, default=32)
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--max_steps", type=int, default=0, help="0 = epochs x steps/epoch")
    g.add_argument("--grad_clip", type=float, default=0.0, help="0 disables clipping")
    g.add_argument("--label_smoothing", type=float, default=0.0)
    g.add_argument("--valid_every_updates", type=int, default=200)
    g.add_argument("--log_every", type=int, default=50)
    g.add_argument("--save_every_updates", type=int, default=0, help="0 = only best/final")
    g.add_argument("--bleu_every", type=int, default=0, help="0 disables periodic BLEU")
    g.add_argument("--bleu_samples", type=int, default=200, help="0 = whole split")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--device", type=str, default="auto")
    g.add_argument("--num_workers", type=int, default=0)
    g.add_argument("--use_bf16", action="store_true",
                   help="autocast in bfloat16 where CUDA is available (no-op on CPU)")
    g.add_argument("--max_seconds", type=float, default=0.0,
                   help="hard wall-clock stop; 0 = no limit")
    g.add_argument("--stall_warn_seconds", type=float, default=300.0)
    g.add_argument("--max_eval_batches", type=int, default=0, help="0 = whole loader")

    # --- run --------------------------------------------------------------- #
    g = ap.add_argument_group("run")
    g.add_argument("--log_dir", type=str, default=S, help="root of the run directories")
    g.add_argument("--name", type=str, default=None, help="run name (default: generated)")
    g.add_argument("--resume", type=str, default=S,
                   help="checkpoint (.pt) to continue from; the step counter picks "
                        "up at its 'update' and its optimizer state is restored")
    g.add_argument("--wandb", action="store_true", default=S, help="log to Weights & Biases")
    g.add_argument("--no_plot", action="store_true", default=S,
                   help="skip training_curve.png at the end")
    return ap


def data_config_from_args(args) -> DataConfig:
    """Assemble a validated :class:`DataConfig` from the data flags.

    ``--dataset_preset`` seeds the keyword arguments first (its ``dataset`` entry
    is the ``DataConfig.source`` field); a flag that was actually given on the
    command line then wins over the preset field it stands for, so
    ``--dataset_preset multi30k-tiny --max_train_samples 500`` trains on 500
    pairs, not 8000.
    """
    namespace = vars(args)
    kwargs: Dict[str, Any] = {}
    if "dataset_preset" in namespace:
        preset = dict(DATASET_PRESETS[namespace["dataset_preset"]])
        if "dataset" in preset:
            preset["source"] = preset.pop("dataset")
        kwargs.update(preset)
    if "dataset" in namespace:
        kwargs["source"] = namespace["dataset"]
    # Everything a flag exists for; the rest of DataConfig keeps its default.
    flag_fields = (
        "max_train_samples", "max_val_samples", "data_dir", "hf_repo",
        "hf_endpoint", "src_field", "tgt_field", "synthetic_task",
        "synthetic_train_size", "synthetic_val_size", "synthetic_test_size",
        "synthetic_vocab", "synthetic_len", "tokenizer", "min_freq", "max_vocab",
        "lowercase", "max_src_len", "max_tgt_len", "fallback_to_synthetic",
    )
    for field in flag_fields:
        if field in namespace:
            kwargs[field] = namespace[field]
    return DataConfig(**kwargs).validate()


def clip_lengths(data_cfg: DataConfig, context_length: int) -> DataConfig:
    """Turn ``max_src_len``/``max_tgt_len`` of ``0`` into a real clipping budget.

    The ported ``data.dataset`` cannot see the model config, so in the new
    package ``0`` means **no clipping at all** (the old
    ``or cfg.model.max_seq_len`` fallback is gone -- see PORT_NOTES section B).
    The training script therefore owns the translation from "0 = context length"
    to a number, and must clamp anything larger: ``TokenPositionalEmbedding``
    indexes an ``nn.Embedding(context_length, n_embd)`` table, so a padded batch
    longer than ``context_length`` is an ``IndexError`` partway through training.

    The target budget applies to the **framed** ``[BOS] ... [EOS]`` list, which
    is what :class:`~symbreak_transformer.data.ParallelTextDataset` passes to
    ``Tokenizer.encode``; it uses ``tgt_ids[:-1]`` for the decoder input, so a
    framed target of exactly ``context_length`` ids fits the table with one
    position to spare.

    Args:
        data_cfg: the config assembled from the data flags (mutated in place).
        context_length: ``model_cfg.context_length``.

    Returns:
        The same config, for chaining.
    """
    budget = int(context_length)
    if budget < 1:
        raise ValueError(f"context_length must be >= 1, got {context_length}")
    for name in ("max_src_len", "max_tgt_len"):
        value = int(getattr(data_cfg, name) or 0)
        if value <= 0:
            setattr(data_cfg, name, budget)
        elif value > budget:
            print(
                f"[data] {name}={value} exceeds ctx={budget} and would overflow the "
                f"positional table; clamping to {budget}"
            )
            setattr(data_cfg, name, budget)
    return data_cfg


def model_config_from_args(args, src_vocab_size: int, tgt_vocab_size: int):
    """Build the validated :class:`Seq2SeqConfig` from the model-shape flags."""
    namespace = vars(args)
    overrides = {
        key: namespace[key]
        for key in (
            "n_embd", "n_head", "n_encoder_layer", "n_decoder_layer", "d_ff",
            "dropout", "attention_dropout", "activation", "prelu_random_init",
            "norm_first", "scale_embedding", "tie_output_embedding",
            "share_embeddings", "init_std",
        )
        if key in namespace
    }
    if namespace.get("use_prelu"):
        overrides["activation"] = "prelu"
    return resolve_config(
        namespace["model"],
        context_length=namespace.get("ctx"),
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        **overrides,
    )


def model_parameter_counts(vocab: int) -> None:
    """Print the exact parameter count of every model preset.

    Shapes differ by a handful of parameters, but the vocabulary dominates the
    total (``2 * vocab * n_embd`` for the two embedding tables), so the count has
    to be quoted for a concrete vocabulary: ``--vocab``.

    Each preset is built for real -- ``Seq2SeqTransformer`` needs non-zero vocab
    sizes, so nothing is estimated -- which costs a second for all five.

    Args:
        vocab: vocabulary size used for both sides.
    """
    vocab = int(vocab)
    print()
    print(
        f"exact parameter counts (src_vocab = tgt_vocab = {vocab}; the vocabulary "
        f"dominates these numbers, so pass --vocab to match your dataset):"
    )
    for name in PRESETS:
        try:
            cfg = resolve_config(name, src_vocab_size=vocab, tgt_vocab_size=vocab)
            model = Seq2SeqTransformer(
                cfg,
                BiasConfig(),
                src_vocab_size=vocab,
                tgt_vocab_size=vocab,
                pad_id=PAD_ID,
                bos_id=Tokenizer.bos_id,
                eos_id=Tokenizer.eos_id,
            )
        except Exception as exc:  # a malformed preset must not hide the others
            print(f"  {name:<8} {'n/a':>12} ({type(exc).__name__}: {exc})")
            continue
        print(f"  {name:<8} {count_parameters(model):>12,} params")


def bias_config_from_args(args) -> BiasConfig:
    """Build the validated :class:`BiasConfig` from the bias flags.

    ``--symmetric`` wins outright (all-off config).  Otherwise the run starts
    from ``--bias_preset`` (or the ``BiasConfig`` defaults) and every bias flag
    that was actually passed on the command line overrides one field of it.
    """
    namespace = vars(args)
    if namespace.get("symmetric"):
        return BiasConfig().validate()

    field_for_flag = {
        "bias_mode": "embed_mode",
        "bias_mean": "embed_mean",
        "bias_std": "embed_std",
        "bias_const": "embed_const_value",
        "bias_resample": "embed_resample",
        "bias_learnable": "embed_learnable",
        "attn_mode": "attn_mode",
        "attn_resample": "attn_resample",
        "attn_learnable": "attn_learnable",
        "use_q_bias": "use_q_bias",
        "use_k_bias": "use_k_bias",
        "use_v_bias": "use_v_bias",
        "mean_Q": "mean_Q",
        "std_Q": "std_Q",
        "mean_K": "mean_K",
        "std_K": "std_K",
        "mean_V": "mean_V",
        "std_V": "std_V",
        "share_across_heads": "share_across_heads",
        "share_across_layers": "share_across_layers",
        "bias_seed": "embed_seed",
    }
    overrides = {
        field: namespace[flag]
        for flag, field in field_for_flag.items()
        if flag in namespace
    }
    return resolve_bias_config(namespace.get("bias_preset"), **overrides)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def main() -> None:
    """Parse the flags, build data/model/optimizer and run the training loop."""
    # The corpus (multi30k German) is non-ASCII and this console is GBK.
    configure_console_encoding()

    ap = build_parser()
    args = ap.parse_args()
    namespace = vars(args)

    if args.list_models:
        print(preset_table())
        model_parameter_counts(args.vocab)
        print()
        print("example:")
        print(
            "  python main.py train --model small --bias_preset b-gaussian "
            "--dataset_preset multi30k-tiny"
        )
        print(
            "  python main.py train --model small --bias_preset b-const "
            "--dataset_preset multi30k-tiny --bias_const 1.0"
        )
        print(
            "  python main.py train --model smoke --dataset_preset synthetic-reverse "
            "--max_steps 60"
        )
        raise SystemExit(0)

    log_dir = namespace.get("log_dir", "runs")
    seed = args.seed
    device = resolve_device(args.device)
    timer = Timer()
    seed_everything(seed)

    # ---------------- config ---------------- #
    data_cfg = data_config_from_args(args)

    print("[data] loading parallel text ...")
    splits_needed = ["train", "val"]
    pairs = load_all_splits(data_cfg, splits_needed + ["test"])
    train_pairs = pairs.get("train") or []
    if not train_pairs:
        raise SystemExit(
            "the training split is empty; check --dataset / --data_dir / --hf_repo"
        )
    test_pairs = pairs.get("test") or []
    print(
        f"[data] source={data_cfg.source} train={len(train_pairs)} "
        f"val={len(pairs.get('val') or [])} test={len(test_pairs)}"
    )

    src_tokenizer, tgt_tokenizer = build_tokenizers(data_cfg, {"train": train_pairs})
    cfg = model_config_from_args(args, src_tokenizer.vocab_size, tgt_tokenizer.vocab_size)
    bias_cfg = bias_config_from_args(args)
    # The data loader cannot see the model config, so "0 = context length" is
    # resolved here, before anything is batched (see clip_lengths).
    clip_lengths(data_cfg, cfg.context_length)
    print(
        f"[data] tokenizer={data_cfg.tokenizer} src_vocab={src_tokenizer.vocab_size} "
        f"tgt_vocab={tgt_tokenizer.vocab_size} "
        f"clip_src={data_cfg.max_src_len} clip_tgt={data_cfg.max_tgt_len}"
    )

    # ---------------- run directory ---------------- #
    name = namespace.get("name") or run_name_for(args.model, args.optimizer, bias_cfg, seed)
    run_dir = Path(log_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- banner ---------------- #
    print("=== Configuration ===")
    print(f"  run         : {name}")
    print(f"  dir         : {run_dir}")
    print(f"  model       : {args.model} (n_embd={cfg.n_embd} n_head={cfg.n_head} "
          f"enc/dec={cfg.n_encoder_layer}/{cfg.n_decoder_layer} d_ff={cfg.d_ff} "
          f"ctx={cfg.context_length})")
    print(f"  activation  : {cfg.activation} (prelu_random_init={cfg.prelu_random_init})")
    print(f"  norm_first  : {cfg.norm_first} | dropout={cfg.dropout} "
          f"attention_dropout={cfg.attention_dropout}")
    print(f"  bias        : {bias_cfg.describe()}")
    print(f"  optimizer   : {args.optimizer}")
    print(f"  schedule    : batch_size={args.batch_size} epochs={args.epochs} "
          f"max_steps={args.max_steps or 'from epochs'}")
    print(f"  device      : {device} | bf16={bool(args.use_bf16 and torch.cuda.is_available())}")
    print(f"  seed        : {seed}")
    print(f"  environment : {environment_report()}")
    print("=====================")

    # Everything that defines the run is archived next to the checkpoints.
    _dump_json(run_dir / "args.json", namespace)
    _dump_json(run_dir / "config.json", cfg.to_dict())
    _dump_json(run_dir / "bias.json", bias_cfg.to_dict())
    src_tokenizer.save(run_dir / "tokenizer_src.json")
    tgt_tokenizer.save(run_dir / "tokenizer_tgt.json")

    # ---------------- data ---------------- #
    loaders = build_dataloaders(
        data_cfg,
        src_tokenizer,
        tgt_tokenizer,
        splits=["train", "val"],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=seed,
    )
    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    # ---------------- model ---------------- #
    model = Seq2SeqTransformer(
        cfg,
        bias_cfg,
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        pad_id=PAD_ID,
        bos_id=Tokenizer.bos_id,
        eos_id=Tokenizer.eos_id,
    ).to(device)
    n_params = count_parameters(model)
    print(f"[model] {n_params:,} trainable parameters on {device}")

    # ---------------- optimizer ---------------- #
    if args.optimizer == "egd":
        kwargs = dict(
            lr=args.egd_lr, eta=args.egd_eta, F0=args.egd_F0, nu=args.egd_nu,
            eps1=args.egd_eps1, eps2=args.egd_eps2, weight_decay=args.egd_wd,
            consEn=args.egd_consEn, seed=args.egd_seed,
        )
        optimizer = build_optimizer(model, "egd", egd_kwargs=kwargs)
        opt_summary = ", ".join(f"{k}={v}" for k, v in kwargs.items())
    elif args.optimizer == "adamw":
        kwargs = dict(
            lr=args.adam_lr, weight_decay=args.adam_wd,
            betas=(args.adam_beta1, args.adam_beta2),
        )
        optimizer = build_optimizer(model, "adamw", adamw_kwargs=kwargs)
        opt_summary = f"lr={args.adam_lr}, weight_decay={args.adam_wd}, " \
                      f"betas=({args.adam_beta1}, {args.adam_beta2})"
    else:  # sgdm
        kwargs = dict(lr=args.sgdm_lr, momentum=args.sgdm_momentum)
        optimizer = build_optimizer(model, "sgdm", sgdm_kwargs=kwargs)
        opt_summary = f"lr={args.sgdm_lr}, momentum={args.sgdm_momentum}"
    print(f"[opt] {type(optimizer).__name__} ({opt_summary})")

    # ---------------- resume ---------------- #
    start_step = 0
    prior_best_val = float("inf")
    if namespace.get("resume"):
        start_step, prior_best_val = _restore_checkpoint(
            Path(namespace["resume"]), model, optimizer, device
        )

    # bf16 autocast only exists on CUDA; on this CPU-only machine the flag is a
    # no-op, exactly as upstream (``enabled=use_bf16 and cuda``).
    amp_dtype = (
        torch.bfloat16
        if (args.use_bf16 and torch.cuda.is_available())
        else torch.float32
    )
    use_amp = amp_dtype is torch.bfloat16

    # Optional Weights & Biases, guarded exactly like upstream.
    if namespace.get("wandb"):
        if WANDB_AVAILABLE:
            wandb.init(
                project="symbreak-transformer",
                name=name,
                tags=[args.model, args.optimizer, bias_tag(bias_cfg)],
                config={
                    "model": cfg.to_dict(),
                    "bias": bias_cfg.to_dict(),
                    "data": data_cfg.to_dict(),
                    "params": n_params,
                    "optimizer": args.optimizer,
                    "argv": sys.argv,
                },
            )
            print(f"[wandb] logging to {name}")
        else:
            print("[wandb] --wandb given but the wandb package is not installed; skipping")

    # ---------------- loop setup ---------------- #
    steps_per_epoch = max(1, len(train_loader))
    total_steps = args.max_steps or (args.epochs * steps_per_epoch)
    per_step_bias = (
        bias_cfg.embed_resample == "per_step"
        or (bias_cfg.attention_enabled and bias_cfg.attn_resample == "per_step")
    )
    if per_step_bias:
        print("[bias] per_step resampling enabled: biases are redrawn every step")

    losses_csv = run_dir / "losses.csv"
    if start_step and losses_csv.exists():
        print(f"[log] appending to the existing per-update losses ({losses_csv.name})")
    else:
        with losses_csv.open("w") as fh:
            fh.write("tag,loss\n")  # one row per update; see _record_loss

    # Bias-norm diagnostics are best-effort: a failure is noted once and said
    # out loud in the summary, it never takes the run down.
    bias_note: Dict[str, Any] = {"warned": False, "error": ""}

    # A resumed run appends to its own log; a fresh run truncates it, so a
    # re-run never silently grows a stale curve.
    logger = _open_csv_logger(
        run_dir / "training_log.csv", LOG_FIELDS, append=bool(start_step)
    )
    logger.write_meta(
        {
            "run_name": name,
            "environment": environment_report(),
            "device": str(device),
            "parameters": n_params,
            "config": cfg.to_dict(),
            "bias": bias_cfg.to_dict(),
            "data": data_cfg.to_dict(),
            "resumed_from": namespace.get("resume") or None,
            "start_step": start_step,
        }
    )
    print(f"[log] {run_dir / 'training_log.csv'}")

    watchdog = StepWatchdog(args.stall_warn_seconds)
    print(
        f"[train] {total_steps} steps "
        f"({args.epochs} epoch(s) x {steps_per_epoch} steps/ep)"
        + (f", starting at step {start_step}" if start_step else "")
    )

    def closure(batch):
        """zero_grad -> forward -> backward -> optional clip -> loss (EGD needs it)."""
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, enabled=use_amp, dtype=amp_dtype
        ):
            out = model(
                batch["src"].to(device),
                batch["tgt_in"].to(device),
                src_padding_mask=batch["src_padding_mask"].to(device),
                tgt_padding_mask=batch["tgt_padding_mask"].to(device),
                labels=batch["tgt_out"].to(device),
                label_smoothing=args.label_smoothing,
            )
        loss = out["loss"]
        if loss is None:
            raise RuntimeError("model returned no loss; labels were not wired through")
        loss.backward()
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        return loss

    best_val = prior_best_val
    best_update = start_step
    last_val = float("nan")
    # A resumed run continues towards total_steps instead of restarting; the
    # dataloader just iterates from its beginning again.
    step = start_step
    epoch = 0
    consumed_tokens = 0
    stop_reason = "completed"
    stop_training = False
    running_loss = 0.0
    running_tokens = 0
    window_start = time.time()
    model.train()

    try:
        # ``stop_training`` -- not just ``break`` -- is required: the max_seconds
        # guard fires inside the inner ``for`` loop, and breaking only that loop
        # would leave the outer ``while`` spinning without ever stepping again.
        while step < total_steps and not stop_training:
            for batch in train_loader:
                if step >= total_steps:
                    break
                if args.max_seconds > 0 and timer.elapsed > args.max_seconds:
                    stop_reason = (
                        f"stopped by --max_seconds={args.max_seconds} after "
                        f"{format_seconds(timer.elapsed)}"
                    )
                    print(f"[stop] {stop_reason}")
                    stop_training = True
                    break

                if per_step_bias:
                    model.resample_biases()

                # One step for both optimizer families: EGD consumes the
                # closure, the others raise TypeError and are stepped normally.
                try:
                    loss_tensor = optimizer.step(lambda: closure(batch))
                except TypeError:
                    loss_tensor = closure(batch)
                    optimizer.step()
                if loss_tensor is None:
                    raise RuntimeError("optimizer returned no loss")

                loss_value = float(loss_tensor.detach().item())
                _record_loss(losses_csv, loss_value)
                dt = watchdog.tick(step)

                n_tokens = int((batch["tgt_out"] != PAD_ID).sum().item())
                running_loss += loss_value * max(n_tokens, 1)
                running_tokens += max(n_tokens, 1)
                consumed_tokens += int(batch["src"].numel())
                step += 1

                # --- periodic train log --- #
                if args.log_every > 0 and step % args.log_every == 0:
                    window = max(time.time() - window_start, 1e-9)
                    mean_loss = running_loss / max(running_tokens, 1)
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": mean_loss,
                        "train_ppl": math.exp(min(mean_loss, 50.0)),
                        "sec_per_step": dt,
                        "tokens_per_sec": running_tokens / window,
                        "best_val_loss": best_val if math.isfinite(best_val) else "",
                    }
                    row.update(_row_extra(model, optimizer, timer.elapsed, bias_note))
                    logger.log(row)
                    print(
                        f"[step {step:6d}/{total_steps}] loss {mean_loss:.4f} "
                        f"ppl {row['train_ppl']:8.2f} | {row['tokens_per_sec']:7.0f} tok/s "
                        f"| {timer.pretty}",
                        flush=True,
                    )
                    if namespace.get("wandb") and WANDB_AVAILABLE:
                        wandb.log(
                            {
                                "train/loss": mean_loss,
                                "speed/tok_per_s": row["tokens_per_sec"],
                                "train/consumed_tokens": consumed_tokens,
                                "update": step,
                            }
                        )
                    running_loss = 0.0
                    running_tokens = 0
                    window_start = time.time()

                # --- periodic validation / checkpointing --- #
                do_eval = val_loader is not None and (
                    (args.valid_every_updates > 0
                     and step % args.valid_every_updates == 0)
                    or step == total_steps
                )
                if do_eval:
                    val_loss = evaluate_loss(
                        model, val_loader, device, pad_id=PAD_ID,
                        max_batches=args.max_eval_batches,
                    )
                    last_val = val_loss
                    improved = math.isfinite(val_loss) and val_loss < best_val
                    if improved:
                        best_val = val_loss
                        best_update = step
                        _save_checkpoint(
                            run_dir / "model_best.pt", model, optimizer, cfg, bias_cfg,
                            step, epoch, val_loss, consumed_tokens,
                        )
                    val_ppl = (
                        math.exp(min(val_loss, 50.0))
                        if math.isfinite(val_loss)
                        else float("nan")
                    )
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "best_val_loss": best_val if math.isfinite(best_val) else "",
                    }
                    row.update(_row_extra(model, optimizer, timer.elapsed, bias_note))
                    logger.log(row)
                    print(
                        f"[step {step:6d}] val loss {val_loss:.4f} ppl {val_ppl:.2f}"
                        f"{' (best, saved)' if improved else ''} | {timer.pretty}",
                        flush=True,
                    )
                    if namespace.get("wandb") and WANDB_AVAILABLE:
                        wandb.log({"val/loss": val_loss, "update": step})
                    model.train()

                    # --- periodic greedy BLEU on the test split (when asked) --- #
                    if (
                        args.bleu_every > 0
                        and step % args.bleu_every == 0
                        and test_pairs
                    ):
                        bleu = evaluate_bleu(
                            model,
                            test_pairs,
                            src_tokenizer,
                            tgt_tokenizer,
                            device,
                            max_len=cfg.context_length,
                            max_samples=args.bleu_samples,
                            batch_size=args.batch_size,
                        )
                        print(
                            f"[step {step:6d}] test BLEU {bleu['bleu']:.2f} "
                            f"(strict {bleu['bleu_strict']:.2f}) on "
                            f"{bleu['n_samples']} pairs"
                        )
                        if namespace.get("wandb") and WANDB_AVAILABLE:
                            wandb.log({"test/bleu": bleu["bleu"], "update": step})
                        model.train()

                # --- periodic checkpoint --- #
                if args.save_every_updates and step % args.save_every_updates == 0:
                    _save_checkpoint(
                        run_dir / f"model_{step:06d}.pt", model, optimizer, cfg,
                        bias_cfg, step, epoch,
                        last_val if math.isfinite(last_val) else 0.0, consumed_tokens,
                    )
                    print(f"[ckpt] wrote model_{step:06d}.pt")

            epoch += 1
            if step >= total_steps:
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted by user"
        print(f"[stop] {stop_reason}; saving what we have")

    # ---------------- wrap up ---------------- #
    if val_loader is not None and not math.isfinite(best_val):
        best_val = evaluate_loss(
            model, val_loader, device, pad_id=PAD_ID, max_batches=args.max_eval_batches
        )
        best_update = step

    final_path = _save_checkpoint(
        run_dir / "model_final.pt", model, optimizer, cfg, bias_cfg, step, epoch,
        best_val if math.isfinite(best_val) else 0.0, consumed_tokens,
    )

    summary = {
        "run_name": name,
        "run_dir": str(run_dir),
        "steps": step,
        "epochs": epoch,
        "parameters": n_params,
        "best_val_loss": best_val if math.isfinite(best_val) else None,
        "best_update": best_update,
        "start_step": start_step,
        "resumed_from": namespace.get("resume") or None,
        "elapsed_sec": timer.elapsed,
        "stop_reason": stop_reason,
        "device": str(device),
        "optimizer": type(optimizer).__name__,
        "bias_report": {
            k: v for k, v in _bias_report(model, bias_note).items()
            if isinstance(v, (int, float, str))
        },
        "bias_report_error": bias_note["error"] or None,
        "csv": str(run_dir / "training_log.csv"),
    }
    _dump_json(run_dir / "summary.json", summary)

    if watchdog.warned:
        print(
            f"[stall-watchdog] {watchdog.warned} step(s) exceeded "
            f"{format_seconds(args.stall_warn_seconds)}; slowest step "
            f"{format_seconds(watchdog.max_seen)}"
        )

    # The training curve: matplotlib is imported lazily by plot_curve, and the
    # plot is skipped (with a one-line note) when it is unavailable.
    if not namespace.get("no_plot"):
        curve = load_plot_curve()
        if curve is None:
            print("[plot] skipped: scripts/plot_curve.py could not be imported")
        else:
            curve(run_dir / "training_log.csv", title=name)

    best_text = f"{best_val:.4f}" if math.isfinite(best_val) else "n/a"
    print(f"\nDone. Best val loss: {best_text}")
    print(f"Saved to {final_path}. Total time: {time.time() - timer.start:.1f}s")

    if namespace.get("wandb") and WANDB_AVAILABLE:
        wandb.finish()


def _dump_json(path: Path, payload: Dict[str, Any]) -> Path:
    """Write ``payload`` as JSON, falling back to ``str`` for odd values."""
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    return path


def load_plot_curve():
    """Import ``scripts/plot_curve.py`` by path, wherever it is invoked from.

    The script is run three ways -- ``python scripts/train.py`` (the script
    directory is on ``sys.path``), ``python -m scripts.train`` and through
    ``main.py``'s ``runpy`` dispatch (neither is) -- so the sibling module is
    loaded from its file rather than by a plain ``import``.

    Returns:
        The ``plot_curve`` callable, or ``None`` when the file cannot be loaded
        (plotting is optional and must never kill a finished run).
    """
    try:
        import importlib.util

        path = Path(__file__).resolve().parent / "plot_curve.py"
        spec = importlib.util.spec_from_file_location("_symbreak_plot_curve", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.plot_curve
    except Exception as exc:  # pragma: no cover - optional convenience
        print(f"[plot] could not import plot_curve ({type(exc).__name__}: {exc})")
        return None


if __name__ == "__main__":
    main()
