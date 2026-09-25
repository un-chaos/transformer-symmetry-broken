"""
Which command-line flag sets which configuration field.

This module is **the** place where the CLI surface and the configuration
dataclasses are tied together.  Keeping that mapping in one table is what makes
"every knob is configurable" a checkable property instead of a promise: the tests
assert that every field of every config dataclass appears here, so a newly added
knob can never quietly end up unreachable from the command line -- and therefore
from the menu's parameter editor in ``run.py``, which reads these same tables.

Two kinds of entry:

* ``<CLASS>_FIELD_TO_DEST``: ``dataclass field name -> argparse dest``.  The dest
  is what appears in ``vars(args)``, which is why ``embed_mode`` maps to the dest
  ``bias_mode`` (the flag is ``--bias_mode``) and ``source`` maps to ``dataset``.
* ``<CLASS>_NOT_EXPOSED``: fields deliberately without a flag, each with the
  reason.  These are the only places a value cannot come from the command line,
  so they are listed explicitly rather than being silently hardcoded.

For the non-dataclass knobs (the training loop, the optimizer, the run directory)
the tables are identity maps; they exist so the parameter editor can list every
flag with a comment instead of relying on a hand-maintained list.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

__all__ = [
    "MODEL_FIELD_TO_DEST",
    "MODEL_NOT_EXPOSED",
    "BIAS_FIELD_TO_DEST",
    "BIAS_NOT_EXPOSED",
    "DATA_FIELD_TO_DEST",
    "DATA_NOT_EXPOSED",
    "SCHEDULE_FIELD_TO_DEST",
    "OPTIMIZER_FIELD_TO_DEST",
    "RUN_FIELD_TO_DEST",
    "field_overrides",
    "collect_fields",
    "assert_complete",
]


# --------------------------------------------------------------------------- #
# Model shape -> `Seq2SeqConfig`
# --------------------------------------------------------------------------- #
MODEL_FIELD_TO_DEST: Dict[str, str] = {
    "context_length": "ctx",
    "n_encoder_layer": "n_encoder_layer",
    "n_decoder_layer": "n_decoder_layer",
    "n_head": "n_head",
    "n_embd": "n_embd",
    "d_ff": "d_ff",
    "dropout": "dropout",
    "attention_dropout": "attention_dropout",
    "activation": "activation",
    "prelu_random_init": "prelu_random_init",
    "prelu_slope_mean": "prelu_slope_mean",
    "prelu_slope_std": "prelu_slope_std",
    "norm_first": "norm_first",
    "scale_embedding": "scale_embedding",
    "tie_output_embedding": "tie_output_embedding",
    "share_embeddings": "share_embeddings",
    "init_std": "init_std",
    "scaled_residual_init": "scaled_residual_init",
}
MODEL_NOT_EXPOSED: Dict[str, str] = {
    # These two are *outputs* of the data pipeline, not inputs: the tokenizer
    # decides them.  Exposing them as flags would let a user build a model whose
    # vocabulary disagrees with its data.  Use --max_vocab (data) instead.
    "src_vocab_size": "derived from the source tokenizer; use --max_vocab",
    "tgt_vocab_size": "derived from the target tokenizer; use --max_vocab",
}

# --------------------------------------------------------------------------- #
# Bias -> `BiasConfig`
# --------------------------------------------------------------------------- #
BIAS_FIELD_TO_DEST: Dict[str, str] = {
    "embed_mode": "bias_mode",
    "embed_mean": "bias_mean",
    "embed_std": "bias_std",
    "embed_const_value": "bias_const",
    "embed_resample": "bias_resample",
    "embed_learnable": "bias_learnable",
    "embed_seed": "bias_seed",
    "use_q_bias": "use_q_bias",
    "use_k_bias": "use_k_bias",
    "use_v_bias": "use_v_bias",
    "attn_mode": "attn_mode",
    "attn_resample": "attn_resample",
    "attn_learnable": "attn_learnable",
    "mean_Q": "mean_Q",
    "std_Q": "std_Q",
    "mean_K": "mean_K",
    "std_K": "std_K",
    "mean_V": "mean_V",
    "std_V": "std_V",
    "const_value": "attn_const",
    "share_across_heads": "share_across_heads",
    "share_across_layers": "share_across_layers",
    "apply_encoder": "apply_encoder",
    "apply_decoder_self": "apply_decoder_self",
    "apply_decoder_cross": "apply_decoder_cross",
    "attn_seed": "attn_seed",
}
BIAS_NOT_EXPOSED: Dict[str, str] = {}

# --------------------------------------------------------------------------- #
# Data -> `DataConfig`
# --------------------------------------------------------------------------- #
DATA_FIELD_TO_DEST: Dict[str, str] = {
    "source": "dataset",
    "data_dir": "data_dir",
    "objective": "objective",
    "hf_repo": "hf_repo",
    "hf_endpoint": "hf_endpoint",
    "src_field": "src_field",
    "tgt_field": "tgt_field",
    "user_agent": "user_agent",
    "download_timeout": "download_timeout",
    "max_train_samples": "max_train_samples",
    "max_val_samples": "max_val_samples",
    "fallback_to_synthetic": "fallback_to_synthetic",
    "local_src_col": "local_src_col",
    "local_tgt_col": "local_tgt_col",
    "synthetic_task": "synthetic_task",
    "synthetic_train_size": "synthetic_train_size",
    "synthetic_val_size": "synthetic_val_size",
    "synthetic_test_size": "synthetic_test_size",
    "synthetic_vocab": "synthetic_vocab",
    "synthetic_len": "synthetic_len",
    "synthetic_seed": "synthetic_seed",
    "fineweb_dir": "fineweb_dir",
    "text_column": "text_column",
    "max_documents": "max_documents",
    "val_every": "val_every",
    "noise_density": "noise_density",
    "mean_span_length": "mean_span_length",
    "bpe_vocab_size": "bpe_vocab_size",
    "tokenizer_train_documents": "tokenizer_train_documents",
    "tokenizer": "tokenizer",
    "lowercase": "lowercase",
    "min_freq": "min_freq",
    "max_vocab": "max_vocab",
    "max_src_len": "max_src_len",
    "max_tgt_len": "max_tgt_len",
}
DATA_NOT_EXPOSED: Dict[str, str] = {
    # A per-split filename mapping; it points at the *files* inside one specific
    # dataset repository.  A scalar flag cannot express it, so a different layout
    # means a different dataset preset (or source="local"), not a flag.
    "hf_files": "per-split filename mapping of one repository; use a dataset preset",
}

# --------------------------------------------------------------------------- #
# Training loop / optimizer / run -- plain flags, no dataclass behind them
# --------------------------------------------------------------------------- #
SCHEDULE_FIELD_TO_DEST: Dict[str, str] = {
    name: name
    for name in (
        "batch_size", "epochs", "max_steps", "grad_clip", "label_smoothing",
        "valid_every_updates", "log_every", "save_every_updates", "bleu_every",
        "bleu_samples", "seed", "device", "num_workers", "use_bf16",
        "max_seconds", "stall_warn_seconds", "max_eval_batches",
    )
}

OPTIMIZER_FIELD_TO_DEST: Dict[str, str] = {
    name: name
    for name in (
        "optimizer", "egd_lr", "egd_eta", "egd_F0", "egd_nu", "egd_eps1",
        "egd_eps2", "egd_wd", "egd_consEn", "egd_auto_F0", "egd_auto_F0_margin",
        "egd_seed", "adam_lr", "adam_wd", "adam_beta1", "adam_beta2",
        "sgdm_lr", "sgdm_momentum",
    )
}

RUN_FIELD_TO_DEST: Dict[str, str] = {
    name: name for name in ("log_dir", "name", "wandb", "no_plot", "resume")
}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def field_overrides(
    table: Mapping[str, str], namespace: Mapping[str, Any]
) -> Dict[str, Any]:
    """
    Map a parsed-argument namespace onto dataclass field values.

    Only dests that are actually *present* in the namespace are taken: the
    training script uses ``argparse.SUPPRESS`` defaults, so "flag not given" is
    distinguishable from "flag given the preset's value", and a flag that was not
    passed must not override a dataset/bias preset.

    Args:
        table: a ``FIELD_TO_DEST`` mapping.
        namespace: ``vars(args)``.

    Returns:
        ``{field: value}`` for every mapped dest present in the namespace.
    """
    return {
        field: namespace[dest]
        for field, dest in table.items()
        if dest in namespace
    }


def collect_fields(
    tables: Mapping[str, Mapping[str, str]], namespace: Mapping[str, Any]
) -> Dict[str, Any]:
    """Merge :func:`field_overrides` over several tables (later tables win)."""
    merged: Dict[str, Any] = {}
    for table in tables.values():
        merged.update(field_overrides(table, namespace))
    return merged


def assert_complete(cls: type, table: Mapping[str, str], not_exposed: Mapping[str, str],
                    label: str) -> Optional[str]:
    """
    Check that every dataclass field is either mapped or explicitly excused.

    Returns:
        ``None`` when complete, otherwise a human-readable list of the fields that
        are neither mapped nor excused (a silent hole in the CLI surface).
    """
    import dataclasses

    fields = {f.name for f in dataclasses.fields(cls)}
    mapped = set(table)
    excused = set(not_exposed)
    holes = sorted(fields - mapped - excused)
    overlaps = sorted(mapped & excused)
    problems = []
    if holes:
        problems.append(f"{label}: no flag and no reason given: {holes}")
    if overlaps:
        problems.append(f"{label}: both mapped and excused: {overlaps}")
    unknown = sorted((mapped | excused) - fields)
    if unknown:
        problems.append(f"{label}: table names fields that do not exist: {unknown}")
    return "; ".join(problems) if problems else None
