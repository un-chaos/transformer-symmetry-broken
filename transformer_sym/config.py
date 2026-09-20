"""Typed configuration objects and YAML I/O.

Everything the user is expected to tune lives here:

* network shape (``d_model``, layer counts, heads, ``d_ff``, dropout, ...)
* the embedding bias ``b`` -- ``zero`` / ``gaussian`` / ``const`` (see ``bias.py``)
* the reference-style per-head attention biases ``bQ/bK/bV``
* the optimizer (``egd`` = energy-conserving descent, or an ``adamw`` baseline)
* the data source (HuggingFace mirror, synthetic, or local files)

Two ways in, both defined in this file:

1. **Named presets** (the reference project's style) -- pick from
   :data:`MODEL_PRESETS`, :data:`DATA_PRESETS` and :data:`BIAS_PRESETS`::

       cfg = make_config(model="small", bias="b-gaussian", data="multi30k")

   or from the command line::

       python main.py --model small --bias b-gaussian --data multi30k --set train.egd.lr=0.05

2. **YAML files** -- a whole experiment in one file, see ``configs/``::

       cfg = load_config("configs/small_multi30k.yaml")

The config is intentionally explicit: unknown keys raise an error rather than
being silently ignored, so a typo in a YAML file cannot silently change an
experiment.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import yaml

Scalar = Union[float, int]
VecOrScalar = Union[float, List[float]]

#: Allowed values for the ``mode`` of every bias (the three choices requested).
BIAS_MODES: tuple = ("zero", "gaussian", "const")
#: ``fixed`` draws once at init; ``per_step`` redraws on every optimizer step.
RESAMPLE_MODES: tuple = ("fixed", "per_step")
ACTIVATIONS: tuple = ("gelu", "relu", "prelu")
OPTIMIZERS: tuple = ("egd", "adamw")
DATA_SOURCES: tuple = ("hf", "synthetic", "local")
TOKENIZER_MODES: tuple = ("word", "char")
SYNTHETIC_TASKS: tuple = ("copy", "reverse", "sort")


# --------------------------------------------------------------------------- #
# Bias configuration
# --------------------------------------------------------------------------- #
@dataclass
class EmbeddingBiasConfig:
    """The bias ``b`` added to the embedding: ``x = Embed(tokens) * scale + b``.

    ``mode`` is the three-way switch requested for the experiments:

    ``zero``
        ``b = 0``.  The embedding stays rotationally symmetric and ``O(d_model)``
        is an exact symmetry of the Q-K sector.
    ``gaussian``
        ``b ~ N(mean, std^2)``, drawn element-wise over the ``d_model``
        dimensions.  Non-zero with probability 1, so the rotation symmetry is
        broken.  ``b`` is *fixed* for the run unless ``resample: per_step``.
    ``const``
        ``b = const_value`` in every dimension.  Also breaks the rotation
        symmetry (like the other non-zero modes, but with a fully isotropic,
        rank-one direction).

    ``learnable: true`` turns ``b`` into an ``nn.Parameter`` that the optimizer
    updates; that requires ``resample: fixed``.
    """

    mode: str = "zero"
    resample: str = "fixed"
    mean: VecOrScalar = 0.0
    std: VecOrScalar = 0.02
    const_value: float = 1.0
    learnable: bool = False
    seed: int = 1234


@dataclass
class AttentionBiasConfig:
    """Per-head additive biases ``bQ``/``bK``/``bV`` inside attention.

    Mirrors the reference implementation (``Symmetry-breaking-attention-bias``):
    ``bQ`` breaks the ``O(d_head)`` rotation symmetry of the Q-K sector, with an
    exponentially amplified effect because it enters through the softmax.  ``bK``
    is available but **off by default**: a key-independent part of a constant
    shift cancels in the softmax normalisation.  ``bV`` acts on the V-O sector
    with a power-law (purely linear) effect.

    The same ``mode`` switch (``zero``/``gaussian``/``const``) applies to all
    three sectors; each sector keeps its own ``mean``/``std``.
    """

    enabled: bool = False
    mode: str = "gaussian"
    resample: str = "fixed"
    const_value: float = 1.0
    learnable: bool = False

    #: With ``True`` all heads share one ``head_dim`` vector (reference behaviour);
    #: with ``False`` every head draws an independent ``(n_heads, head_dim)`` block.
    share_across_heads: bool = True
    #: With ``True`` one bias object is shared by every layer.
    share_across_layers: bool = True

    #: Which attention sites receive a bias.
    apply_encoder: bool = True
    apply_decoder_self: bool = True
    apply_decoder_cross: bool = True

    #: Per-sector switches and distributions.
    q_enabled: bool = True
    q_mean: VecOrScalar = 0.5
    q_std: VecOrScalar = 0.05

    k_enabled: bool = False
    k_mean: VecOrScalar = 0.3
    k_std: VecOrScalar = 0.1

    v_enabled: bool = True
    v_mean: VecOrScalar = 0.5
    v_std: VecOrScalar = 0.05

    seed: int = 1234


@dataclass
class BiasConfig:
    """Container for both bias families."""

    embed: EmbeddingBiasConfig = field(default_factory=EmbeddingBiasConfig)
    attention: AttentionBiasConfig = field(default_factory=AttentionBiasConfig)


# --------------------------------------------------------------------------- #
# Model / data / training configuration
# --------------------------------------------------------------------------- #
@dataclass
class ModelConfig:
    """Encoder-decoder Transformer shape.

    ``vocab_size_src`` / ``vocab_size_tgt`` are resolved at runtime from the
    tokenizer when left at ``0``.
    """

    vocab_size_src: int = 0
    vocab_size_tgt: int = 0

    d_model: int = 256
    n_encoder_layers: int = 3
    n_decoder_layers: int = 3
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.0
    max_seq_len: int = 128

    activation: str = "gelu"
    #: Only used when ``activation == "prelu"``: break the activation symmetry
    #: with per-feature negative slopes drawn from ``N(mean, std^2)``.
    prelu_random_init: bool = False
    prelu_slope_mean: float = 0.2
    prelu_slope_std: float = 1.0

    norm_first: bool = False
    scale_embedding: bool = True
    #: Tie the decoder output projection to the target embedding matrix.
    tie_output_embedding: bool = True
    #: Share one embedding matrix between source and target vocabularies.
    #: Only valid when both vocabularies are identical.
    share_embeddings: bool = False
    init_std: float = 0.02
    #: Multiply the output-projection init by ``(2 * n_layers) ** -0.5``
    #: (nanoGPT-style residual scaling).
    scaled_residual_init: bool = True


@dataclass
class DataConfig:
    """Where the parallel text comes from and how it is tokenized."""

    source: str = "hf"
    data_dir: str = "data"

    # --- source == "hf" ---
    #: huggingface.co is not reachable from this machine; the mirror is used.
    hf_repo: str = "bentrevett/multi30k"
    hf_endpoint: str = "https://hf-mirror.com"
    hf_files: Dict[str, str] = field(
        default_factory=lambda: {
            "train": "train.jsonl",
            "val": "val.jsonl",
            "test": "test.jsonl",
        }
    )
    src_field: str = "de"
    tgt_field: str = "en"
    #: Sent as ``User-Agent``; the mirror answers 403 without one.
    user_agent: str = "Mozilla/5.0 (compatible; transformer-sym/0.1)"
    #: Seconds before a download attempt is abandoned.
    download_timeout: float = 120.0
    #: 0 disables the limit.
    max_train_samples: int = 0
    max_val_samples: int = 0
    #: If a HuggingFace download fails (offline / blocked host), silently fall
    #: back to the synthetic task instead of aborting the run.
    fallback_to_synthetic: bool = True

    # --- source == "local": ``<data_dir>/local/<split>.tsv`` (no header) ---
    local_src_col: int = 0
    local_tgt_col: int = 1

    # --- source == "synthetic" (offline fallback / debugging) ---
    synthetic_task: str = "reverse"
    synthetic_train_size: int = 4000
    synthetic_val_size: int = 500
    synthetic_test_size: int = 500
    synthetic_vocab: int = 20
    synthetic_len: int = 8
    synthetic_seed: int = 1234

    # --- tokenizer ---
    tokenizer: str = "word"
    lowercase: bool = True
    min_freq: int = 2
    max_vocab: int = 30000
    #: Clip lengths (0 -> ``model.max_seq_len``).
    max_src_len: int = 0
    max_tgt_len: int = 0


@dataclass
class EGDConfig:
    """Energy-conserving descent (the requested EGD optimizer).

    The dynamics keeps ``F(q) - F0`` as the "energy denominator", so ``F0`` must
    stay **below the smallest loss the model can reach**; otherwise the update is
    skipped entirely (``loss - F0`` must be positive) and, worse, as
    ``loss -> F0+`` the denominator collapses and the step explodes.  The default
    is therefore ``F0: null``, meaning ``initial_loss - auto_F0_margin``.

    ``lr`` is **not** the reference's value.  The reference
    (``ECD_q1_scaled``) used ``lr=1.0`` on a 124M-parameter GPT; on the small
    models in this repo that diverges immediately.  A measured sweep over 150
    steps on Multi30k (see README section 5) gives a stable window of roughly
    ``0.05 - 0.15``, so the default is ``0.1``.  Note ``lr`` is internally
    rescaled by ``1/sqrt(eta)``.
    """

    lr: float = 0.1
    eta: float = 100.0
    F0: Optional[float] = None
    auto_F0_margin: float = 1.0
    nu: float = 0.0
    eps1: float = 1e-10
    eps2: float = 1e-40
    weight_decay: float = 0.0
    consEn: bool = True
    seed: Optional[int] = None


@dataclass
class AdamWConfig:
    """Baseline optimizer, for comparison runs."""

    lr: float = 3e-4
    betas: List[float] = field(default_factory=lambda: [0.9, 0.98])
    eps: float = 1e-8
    weight_decay: float = 0.01


@dataclass
class TrainConfig:
    """Training-loop knobs."""

    optimizer: str = "egd"
    batch_size: int = 32
    epochs: int = 10
    #: 0 means "derive from ``epochs``".
    max_steps: int = 0
    #: 0 disables gradient clipping (Recommended to leave at 0 for EGD, whose
    #: update is already normalised).
    grad_clip: float = 0.0
    label_smoothing: float = 0.0
    seed: int = 42
    device: str = "auto"
    log_every: int = 50
    eval_every: int = 200
    max_eval_batches: int = 0
    #: 0 disables periodic BLEU; the final model is always scored if a test
    #: split is available and ``--eval`` is passed.
    bleu_every: int = 0
    bleu_samples: int = 200
    save_dir: str = "runs"
    run_name: str = ""
    num_workers: int = 0
    #: Hard wall-clock stop for the whole run, in seconds (0 disables).
    max_seconds: float = 0.0
    #: Warn when a single optimizer step takes longer than this (stall detector).
    stall_warn_seconds: float = 300.0
    egd: EGDConfig = field(default_factory=EGDConfig)
    adamw: AdamWConfig = field(default_factory=AdamWConfig)


@dataclass
class ExperimentConfig:
    """Root configuration: one YAML file describes one complete experiment."""

    model: ModelConfig = field(default_factory=ModelConfig)
    bias: BiasConfig = field(default_factory=BiasConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def validate(self) -> "ExperimentConfig":
        """Check every field and raise ``ValueError`` on the first problem."""
        m, b, d, t = self.model, self.bias, self.data, self.train

        # ---- model ----
        if m.d_model % m.n_heads != 0:
            raise ValueError(
                f"model.d_model ({m.d_model}) must be divisible by "
                f"model.n_heads ({m.n_heads})"
            )
        if m.n_encoder_layers < 1 or m.n_decoder_layers < 1:
            raise ValueError("encoder and decoder need at least one layer")
        if m.activation not in ACTIVATIONS:
            raise ValueError(
                f"model.activation must be one of {ACTIVATIONS}, got {m.activation!r}"
            )
        if m.max_seq_len < 2:
            raise ValueError("model.max_seq_len must be >= 2")
        if not 0.0 <= m.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if not 0.0 <= m.attention_dropout < 1.0:
            raise ValueError("model.attention_dropout must be in [0, 1)")
        if m.tie_output_embedding and m.share_embeddings is False:
            # Tying the output projection to the *target* embedding is always
            # possible; nothing to check unless vocabularies differ.
            pass
        if m.share_embeddings and m.vocab_size_src and m.vocab_size_tgt \
                and m.vocab_size_src != m.vocab_size_tgt:
            raise ValueError(
                "model.share_embeddings requires identical source/target vocab sizes"
            )

        # ---- bias ----
        for name, sub in (("bias.embed", b.embed), ("bias.attention", b.attention)):
            if sub.mode not in BIAS_MODES:
                raise ValueError(
                    f"{name}.mode must be one of {BIAS_MODES}, got {sub.mode!r}"
                )
            if sub.resample not in RESAMPLE_MODES:
                raise ValueError(
                    f"{name}.resample must be one of {RESAMPLE_MODES}, "
                    f"got {sub.resample!r}"
                )
            if sub.learnable and sub.resample != "fixed":
                raise ValueError(
                    f"{name}: learnable=True requires resample='fixed'"
                )
            if sub.learnable and sub.mode == "zero":
                raise ValueError(
                    f"{name}: learnable=True with mode='zero' has nothing to learn"
                )
        for sector in ("q", "k", "v"):
            for stat in ("mean", "std"):
                val = getattr(b.attention, f"{sector}_{stat}")
                if isinstance(val, (list, tuple)) and not all(
                    isinstance(x, (int, float)) for x in val
                ):
                    raise ValueError(
                        f"bias.attention.{sector}_{stat} must be a number or a "
                        f"list of numbers"
                    )
        if b.attention.enabled is False and b.attention.mode != "zero":
            # Allowed: ``enabled`` is the master switch, mode then describes what
            # the biases *would* be.  Keep it explicit rather than silently odd.
            pass

        # ---- data ----
        if d.source not in DATA_SOURCES:
            raise ValueError(
                f"data.source must be one of {DATA_SOURCES}, got {d.source!r}"
            )
        if d.tokenizer not in TOKENIZER_MODES:
            raise ValueError(
                f"data.tokenizer must be one of {TOKENIZER_MODES}, got {d.tokenizer!r}"
            )
        if d.synthetic_task not in SYNTHETIC_TASKS:
            # Checked regardless of ``source``: an invalid value is a typo either
            # way, and failing fast beats discovering it after switching source.
            raise ValueError(
                f"data.synthetic_task must be one of {SYNTHETIC_TASKS}, "
                f"got {d.synthetic_task!r}"
            )
        if d.min_freq < 1:
            raise ValueError("data.min_freq must be >= 1")

        # ---- train ----
        if t.optimizer not in OPTIMIZERS:
            raise ValueError(
                f"train.optimizer must be one of {OPTIMIZERS}, got {t.optimizer!r}"
            )
        if t.batch_size < 1:
            raise ValueError("train.batch_size must be >= 1")
        if t.epochs < 1:
            raise ValueError("train.epochs must be >= 1")
        if t.egd.eta <= 0:
            raise ValueError("train.egd.eta must be > 0")
        if t.egd.lr <= 0:
            raise ValueError("train.egd.lr must be > 0")
        if t.egd.nu < 0:
            raise ValueError("train.egd.nu must be >= 0")
        if t.adamw.lr <= 0:
            raise ValueError("train.adamw.lr must be > 0")
        if len(t.adamw.betas) != 2:
            raise ValueError("train.adamw.betas must have exactly two entries")
        return self

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #
    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    def describe(self) -> str:
        """Human-readable one-screen summary used at the start of a run."""
        m, b, d, t = self.model, self.bias, self.data, self.train
        attn = b.attention
        if not attn.enabled or attn.mode == "zero":
            attn_desc = "off (bQ=bK=bV=0)"
        else:
            sectors = [
                s
                for s, on in (("Q", attn.q_enabled), ("K", attn.k_enabled), ("V", attn.v_enabled))
                if on
            ]
            attn_desc = (
                f"mode={attn.mode} sectors=b{''.join(sectors)} "
                f"resample={attn.resample} shared_heads={attn.share_across_heads} "
                f"shared_layers={attn.share_across_layers}"
            )
        return "\n".join(
            [
                "=== Experiment configuration ===",
                f"  model          : d_model={m.d_model} enc={m.n_encoder_layers} "
                f"dec={m.n_decoder_layers} heads={m.n_heads} d_ff={m.d_ff} "
                f"dropout={m.dropout} act={m.activation} norm_first={m.norm_first}",
                f"  embedding bias : mode={b.embed.mode} resample={b.embed.resample} "
                f"mean={b.embed.mean} std={b.embed.std} "
                f"const={b.embed.const_value} learnable={b.embed.learnable}",
                f"  attention bias : {attn_desc}",
                f"  data           : source={d.source} tokenizer={d.tokenizer} "
                f"repo={d.hf_repo} max_vocab={d.max_vocab} min_freq={d.min_freq}",
                f"  optimizer      : {t.optimizer}",
                f"    egd          : lr={t.egd.lr} eta={t.egd.eta} F0={t.egd.F0} "
                f"nu={t.egd.nu} consEn={t.egd.consEn} wd={t.egd.weight_decay}",
                f"    adamw        : lr={t.adamw.lr} betas={t.adamw.betas} "
                f"wd={t.adamw.weight_decay}",
                f"  train          : batch={t.batch_size} epochs={t.epochs} "
                f"max_steps={t.max_steps} seed={t.seed} device={t.device}",
                "================================",
            ]
        )


# Alias: ``Config`` is the root object.
Config = ExperimentConfig


# --------------------------------------------------------------------------- #
# Named presets
# --------------------------------------------------------------------------- #
# The reference project defines its presets as a plain ``name -> dataclass``
# dict at the bottom of its config module.  Same idea here, but split into three
# orthogonal tables (model shape / data source / bias setting) plus a factory,
# so the three can be varied independently instead of needing a preset for every
# combination:
#
#     cfg = make_config(model="small", bias="b-gaussian", data="multi30k")
#
# Every table is read-only in spirit: ``make_config`` deep-copies whatever it
# takes, so mutating the returned config never corrupts a preset.

#: Encoder-decoder shapes. Names are descriptive; run ``--list-models`` to print
#: the exact parameter count of each one (it depends on the vocabulary size).
MODEL_PRESETS: Dict[str, ModelConfig] = {
    "smoke": ModelConfig(
        d_model=64, n_heads=4, n_encoder_layers=2, n_decoder_layers=2,
        d_ff=128, max_seq_len=16, dropout=0.1,
    ),
    "tiny": ModelConfig(
        d_model=128, n_heads=4, n_encoder_layers=2, n_decoder_layers=2,
        d_ff=256, max_seq_len=32, dropout=0.1,
    ),
    "small": ModelConfig(
        d_model=256, n_heads=8, n_encoder_layers=3, n_decoder_layers=3,
        d_ff=1024, max_seq_len=64, dropout=0.1,
    ),
    "base": ModelConfig(
        d_model=384, n_heads=8, n_encoder_layers=4, n_decoder_layers=4,
        d_ff=1536, max_seq_len=128, dropout=0.1,
    ),
    "large": ModelConfig(
        d_model=512, n_heads=8, n_encoder_layers=6, n_decoder_layers=6,
        d_ff=2048, max_seq_len=128, dropout=0.1,
    ),
    "xl": ModelConfig(
        d_model=768, n_heads=12, n_encoder_layers=12, n_decoder_layers=12,
        d_ff=3072, max_seq_len=256, dropout=0.1,
    ),
}

#: Where the parallel text comes from.
DATA_PRESETS: Dict[str, DataConfig] = {
    # Full Multi30k de->en through the HuggingFace mirror.
    "multi30k": DataConfig(),
    # The same, capped: the cheap preset the README starts from.
    "multi30k-tiny": DataConfig(
        max_train_samples=8000, max_val_samples=1000, max_vocab=12000
    ),
    # Offline toy tasks -- no network at all.
    "synthetic-copy": DataConfig(source="synthetic", synthetic_task="copy"),
    "synthetic-reverse": DataConfig(source="synthetic", synthetic_task="reverse"),
    "synthetic-sort": DataConfig(source="synthetic", synthetic_task="sort"),
}

#: The symmetry-breaking settings -- the actual point of the experiments.
BIAS_PRESETS: Dict[str, BiasConfig] = {
    # Control: b = 0 and no attention bias, so O(d_model) is preserved exactly.
    "symmetric": BiasConfig(),
    # Non-zero fixed b: breaks the embedding rotation symmetry once, at init.
    "b-gaussian": BiasConfig(
        embed=EmbeddingBiasConfig(mode="gaussian", mean=0.0, std=0.02, resample="fixed")
    ),
    # The same breaking, redrawn on every optimizer step.
    "b-gaussian-per-step": BiasConfig(
        embed=EmbeddingBiasConfig(mode="gaussian", mean=0.0, std=0.02, resample="per_step")
    ),
    # Isotropic (rank-one) direction instead of a random one.
    "b-const": BiasConfig(
        embed=EmbeddingBiasConfig(mode="const", const_value=1.0, resample="fixed")
    ),
    # b as a trained nn.Parameter rather than a fixed buffer.
    "b-learnable": BiasConfig(
        embed=EmbeddingBiasConfig(mode="gaussian", std=0.02, learnable=True, resample="fixed")
    ),
    # Reference-style per-head biases. bK stays off: a constant shift partly
    # cancels in the softmax normalisation.
    "attn-bQ": BiasConfig(
        attention=AttentionBiasConfig(
            enabled=True, mode="gaussian", q_enabled=True, k_enabled=False, v_enabled=False
        )
    ),
    "attn-bQbV": BiasConfig(
        attention=AttentionBiasConfig(
            enabled=True, mode="gaussian", q_enabled=True, k_enabled=False, v_enabled=True
        )
    ),
    "attn-learnable": BiasConfig(
        attention=AttentionBiasConfig(
            enabled=True, mode="gaussian", q_enabled=True, k_enabled=False,
            v_enabled=True, learnable=True, resample="fixed",
        )
    ),
}


def get_model_preset(name: str) -> ModelConfig:
    """Return a fresh copy of a model preset (mutating it is safe)."""
    if name not in MODEL_PRESETS:
        raise KeyError(
            f"unknown model preset {name!r}. Available: {sorted(MODEL_PRESETS)}"
        )
    return copy.deepcopy(MODEL_PRESETS[name])


def get_data_preset(name: str) -> DataConfig:
    """Return a fresh copy of a data preset."""
    if name not in DATA_PRESETS:
        raise KeyError(
            f"unknown data preset {name!r}. Available: {sorted(DATA_PRESETS)}"
        )
    return copy.deepcopy(DATA_PRESETS[name])


def get_bias_preset(name: str) -> BiasConfig:
    """Return a fresh copy of a bias preset."""
    if name not in BIAS_PRESETS:
        raise KeyError(
            f"unknown bias preset {name!r}. Available: {sorted(BIAS_PRESETS)}"
        )
    return copy.deepcopy(BIAS_PRESETS[name])


def make_config(
    model: Union[str, ModelConfig] = "small",
    bias: Union[str, BiasConfig] = "symmetric",
    data: Union[str, DataConfig] = "multi30k",
    optimizer: str = "egd",
    overrides: Optional[Sequence[str]] = None,
) -> ExperimentConfig:
    """Build a validated config from named presets -- the reference-style way in.

    Args:
        model: a key of :data:`MODEL_PRESETS`, or a ``ModelConfig`` to use as-is.
        bias: a key of :data:`BIAS_PRESETS`, or a ``BiasConfig``.
        data: a key of :data:`DATA_PRESETS`, or a ``DataConfig``.
        optimizer: ``"egd"`` or ``"adamw"``.
        overrides: optional ``dotted.path=value`` strings applied last, exactly
            as on the command line (``--set``).

    Returns:
        A validated :class:`ExperimentConfig`.

    Example:
        >>> cfg = make_config("tiny", "b-gaussian", "synthetic-copy")
        >>> cfg.bias.embed.mode
        'gaussian'
    """
    if isinstance(model, str):
        model = get_model_preset(model)
    else:
        model = copy.deepcopy(model)
    if isinstance(bias, str):
        bias = get_bias_preset(bias)
    else:
        bias = copy.deepcopy(bias)
    if isinstance(data, str):
        data = get_data_preset(data)
    else:
        data = copy.deepcopy(data)

    cfg = ExperimentConfig(
        model=model,
        bias=bias,
        data=data,
        train=TrainConfig(optimizer=optimizer),
    )
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg.validate()


def preset_table() -> str:
    """Readable listing of every preset name, for ``--list-models`` etc."""
    lines = ["model presets (shape; exact parameter count needs a vocabulary):"]
    for name, m in MODEL_PRESETS.items():
        lines.append(
            f"  {name:<9} d_model={m.d_model:<4} heads={m.n_heads:<3} "
            f"enc/dec={m.n_encoder_layers}/{m.n_decoder_layers} "
            f"d_ff={m.d_ff:<5} max_seq_len={m.max_seq_len}"
        )
    lines.append("")
    lines.append("data presets:")
    for name, d in DATA_PRESETS.items():
        detail = (
            f"hf repo={d.hf_repo} max_train_samples={d.max_train_samples}"
            if d.source == "hf"
            else f"synthetic task={d.synthetic_task}"
        )
        lines.append(f"  {name:<18} source={d.source:<9} {detail}")
    lines.append("")
    lines.append("bias presets (the symmetry-breaking switch):")
    for name, b in BIAS_PRESETS.items():
        attn = b.attention
        if not attn.enabled or attn.mode == "zero":
            attn_desc = "attn off"
        else:
            sectors = "".join(
                s
                for s, on in (("Q", attn.q_enabled), ("K", attn.k_enabled), ("V", attn.v_enabled))
                if on
            )
            attn_desc = f"attn b{sectors} {attn.mode}" + (" learnable" if attn.learnable else "")
        lines.append(
            f"  {name:<18} embed b: {b.embed.mode:<8} resample={b.embed.resample:<8} "
            f"learnable={str(b.embed.learnable):<5} | {attn_desc}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# YAML loading
# --------------------------------------------------------------------------- #
#: Explicit nesting map (avoids resolving postponed annotations at runtime).
_NESTED: Dict[type, Dict[str, type]] = {
    ExperimentConfig: {
        "model": ModelConfig,
        "bias": BiasConfig,
        "data": DataConfig,
        "train": TrainConfig,
    },
    BiasConfig: {
        "embed": EmbeddingBiasConfig,
        "attention": AttentionBiasConfig,
    },
    TrainConfig: {
        "egd": EGDConfig,
        "adamw": AdamWConfig,
    },
}


def _build(cls: type, raw: Any, path: str) -> Any:
    """Recursively build a dataclass from a plain dict, rejecting unknown keys."""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise TypeError(f"config section '{path}' must be a mapping, got {type(raw).__name__}")

    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(raw) - set(fields))
    if unknown:
        raise ValueError(
            f"unknown key(s) in config section '{path}': {unknown}. "
            f"Valid keys: {sorted(fields)}"
        )

    kwargs: Dict[str, Any] = {}
    for name, f in fields.items():
        if name not in raw:
            continue
        value = raw[name]
        sub = _NESTED.get(cls, {}).get(name)
        child_path = f"{path}.{name}" if path else name
        kwargs[name] = _build(sub, value, child_path) if sub is not None else value
    return cls(**kwargs)


def load_config(path: Union[str, Path], overrides: Optional[Sequence[str]] = None) -> ExperimentConfig:
    """Load a YAML config from ``path``, apply ``key.sub=value`` overrides, validate.

    Override values are parsed with ``yaml.safe_load`` so ``F0=null``, ``[0.9,0.98]``
    and ``true`` all behave as expected.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _build(ExperimentConfig, raw, "")
    if overrides:
        apply_overrides(cfg, overrides)
    return cfg.validate()


def apply_overrides(cfg: ExperimentConfig, overrides: Sequence[str]) -> ExperimentConfig:
    """Apply ``dotted.path=value`` overrides in place."""
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override {item!r} must look like 'train.egd.lr=0.5'")
        key, _, raw_value = item.partition("=")
        value = yaml.safe_load(raw_value)
        target: Any = cfg
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not hasattr(target, part):
                raise ValueError(f"unknown config path {key!r} (no attribute {part!r})")
            target = getattr(target, part)
        leaf = parts[-1]
        if not hasattr(target, leaf):
            raise ValueError(f"unknown config path {key!r} (no attribute {leaf!r})")
        setattr(target, leaf, value)
    return cfg


def save_config(cfg: ExperimentConfig, path: Union[str, Path]) -> Path:
    """Dump the resolved config to YAML (used next to every checkpoint)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, sort_keys=False, allow_unicode=True)
    return path


def config_from_dict(raw: Dict[str, Any]) -> ExperimentConfig:
    """Build a validated config from a plain dict (used for checkpoints/tests)."""
    return _build(ExperimentConfig, raw, "").validate()
