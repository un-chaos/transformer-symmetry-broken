"""
Configuration objects and model presets.

Follows the layout of the companion project (``ecd_symbreak/config.py``): one
plain dataclass describing the network shape, plus a ``PRESETS`` dict at the
bottom selected from the command line with ``--model``.  Everything that is not
part of the architecture (batch size, optimizer, data paths, ...) stays a
command-line flag, and ``examples/*.sh`` archive a complete flag set per
experiment.

One addition: ``BiasPresets`` names the symmetry-breaking settings, because the
bias ``b`` is the knob this repository exists to study.  A preset can be picked
with ``--bias_preset`` and any individual field overridden by a flag.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

#: Allowed values for the embedding bias ``b``.
BIAS_MODES = ("zero", "gaussian", "const")
#: ``fixed`` draws once at init, ``per_step`` redraws on every optimizer step.
RESAMPLE_MODES = ("fixed", "per_step")
ACTIVATIONS = ("gelu", "relu", "prelu")
#: Where the text comes from.
DATA_SOURCES = ("hf", "synthetic", "local", "fineweb")
SYNTHETIC_TASKS = ("copy", "reverse", "sort")
TOKENIZER_MODES = ("word", "char", "bpe")
#: How a corpus becomes encoder/decoder training pairs.
#:
#: ``translation`` needs a *parallel* corpus (Multi30k): source and target are two
#: languages.  ``denoising`` needs only raw *monolingual* text (FineWeb-Edu): the
#: encoder reads a span-corrupted copy and the decoder reconstructs the removed
#: spans, which is how an encoder-decoder is trained on a 10B-token web corpus.
OBJECTIVES = ("translation", "denoising")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
@dataclass
class DataConfig:
    """
    Where the parallel text comes from and how it is tokenized.

    Unlike the model shape this is not a preset table -- the training script
    fills it in from the data flags on the command line.
    """

    source: str = "hf"
    data_dir: str = "data"
    #: ``translation`` (needs parallel text) or ``denoising`` (needs only raw text).
    objective: str = "translation"

    # --- source == "hf" ---
    hf_repo: str = "bentrevett/multi30k"
    #: huggingface.co is unreachable from some networks; the mirror is default.
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
    #: The mirror answers 403 without a User-Agent header.
    user_agent: str = "Mozilla/5.0 (compatible; symbreak-transformer/1.0)"
    download_timeout: float = 120.0
    #: 0 means "no limit".
    max_train_samples: int = 0
    max_val_samples: int = 0
    #: Fall back to the synthetic task when a download fails.
    fallback_to_synthetic: bool = True

    # --- source == "local": <data_dir>/local/<split>.tsv ---
    local_src_col: int = 0
    local_tgt_col: int = 1

    # --- source == "synthetic" ---
    synthetic_task: str = "reverse"
    synthetic_train_size: int = 4000
    synthetic_val_size: int = 500
    synthetic_test_size: int = 500
    synthetic_vocab: int = 20
    synthetic_len: int = 8
    synthetic_seed: int = 1234

    # --- source == "fineweb": a large monolingual corpus of parquet shards ---
    #: Directory holding the shards.  ``python main.py download-data`` fills it.
    fineweb_dir: str = "data/fineweb-edu/sample-10BT"
    #: The column holding the document text.
    text_column: str = "text"
    #: 0 means "read everything"; a cap makes a quick trial run possible.
    max_documents: int = 0
    #: A monolingual corpus has no val split, so every N-th document is held out.
    val_every: int = 1000
    #: Fraction of tokens the denoising objective removes (T5 uses 0.15).
    noise_density: float = 0.15
    #: Average length of a removed span (T5 uses 3).
    mean_span_length: float = 3.0
    #: Vocabulary size used when training a `bpe` tokenizer on the corpus.
    bpe_vocab_size: int = 32000
    #: Documents used to train the bpe tokenizer (0 = derive from ``max_documents``).
    tokenizer_train_documents: int = 20000

    # --- tokenizer ---
    tokenizer: str = "word"
    lowercase: bool = True
    min_freq: int = 2
    max_vocab: int = 30000
    #: 0 falls back to the model's context_length.
    max_src_len: int = 0
    max_tgt_len: int = 0

    def validate(self) -> "DataConfig":
        if self.source not in DATA_SOURCES:
            raise ValueError(
                f"data source must be one of {DATA_SOURCES}, got {self.source!r}"
            )
        if self.objective not in OBJECTIVES:
            raise ValueError(
                f"objective must be one of {OBJECTIVES}, got {self.objective!r}"
            )
        if self.objective == "denoising" and self.source == "hf":
            # A parallel corpus is not what denoising wants, and a monolingual
            # one cannot be used for translation: catch the mismatch up front.
            raise ValueError(
                "objective='denoising' needs raw text; use source='fineweb' (or "
                "'local'/'synthetic'). objective='translation' needs parallel text."
            )
        if not 0.0 < self.noise_density < 1.0:
            raise ValueError(
                f"noise_density must be in (0, 1), got {self.noise_density}"
            )
        if self.mean_span_length < 1.0:
            raise ValueError("mean_span_length must be >= 1")
        if self.bpe_vocab_size < 100:
            raise ValueError("bpe_vocab_size must be >= 100")
        if self.tokenizer not in TOKENIZER_MODES:
            raise ValueError(
                f"tokenizer must be one of {TOKENIZER_MODES}, got {self.tokenizer!r}"
            )
        if self.synthetic_task not in SYNTHETIC_TASKS:
            raise ValueError(
                f"synthetic_task must be one of {SYNTHETIC_TASKS}, "
                f"got {self.synthetic_task!r}"
            )
        if self.min_freq < 1:
            raise ValueError("min_freq must be >= 1")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "DataConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown data key(s): {unknown}")
        return cls(**raw)


# --------------------------------------------------------------------------- #
# Network shape
# --------------------------------------------------------------------------- #
@dataclass
class Seq2SeqConfig:
    """
    Shape of the encoder-decoder transformer.

    The vocabulary sizes are runtime values: leave them at ``0`` and the
    training script fills them in from the tokenizers it just built.
    """

    context_length: int = 128
    src_vocab_size: int = 0
    tgt_vocab_size: int = 0

    n_encoder_layer: int = 3
    n_decoder_layer: int = 3
    n_head: int = 8
    n_embd: int = 256

    d_ff: int = 1024
    dropout: float = 0.1
    attention_dropout: float = 0.0

    activation: str = "gelu"
    #: PReLU with per-feature learnable slopes; the reference project's default
    #: for symmetry-breaking runs.
    prelu_random_init: bool = False
    prelu_slope_mean: float = 0.2
    prelu_slope_std: float = 1.0

    norm_first: bool = False
    scale_embedding: bool = True
    #: Tie the output projection to the target embedding matrix.
    tie_output_embedding: bool = True
    #: Share one embedding matrix between source and target (needs equal vocabs).
    share_embeddings: bool = False
    init_std: float = 0.02
    #: nanoGPT-style ``(2 * n_layer) ** -0.5`` scaling of residual outputs.
    scaled_residual_init: bool = True

    def validate(self) -> "Seq2SeqConfig":
        """Raise ``ValueError`` on an inconsistent shape."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.n_encoder_layer < 1 or self.n_decoder_layer < 1:
            raise ValueError("the encoder and decoder each need at least one layer")
        if self.context_length < 2:
            raise ValueError("context_length must be >= 2")
        if self.activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {ACTIVATIONS}, got {self.activation!r}"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")
        if self.share_embeddings and self.src_vocab_size and self.tgt_vocab_size:
            if self.src_vocab_size != self.tgt_vocab_size:
                raise ValueError(
                    "share_embeddings requires identical source/target vocab sizes"
                )
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Seq2SeqConfig":
        """Rebuild from a checkpoint's ``config`` entry."""
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown config key(s) in checkpoint: {unknown}")
        return cls(**raw)


# --------------------------------------------------------------------------- #
# Symmetry-breaking biases
# --------------------------------------------------------------------------- #
@dataclass
class BiasConfig:
    """
    The two bias families that break the rotation symmetry of attention.

    ``embed_*`` is the bias ``b`` added to the embedding, ``x -> x + b``.  With
    ``b = 0`` the embedding keeps its ``O(n_embd)`` rotation symmetry; any
    non-zero ``b`` breaks it, because a rotation ``R`` cannot be pushed through a
    fixed offset (``R(x + b) = Rx + Rb != Rx + b``).

    ``use_q_bias`` / ``use_k_bias`` / ``use_v_bias`` add the companion project's
    per-head biases inside attention.  ``bQ`` enters through the softmax and is
    therefore exponentially amplified; ``bV`` only passes through a linear map
    (power-law effect).  ``bK`` is off by default: the key-independent part of a
    constant shift cancels in the softmax normalisation.
    """

    # --- the embedding bias b: "zero" | "gaussian" | "const" ---
    embed_mode: str = "zero"
    embed_mean: float = 0.0
    embed_std: float = 0.02
    embed_const_value: float = 1.0
    embed_resample: str = "fixed"
    #: Turn b into a trained nn.Parameter instead of a fixed buffer.
    embed_learnable: bool = False
    embed_seed: int = 1234

    # --- reference-style per-head attention biases ---
    use_q_bias: bool = False
    use_k_bias: bool = False
    use_v_bias: bool = False
    attn_mode: str = "gaussian"
    attn_resample: str = "fixed"
    attn_learnable: bool = False
    mean_Q: float = 0.5
    std_Q: float = 0.05
    mean_K: float = 0.3
    std_K: float = 0.1
    mean_V: float = 0.5
    std_V: float = 0.05
    const_value: float = 1.0
    #: All heads share one ``head_dim`` vector (reference behaviour).
    share_across_heads: bool = True
    #: One bias object is reused by every layer of a given attention site.
    share_across_layers: bool = True
    #: Which attention sites carry a bias.
    apply_encoder: bool = True
    apply_decoder_self: bool = True
    apply_decoder_cross: bool = True
    attn_seed: int = 1234

    # ------------------------------------------------------------------ #
    @property
    def attention_enabled(self) -> bool:
        """True when at least one per-head bias is switched on."""
        return self.use_q_bias or self.use_k_bias or self.use_v_bias

    @property
    def attention_breaks_symmetry(self) -> bool:
        """
        True when a per-head bias can actually change the forward pass.

        A sector that is switched on with ``attn_mode == "zero"`` is still
        allocated but pinned to zero, so it breaks nothing.  This mirrors what
        the model's ``AttentionBias.active`` reports.
        """
        return self.attention_enabled and (
            self.attn_mode != "zero" or self.attn_learnable
        )

    @property
    def symmetric(self) -> bool:
        """True when nothing breaks the symmetry at all."""
        return (
            self.embed_mode == "zero"
            and not self.embed_learnable
            and not self.attention_breaks_symmetry
        )

    def validate(self) -> "BiasConfig":
        if self.embed_mode not in BIAS_MODES:
            raise ValueError(
                f"embed_mode must be one of {BIAS_MODES}, got {self.embed_mode!r}"
            )
        if self.attn_mode not in BIAS_MODES:
            raise ValueError(
                f"attn_mode must be one of {BIAS_MODES}, got {self.attn_mode!r}"
            )
        for name, value in (
            ("embed_resample", self.embed_resample),
            ("attn_resample", self.attn_resample),
        ):
            if value not in RESAMPLE_MODES:
                raise ValueError(
                    f"{name} must be one of {RESAMPLE_MODES}, got {value!r}"
                )
        # A learnable bias is drawn once and then trained; redrawing it would
        # fight the optimizer.
        if self.embed_learnable and self.embed_resample != "fixed":
            raise ValueError("embed_learnable requires embed_resample='fixed'")
        if self.attn_learnable and self.attn_resample != "fixed":
            raise ValueError("attn_learnable requires attn_resample='fixed'")
        if self.embed_learnable and self.embed_mode == "zero":
            raise ValueError("embed_learnable with embed_mode='zero' has nothing to learn")
        if self.attn_learnable and self.attn_mode == "zero":
            raise ValueError("attn_learnable with attn_mode='zero' has nothing to learn")
        return self

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "BiasConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(raw) - known)
        if unknown:
            raise ValueError(f"unknown bias key(s) in checkpoint: {unknown}")
        return cls(**raw)

    def describe(self) -> str:
        """One-line summary for the training banner."""
        if self.symmetric:
            return "b = 0, attention biases off (symmetric)"
        parts = []
        if self.embed_mode != "zero" or self.embed_learnable:
            parts.append(
                f"embedding b: mode={self.embed_mode} std={self.embed_std} "
                f"const={self.embed_const_value} resample={self.embed_resample} "
                f"learnable={self.embed_learnable}"
            )
        if self.attention_enabled:
            sectors = "".join(
                name
                for name, on in (
                    ("Q", self.use_q_bias),
                    ("K", self.use_k_bias),
                    ("V", self.use_v_bias),
                )
                if on
            )
            parts.append(
                f"attention b{sectors}: mode={self.attn_mode} "
                f"resample={self.attn_resample} learnable={self.attn_learnable} "
                f"shared_heads={self.share_across_heads} "
                f"shared_layers={self.share_across_layers}"
            )
        return " | ".join(parts)


# --------------------------------------------------------------------------- #
# Presets
# --------------------------------------------------------------------------- #
#: Model sizes, in the spirit of the companion project's ``PRESETS`` dict.
#: ``python scripts/train.py --list_models`` prints the exact parameter count of
#: each one for a given vocabulary.
PRESETS: Dict[str, Seq2SeqConfig] = {
    "smoke": Seq2SeqConfig(
        context_length=16, n_encoder_layer=2, n_decoder_layer=2,
        n_head=4, n_embd=64, d_ff=128, dropout=0.1,
    ),
    "tiny": Seq2SeqConfig(
        context_length=32, n_encoder_layer=2, n_decoder_layer=2,
        n_head=4, n_embd=128, d_ff=256, dropout=0.1,
    ),
    "small": Seq2SeqConfig(
        context_length=64, n_encoder_layer=3, n_decoder_layer=3,
        n_head=8, n_embd=256, d_ff=1024, dropout=0.1,
    ),
    "base": Seq2SeqConfig(
        context_length=128, n_encoder_layer=4, n_decoder_layer=4,
        n_head=8, n_embd=384, d_ff=1536, dropout=0.1,
    ),
    "large": Seq2SeqConfig(
        context_length=128, n_encoder_layer=6, n_decoder_layer=6,
        n_head=8, n_embd=512, d_ff=2048, dropout=0.1,
    ),
}

#: Named symmetry-breaking settings.  Pick one with ``--bias_preset`` and
#: override any single field with the matching flag.
BiasPresets: Dict[str, BiasConfig] = {
    # Control run: nothing is broken, O(n_embd) is an exact symmetry.
    "symmetric": BiasConfig(),
    # A non-zero b drawn once at initialisation.
    "b-gaussian": BiasConfig(
        embed_mode="gaussian", embed_mean=0.0, embed_std=0.02, embed_resample="fixed"
    ),
    # The same breaking, redrawn every optimizer step.
    "b-gaussian-per-step": BiasConfig(
        embed_mode="gaussian", embed_mean=0.0, embed_std=0.02, embed_resample="per_step"
    ),
    # An isotropic rank-one direction instead of a random one.
    "b-const": BiasConfig(
        embed_mode="const", embed_const_value=1.0, embed_resample="fixed"
    ),
    # b as a trained parameter rather than a fixed buffer.
    "b-learnable": BiasConfig(
        embed_mode="gaussian", embed_std=0.02, embed_learnable=True, embed_resample="fixed"
    ),
    # Companion-project style per-head biases.
    "attn-bQ": BiasConfig(use_q_bias=True, use_v_bias=False),
    "attn-bQbV": BiasConfig(use_q_bias=True, use_v_bias=True),
    "attn-full": BiasConfig(use_q_bias=True, use_k_bias=True, use_v_bias=True),
    "attn-learnable": BiasConfig(
        use_q_bias=True, use_v_bias=True, attn_learnable=True, attn_resample="fixed"
    ),
}

#: Dataset shorthand: name -> ``DataConfig`` keyword arguments. The keys are the
#: real field names, so a preset can be splatted straight into ``DataConfig``.
DATASET_PRESETS: Dict[str, Dict[str, Any]] = {
    "multi30k": {"source": "hf", "hf_repo": "bentrevett/multi30k"},
    # The smallest slice that still shows real translation behaviour.  This is
    # what the foolproof ``run.py`` menu offers as "quick": a couple of minutes
    # on CPU instead of tens of minutes.
    "multi30k-quick": {
        "source": "hf",
        "hf_repo": "bentrevett/multi30k",
        "max_train_samples": 2000,
        "max_val_samples": 300,
        "max_vocab": 8000,
    },
    "multi30k-tiny": {
        "source": "hf",
        "hf_repo": "bentrevett/multi30k",
        "max_train_samples": 8000,
        "max_val_samples": 1000,
        "max_vocab": 12000,
    },
    "synthetic-copy": {"source": "synthetic", "synthetic_task": "copy"},
    "synthetic-reverse": {"source": "synthetic", "synthetic_task": "reverse"},
    "synthetic-sort": {"source": "synthetic", "synthetic_task": "sort"},
    # --- the large monolingual corpus (needs `main.py download-data` first) ---
    #: FineWeb-Edu sample/10BT: ~10B tokens, 14 parquet shards, ~28.5 GB.  Trained
    #: with the denoising objective, because the corpus has no translations.
    "fineweb-10b": {
        "source": "fineweb",
        "objective": "denoising",
        "tokenizer": "bpe",
        "fineweb_dir": "data/fineweb-edu/sample-10BT",
        "bpe_vocab_size": 32000,
        "noise_density": 0.15,
        "mean_span_length": 3.0,
    },
    #: The same corpus, capped: enough to see the pipeline work in minutes.
    "fineweb-quick": {
        "source": "fineweb",
        "objective": "denoising",
        "tokenizer": "bpe",
        "fineweb_dir": "data/fineweb-edu/sample-10BT",
        "max_documents": 20000,
        "bpe_vocab_size": 8000,
        "tokenizer_train_documents": 5000,
        "noise_density": 0.15,
        "mean_span_length": 3.0,
    },
}


def get_preset(name: str) -> Seq2SeqConfig:
    """Return a fresh copy of a model preset."""
    if name not in PRESETS:
        raise KeyError(f"unknown model preset {name!r}. Available: {sorted(PRESETS)}")
    return dataclasses.replace(PRESETS[name])


def get_bias_preset(name: str) -> BiasConfig:
    """Return a fresh copy of a bias preset."""
    if name not in BiasPresets:
        raise KeyError(
            f"unknown bias preset {name!r}. Available: {sorted(BiasPresets)}"
        )
    return dataclasses.replace(BiasPresets[name])


def resolve_config(
    model: str,
    context_length: Optional[int] = None,
    src_vocab_size: Optional[int] = None,
    tgt_vocab_size: Optional[int] = None,
    **overrides: Any,
) -> Seq2SeqConfig:
    """
    Build a validated model config from a preset name plus CLI overrides.

    Mirrors the companion project's ``cfg = PRESETS[args.model]`` followed by
    ``cfg.context_length = args.ctx``, but on a copy so the preset is untouched.
    """
    cfg = get_preset(model)
    if context_length is not None:
        cfg.context_length = int(context_length)
    if src_vocab_size is not None:
        cfg.src_vocab_size = int(src_vocab_size)
    if tgt_vocab_size is not None:
        cfg.tgt_vocab_size = int(tgt_vocab_size)
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise ValueError(f"unknown model config field {key!r}")
        setattr(cfg, key, value)
    return cfg.validate()


def resolve_bias_config(
    preset: Optional[str] = None,
    **overrides: Any,
) -> BiasConfig:
    """
    Build a validated :class:`BiasConfig`.

    Starts from ``BiasPresets[preset]`` when a preset name is given, otherwise
    from the defaults, then applies every non-``None`` override.
    """
    cfg = get_bias_preset(preset) if preset else BiasConfig()
    for key, value in overrides.items():
        if value is None:
            continue
        if not hasattr(cfg, key):
            raise ValueError(f"unknown bias config field {key!r}")
        setattr(cfg, key, value)
    return cfg.validate()


def preset_table() -> str:
    """Human-readable listing of every preset, for ``--list_models``."""
    lines = ["model presets (--model):"]
    for name, cfg in PRESETS.items():
        lines.append(
            f"  {name:<8} n_embd={cfg.n_embd:<4} n_head={cfg.n_head:<3} "
            f"enc/dec={cfg.n_encoder_layer}/{cfg.n_decoder_layer} "
            f"d_ff={cfg.d_ff:<5} context_length={cfg.context_length}"
        )
    lines.append("")
    lines.append("bias presets (--bias_preset):")
    for name, bias in BiasPresets.items():
        lines.append(f"  {name:<20} {bias.describe()}")
    lines.append("")
    lines.append("dataset presets (--dataset_preset):")
    for name, kwargs in DATASET_PRESETS.items():
        detail = ", ".join(f"{k}={v}" for k, v in kwargs.items() if k != "source")
        lines.append(f"  {name:<18} {kwargs['source']:<10} {detail}")
    return "\n".join(lines)


__all__ = [
    "BIAS_MODES",
    "RESAMPLE_MODES",
    "ACTIVATIONS",
    "DATA_SOURCES",
    "SYNTHETIC_TASKS",
    "TOKENIZER_MODES",
    "OBJECTIVES",
    "Seq2SeqConfig",
    "BiasConfig",
    "DataConfig",
    "PRESETS",
    "BiasPresets",
    "DATASET_PRESETS",
    "get_preset",
    "get_bias_preset",
    "resolve_config",
    "resolve_bias_config",
    "preset_table",
]
