"""Tests for the configuration layer: presets, validation, serialization.

The preset tables are the user-facing entry point, so every entry is checked the
same way the model is: it must produce a config that passes ``validate()``.
"""

from __future__ import annotations

import pytest
import torch

from symbreak_transformer.config import (
    BIAS_MODES,
    DATASET_PRESETS,
    PRESETS,
    BiasConfig,
    BiasPresets,
    DataConfig,
    Seq2SeqConfig,
    get_bias_preset,
    get_preset,
    preset_table,
    resolve_bias_config,
    resolve_config,
)


# --------------------------------------------------------------------------- #
# Preset tables
# --------------------------------------------------------------------------- #
def test_every_model_preset_is_valid():
    assert PRESETS, "no model presets defined"
    for name, cfg in PRESETS.items():
        assert isinstance(cfg, Seq2SeqConfig)
        assert cfg.n_embd % cfg.n_head == 0, f"{name}: n_embd not divisible by n_head"
        assert cfg.validate() is cfg


def test_every_bias_preset_is_valid():
    assert BiasPresets
    for name, bias in BiasPresets.items():
        assert isinstance(bias, BiasConfig)
        assert bias.validate() is bias
        assert bias.embed_mode in BIAS_MODES


def test_bias_presets_cover_the_three_requested_modes():
    """zero / gaussian / const must all be reachable by name, plus the controls."""
    modes = {name: cfg.embed_mode for name, cfg in BiasPresets.items()}
    assert modes["symmetric"] == "zero"
    assert modes["b-gaussian"] == "gaussian"
    assert modes["b-const"] == "const"
    assert BiasPresets["symmetric"].symmetric is True
    assert BiasPresets["b-gaussian"].symmetric is False
    # Reference-style sectors, with bK off by default because a constant shift
    # partly cancels in the softmax.
    assert BiasPresets["attn-bQ"].attention_enabled is True
    assert BiasPresets["attn-bQ"].use_k_bias is False
    assert BiasPresets["attn-bQbV"].use_v_bias is True
    assert BiasPresets["attn-full"].use_k_bias is True


def test_every_dataset_preset_names_a_real_source():
    assert DATASET_PRESETS
    known = set(vars(DataConfig()))
    for name, kwargs in DATASET_PRESETS.items():
        assert "source" in kwargs, f"{name} must name a data source"
        unknown = sorted(set(kwargs) - known)
        assert not unknown, f"{name} uses unknown DataConfig fields: {unknown}"
        # A preset must be splattable straight into DataConfig.
        DataConfig(**kwargs).validate()


def test_preset_table_mentions_every_name():
    table = preset_table()
    for name in list(PRESETS) + list(BiasPresets) + list(DATASET_PRESETS):
        assert name in table, f"{name} missing from preset_table()"


def test_get_preset_returns_an_independent_copy():
    cfg = get_preset("small")
    cfg.n_embd = 8
    assert PRESETS["small"].n_embd != 8

    bias = get_bias_preset("b-gaussian")
    bias.embed_std = 99.0
    assert BiasPresets["b-gaussian"].embed_std != 99.0


def test_get_preset_raises_a_helpful_error():
    for getter, table in ((get_preset, PRESETS), (get_bias_preset, BiasPresets)):
        with pytest.raises(KeyError) as excinfo:
            getter("definitely-not-a-preset")
        message = str(excinfo.value)
        assert "definitely-not-a-preset" in message
        assert any(name in message for name in table)


# --------------------------------------------------------------------------- #
# resolve_*
# --------------------------------------------------------------------------- #
def test_resolve_config_overrides_the_preset_without_mutating_it():
    before = PRESETS["smoke"].context_length
    cfg = resolve_config(
        "smoke",
        context_length=48,
        src_vocab_size=100,
        tgt_vocab_size=200,
        n_embd=64,
        dropout=0.0,
    )
    assert cfg.context_length == 48
    assert cfg.src_vocab_size == 100 and cfg.tgt_vocab_size == 200
    assert cfg.dropout == 0.0
    assert PRESETS["smoke"].context_length == before
    assert PRESETS["smoke"].src_vocab_size == 0


def test_resolve_config_ignores_none_overrides():
    cfg = resolve_config("smoke", n_embd=None, dropout=None)
    assert cfg.n_embd == PRESETS["smoke"].n_embd


def test_resolve_config_rejects_an_unknown_field():
    with pytest.raises(ValueError, match="unknown model config field"):
        resolve_config("smoke", not_a_field=1)


def test_resolve_bias_config_starts_from_a_preset_and_overrides():
    cfg = resolve_bias_config("b-gaussian", embed_std=0.07)
    assert cfg.embed_mode == "gaussian"
    assert cfg.embed_std == 0.07
    # No preset -> plain defaults.
    default = resolve_bias_config()
    assert default.symmetric is True
    with pytest.raises(ValueError, match="unknown bias config field"):
        resolve_bias_config(not_a_field=1)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validation_catches_bad_model_shapes():
    with pytest.raises(ValueError, match="divisible"):
        Seq2SeqConfig(n_embd=10, n_head=3).validate()
    with pytest.raises(ValueError, match="activation"):
        Seq2SeqConfig(activation="swish").validate()
    with pytest.raises(ValueError, match="at least one layer"):
        Seq2SeqConfig(n_encoder_layer=0).validate()
    with pytest.raises(ValueError, match="context_length"):
        Seq2SeqConfig(context_length=1).validate()
    with pytest.raises(ValueError, match="dropout"):
        Seq2SeqConfig(dropout=1.5).validate()
    with pytest.raises(ValueError, match="share_embeddings"):
        Seq2SeqConfig(
            share_embeddings=True, src_vocab_size=10, tgt_vocab_size=11
        ).validate()


def test_validation_catches_bad_bias_settings():
    with pytest.raises(ValueError, match="embed_mode"):
        BiasConfig(embed_mode="uniform").validate()
    with pytest.raises(ValueError, match="attn_mode"):
        BiasConfig(attn_mode="uniform").validate()
    with pytest.raises(ValueError, match="embed_resample"):
        BiasConfig(embed_resample="sometimes").validate()
    # A learnable bias is drawn once and then trained.
    with pytest.raises(ValueError, match="embed_learnable requires"):
        BiasConfig(
            embed_mode="gaussian", embed_learnable=True, embed_resample="per_step"
        ).validate()
    with pytest.raises(ValueError, match="nothing to learn"):
        BiasConfig(embed_mode="zero", embed_learnable=True).validate()
    with pytest.raises(ValueError, match="nothing to learn"):
        BiasConfig(attn_mode="zero", attn_learnable=True).validate()


def test_validation_catches_bad_data_settings():
    with pytest.raises(ValueError, match="source"):
        DataConfig(source="ftp").validate()
    with pytest.raises(ValueError, match="tokenizer"):
        DataConfig(tokenizer="bpe").validate()
    with pytest.raises(ValueError, match="synthetic_task"):
        DataConfig(synthetic_task="shuffle").validate()
    with pytest.raises(ValueError, match="min_freq"):
        DataConfig(min_freq=0).validate()


def test_bias_symmetric_property():
    assert BiasConfig().symmetric is True
    assert BiasConfig(embed_mode="gaussian").symmetric is False
    assert BiasConfig(embed_mode="const").symmetric is False
    assert BiasConfig(use_q_bias=True, attn_mode="gaussian").symmetric is False
    # A zero-mode attention bias breaks nothing.
    assert BiasConfig(use_q_bias=True, attn_mode="zero").symmetric is True
    # A learnable-but-zero bias is still a symmetry break (it can leave zero).
    assert BiasConfig(embed_mode="gaussian", embed_learnable=True).symmetric is False


# --------------------------------------------------------------------------- #
# Serialization (checkpoints save plain dicts, not pickled dataclasses)
# --------------------------------------------------------------------------- #
def test_config_round_trips_through_a_dict():
    cfg = resolve_config("tiny", context_length=24, src_vocab_size=50, tgt_vocab_size=60)
    again = Seq2SeqConfig.from_dict(cfg.to_dict())
    assert again == cfg


def test_bias_config_round_trips_through_a_dict():
    bias = get_bias_preset("attn-bQbV")
    again = BiasConfig.from_dict(bias.to_dict())
    assert again == bias


def test_data_config_round_trips_through_a_dict():
    cfg = DataConfig(source="synthetic", synthetic_task="copy", max_vocab=128)
    assert DataConfig.from_dict(cfg.to_dict()) == cfg


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown config key"):
        Seq2SeqConfig.from_dict({"n_embd": 64, "d_model": 64})
    with pytest.raises(ValueError, match="unknown bias key"):
        BiasConfig.from_dict({"embed_mode": "zero", "nonsense": 1})
    with pytest.raises(ValueError, match="unknown data key"):
        DataConfig.from_dict({"source": "hf", "nonsense": 1})


def test_describe_is_informative():
    assert "symmetric" in BiasConfig().describe()
    text = get_bias_preset("attn-bQbV").describe()
    assert "bQV" in text
    assert "gaussian" in text
    assert "embedding b" not in text  # embedding bias is off in that preset


def test_egd_defaults_are_the_tuned_ones():
    """Regression guard on the two optimizer settings that were wrong at first."""
    from symbreak_transformer.optimizer import EGD

    opt = EGD([torch.nn.Parameter(torch.zeros(2))])
    assert opt.lr == pytest.approx(0.1), (
        "the upstream reference's lr=1.0 diverges at this model scale"
    )
    assert opt.F0 is None, (
        "a fixed F0 above the reachable loss silently freezes training"
    )
    assert opt.eta == pytest.approx(100.0)
