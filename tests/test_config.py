"""Tests for the configuration layer: named presets, YAML loading, overrides.

The preset tables are the reference-style entry point, so they are checked the
same way the model is: every entry must produce a config that passes
``validate()``, and the factory must not let a caller corrupt a preset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from transformer_sym.config import (
    BIAS_PRESETS,
    DATA_PRESETS,
    MODEL_PRESETS,
    AttentionBiasConfig,
    BiasConfig,
    EmbeddingBiasConfig,
    ExperimentConfig,
    ModelConfig,
    apply_overrides,
    config_from_dict,
    get_bias_preset,
    get_data_preset,
    get_model_preset,
    load_config,
    make_config,
    preset_table,
    save_config,
)

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Preset tables
# --------------------------------------------------------------------------- #
def test_every_model_preset_is_valid():
    assert MODEL_PRESETS, "no model presets defined"
    for name, model in MODEL_PRESETS.items():
        assert isinstance(model, ModelConfig)
        assert model.d_model % model.n_heads == 0, f"{name}: d_model not divisible by n_heads"
        # A whole config built from the preset must validate.
        make_config(model=name, bias="symmetric", data="synthetic-copy")


def test_every_data_preset_is_valid():
    assert DATA_PRESETS
    for name in DATA_PRESETS:
        make_config(model="smoke", bias="symmetric", data=name)


def test_every_bias_preset_is_valid():
    assert BIAS_PRESETS
    for name in BIAS_PRESETS:
        make_config(model="smoke", bias=name, data="synthetic-copy")


def test_bias_presets_cover_the_three_requested_modes():
    """zero / gaussian / const must all be reachable by name."""
    modes = {name: cfg.embed.mode for name, cfg in BIAS_PRESETS.items()}
    assert modes["symmetric"] == "zero"
    assert modes["b-gaussian"] == "gaussian"
    assert modes["b-const"] == "const"
    # And the reference-style per-head variants exist too.
    assert BIAS_PRESETS["attn-bQ"].attention.q_enabled is True
    assert BIAS_PRESETS["attn-bQbV"].attention.v_enabled is True
    assert BIAS_PRESETS["attn-bQ"].attention.k_enabled is False, "bK must default off"


def test_preset_table_mentions_every_name():
    table = preset_table()
    for name in list(MODEL_PRESETS) + list(DATA_PRESETS) + list(BIAS_PRESETS):
        assert name in table, f"{name} missing from preset_table()"


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def test_make_config_composes_the_three_tables():
    cfg = make_config(model="tiny", bias="b-gaussian", data="synthetic-copy")
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.model.d_model == MODEL_PRESETS["tiny"].d_model
    assert cfg.bias.embed.mode == "gaussian"
    assert cfg.data.source == "synthetic"
    assert cfg.data.synthetic_task == "copy"
    assert cfg.train.optimizer == "egd"


def test_make_config_honours_optimizer_and_overrides():
    cfg = make_config(
        model="smoke",
        bias="symmetric",
        data="synthetic-copy",
        optimizer="adamw",
        overrides=["train.egd.lr=0.25", "model.d_model=64", "bias.embed.mode=const"],
    )
    assert cfg.train.optimizer == "adamw"
    assert cfg.train.egd.lr == 0.25
    assert cfg.model.d_model == 64
    assert cfg.bias.embed.mode == "const"


def test_make_config_accepts_dataclass_instances():
    cfg = make_config(
        model=ModelConfig(d_model=32, n_heads=2, n_encoder_layers=1, n_decoder_layers=1),
        bias=BiasConfig(embed=EmbeddingBiasConfig(mode="const")),
        data="synthetic-copy",
    )
    assert cfg.model.d_model == 32
    assert cfg.bias.embed.mode == "const"


def test_make_config_does_not_let_callers_mutate_presets():
    """A returned config must be independent of the module-level preset."""
    first = make_config(model="smoke", bias="b-gaussian", data="synthetic-copy")
    first.model.d_model = 4096
    first.bias.embed.std = 99.0
    first.data.synthetic_task = "sort"

    assert MODEL_PRESETS["smoke"].d_model != 4096
    assert BIAS_PRESETS["b-gaussian"].embed.std != 99.0
    assert DATA_PRESETS["synthetic-copy"].synthetic_task == "copy"

    second = make_config(model="smoke", bias="b-gaussian", data="synthetic-copy")
    assert second.model.d_model == MODEL_PRESETS["smoke"].d_model
    assert second.bias.embed.std == BIAS_PRESETS["b-gaussian"].embed.std


def test_make_config_rejects_an_invalid_combination():
    """learnable b with mode='zero' has nothing to learn -> validation error."""
    with pytest.raises(ValueError, match="nothing to learn"):
        make_config(
            model="smoke",
            bias=BiasConfig(embed=EmbeddingBiasConfig(mode="zero", learnable=True)),
            data="synthetic-copy",
        )


def test_get_presets_raise_a_helpful_error():
    for getter, table in (
        (get_model_preset, MODEL_PRESETS),
        (get_bias_preset, BIAS_PRESETS),
        (get_data_preset, DATA_PRESETS),
    ):
        with pytest.raises(KeyError) as excinfo:
            getter("definitely-not-a-preset")
        message = str(excinfo.value)
        assert "definitely-not-a-preset" in message
        # The message lists what is actually available.
        assert any(name in message for name in table)


def test_get_preset_returns_a_copy():
    model = get_model_preset("small")
    model.d_model = 1
    assert MODEL_PRESETS["small"].d_model != 1


# --------------------------------------------------------------------------- #
# YAML path (kept alongside the presets)
# --------------------------------------------------------------------------- #
def test_every_yaml_config_in_the_repo_loads():
    configs = sorted((ROOT / "configs").glob("*.yaml"))
    assert configs, "no YAML configs found"
    for path in configs:
        cfg = load_config(path)
        assert isinstance(cfg, ExperimentConfig)
        assert cfg.train.optimizer in ("egd", "adamw")


def test_yaml_round_trip_and_overrides(tmp_path: Path):
    cfg = make_config(model="smoke", bias="b-const", data="synthetic-copy")
    path = save_config(cfg, tmp_path / "roundtrip.yaml")
    again = load_config(path)
    assert again.to_dict() == cfg.to_dict()

    overridden = load_config(path, overrides=["train.egd.lr=0.5", "model.d_ff=256"])
    assert overridden.train.egd.lr == 0.5
    assert overridden.model.d_ff == 256


def test_unknown_yaml_key_is_rejected():
    with pytest.raises(ValueError, match="unknown key"):
        config_from_dict({"model": {"d_modell": 128}})


def test_unknown_override_path_is_rejected():
    cfg = make_config(model="smoke", bias="symmetric", data="synthetic-copy")
    with pytest.raises(ValueError, match="unknown config path"):
        apply_overrides(cfg, ["train.not_a_field=1"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["not_a_section.field=1"])
    with pytest.raises(ValueError, match="must look like"):
        apply_overrides(cfg, ["train.egd.lr"])


def test_validation_catches_bad_shapes():
    with pytest.raises(ValueError, match="divisible"):
        config_from_dict({"model": {"d_model": 10, "n_heads": 3}})
    with pytest.raises(ValueError, match="activation"):
        config_from_dict({"model": {"activation": "swish"}})
    with pytest.raises(ValueError, match="mode must be one of"):
        config_from_dict({"bias": {"embed": {"mode": "uniform"}}})
    with pytest.raises(ValueError):
        config_from_dict({"data": {"synthetic_task": "shuffle"}})
    with pytest.raises(ValueError):
        config_from_dict({"train": {"optimizer": "lbfgs"}})


def test_attention_bias_config_defaults_match_the_reference():
    attn = AttentionBiasConfig()
    assert attn.enabled is False
    assert attn.q_enabled is True
    assert attn.k_enabled is False, "bK cancels in the softmax, so it is off by default"
    assert attn.v_enabled is True
    assert attn.share_across_heads is True


def test_egd_defaults_are_the_tuned_ones():
    """Regression guard on the two settings that were wrong out of the box."""
    from transformer_sym.config import EGDConfig

    egd = EGDConfig()
    assert egd.F0 is None, "a fixed F0 above the reachable loss silently freezes training"
    assert egd.lr == pytest.approx(0.1), "the reference's lr=1.0 diverges at this scale"
    assert egd.eta == pytest.approx(100.0)
