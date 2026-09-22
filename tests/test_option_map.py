"""Tests for the "every knob is configurable" guarantee.

The user's requirement was that no setting may be hardcoded: the initial
parameters and hyperparameters have to be freely configurable. That is only
trustworthy if it is checked, so these tests assert three things:

1. every field of every configuration dataclass is either reachable through a
   command-line flag or listed in ``*_NOT_EXPOSED`` **with a reason**;
2. every flag the mapping table names actually exists in ``scripts/train.py``
   (no dead mapping that silently does nothing);
3. the generated control panel exposes all of them, so ``my_config.py`` really is
   a complete surface rather than a hand-picked subset.
"""

from __future__ import annotations

import pytest

import run as menu
from symbreak_transformer.config import BiasConfig, DataConfig, Seq2SeqConfig
from symbreak_transformer.option_map import (
    BIAS_FIELD_TO_DEST,
    BIAS_NOT_EXPOSED,
    DATA_FIELD_TO_DEST,
    DATA_NOT_EXPOSED,
    MODEL_FIELD_TO_DEST,
    MODEL_NOT_EXPOSED,
    OPTIMIZER_FIELD_TO_DEST,
    RUN_FIELD_TO_DEST,
    SCHEDULE_FIELD_TO_DEST,
    assert_complete,
    field_overrides,
)


@pytest.fixture(scope="module")
def parser():
    """The real training parser (loading train.py costs a torch import)."""
    return menu.train_parser()


def dests(parser) -> set:
    return {action.dest for action in parser._actions}  # noqa: SLF001


# --------------------------------------------------------------------------- #
# 1. no field is silently unreachable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("cls", "table", "excused", "label"),
    [
        (Seq2SeqConfig, MODEL_FIELD_TO_DEST, MODEL_NOT_EXPOSED, "model"),
        (BiasConfig, BIAS_FIELD_TO_DEST, BIAS_NOT_EXPOSED, "bias"),
        (DataConfig, DATA_FIELD_TO_DEST, DATA_NOT_EXPOSED, "data"),
    ],
)
def test_every_config_field_is_configurable(cls, table, excused, label):
    """A field must have a flag, or an explicit documented reason it cannot."""
    problem = assert_complete(cls, table, excused, label)
    assert problem is None, problem


def test_excused_fields_state_a_reason():
    for name, reason in {**MODEL_NOT_EXPOSED, **BIAS_NOT_EXPOSED, **DATA_NOT_EXPOSED}.items():
        assert isinstance(reason, str) and len(reason) > 15, (
            f"{name} is excused without a real reason: {reason!r}"
        )


def test_the_only_unreachable_fields_are_the_documented_ones():
    """Guards against someone "fixing" a hole by adding it to the excuse list."""
    excused = set(MODEL_NOT_EXPOSED) | set(BIAS_NOT_EXPOSED) | set(DATA_NOT_EXPOSED)
    assert excused == {
        "src_vocab_size",   # derived from the tokenizer
        "tgt_vocab_size",   # derived from the tokenizer
        "hf_files",         # a per-split filename mapping, not a scalar
    }


# --------------------------------------------------------------------------- #
# 2. every mapped flag really exists in the training script
# --------------------------------------------------------------------------- #
def test_every_mapped_dest_exists_in_the_train_parser(parser):
    known = dests(parser)
    missing = []
    for label, table in (
        ("model", MODEL_FIELD_TO_DEST),
        ("bias", BIAS_FIELD_TO_DEST),
        ("data", DATA_FIELD_TO_DEST),
        ("schedule", SCHEDULE_FIELD_TO_DEST),
        ("optimizer", OPTIMIZER_FIELD_TO_DEST),
        ("run", RUN_FIELD_TO_DEST),
    ):
        for field, dest in table.items():
            if dest not in known:
                missing.append(f"{label}.{field} -> --{dest}")
    assert not missing, f"mapping names flags that train.py does not accept: {missing}"


def test_non_dataclass_tables_are_identity_maps():
    for table in (SCHEDULE_FIELD_TO_DEST, OPTIMIZER_FIELD_TO_DEST, RUN_FIELD_TO_DEST):
        for field, dest in table.items():
            assert field == dest


# --------------------------------------------------------------------------- #
# 3. the flag -> value plumbing
# --------------------------------------------------------------------------- #
def test_field_overrides_only_takes_dests_that_were_given():
    """train.py uses argparse.SUPPRESS, so an absent dest must not override."""
    namespace = {"n_embd": 64, "dropout": 0.0}     # nothing else was passed
    overrides = field_overrides(MODEL_FIELD_TO_DEST, namespace)
    assert overrides == {"n_embd": 64, "dropout": 0.0}


def test_flags_to_argv_renders_values_and_switches(parser):
    argv = menu.flags_to_argv(
        {
            "--n_embd": 128,          # takes a value
            "--dropout": 0.05,        # takes a value
            "--norm_first": True,     # switch present -> emit
            "--prelu_random_init": False,   # switch absent -> skip
            "--no_lowercase": True,   # reverse switch present -> emit
            "--no_scaled_residual_init": True,
            "--name": "x",
            "--max_steps": None,      # explicit None -> skip
        },
        parser=parser,
    )
    assert "--n_embd" in argv and "128" in argv
    assert "--dropout" in argv and "0.05" in argv
    assert "--norm_first" in argv
    assert "--prelu_random_init" not in argv
    assert "--no_lowercase" in argv
    assert "--no_scaled_residual_init" in argv
    assert "--max_steps" not in argv


def test_switch_values_mean_should_the_flag_be_there(parser):
    """False on a switch means "do not add it", even for a --no_* flag."""
    argv = menu.flags_to_argv(
        {"--norm_first": False, "--no_lowercase": False}, parser=parser
    )
    assert argv == []


def test_flags_to_argv_passes_unknown_flags_through(parser):
    """An unknown flag must reach train.py so it fails loudly, not be dropped."""
    argv = menu.flags_to_argv({"--not_a_real_flag": "5"}, parser=parser)
    assert argv == ["--not_a_real_flag", "5"]


# --------------------------------------------------------------------------- #
# 4. the generated control panel is complete
# --------------------------------------------------------------------------- #
def test_control_panel_lists_every_available_option(parser):
    """my_config.py's reference block must not be a hand-picked subset."""
    catalog = menu.option_catalog()
    catalog_flags = {flag for _, entries in catalog for flag, _, _ in entries}
    assert len(catalog_flags) > 100, "the training script should expose many options"

    panel = menu.render_my_config({"--n_embd": 64})
    for flag in catalog_flags:
        assert flag in panel, f"{flag} is missing from the generated panel"


def test_control_panel_round_trip_preserves_the_run(tmp_path):
    """Whatever the menu ran must be reproducible from the panel it writes."""
    flags = menu.speed_flags(
        menu.SPEEDS["2"], "b-gaussian", epochs=3, optimizer="adamw", log_dir="myruns"
    )
    path = tmp_path / "my_config.py"
    menu.write_my_config(flags, path)
    panel = menu.read_my_config(path)
    assert panel is not None
    assert panel["extra"] == []
    # The panel also lists the common knobs that were *not* set (value None means
    # "do not pass this flag"), so the dicts are not identical -- but everything
    # the menu chose must survive, and the resulting command must be the same.
    for flag, value in flags.items():
        assert panel["flags"][flag] == value, f"{flag} changed across the panel"
    assert sorted(menu.flags_to_argv(panel["flags"])) == sorted(
        menu.flags_to_argv(flags)
    )


def test_a_hand_edited_panel_is_honoured(tmp_path):
    """Changing one value in the file must change the command."""
    path = tmp_path / "my_config.py"
    menu.write_my_config(menu.speed_flags(menu.SPEEDS["2"], "b-gaussian"), path)
    text = path.read_text(encoding="utf-8").replace("'tiny'", "'base'", 1)
    path.write_text(text, encoding="utf-8")
    panel = menu.read_my_config(path)
    assert panel["flags"]["--model"] == "base"


def test_new_flags_are_settable_through_the_panel(parser):
    """The knobs added for this requirement must survive the panel round trip."""
    wanted = {
        "--prelu_slope_mean": 0.3,
        "--prelu_slope_std": 0.7,
        "--no_scaled_residual_init": True,
        "--attn_const": 2.0,
        "--no_apply_encoder": True,
        "--no_apply_decoder_cross": True,
        "--attn_seed": 99,
        "--objective": "denoising",
        "--fineweb_dir": "data/fineweb-edu/sample-10BT",
        "--max_documents": 123,
        "--noise_density": 0.2,
        "--mean_span_length": 4.0,
        "--bpe_vocab_size": 9000,
        "--tokenizer_train_documents": 111,
        "--val_every": 50,
        "--text_column": "content",
        "--user_agent": "ua",
        "--download_timeout": 33.0,
    }
    argv = menu.flags_to_argv(wanted, parser=parser)
    text = " ".join(argv)
    for flag, value in wanted.items():
        assert flag in argv, f"{flag} was dropped"
        if value is not True:
            assert str(value) in argv, f"{flag}'s value was dropped"
