"""Tests for the "every knob is configurable" guarantee.

The user's requirement was that no setting may be hardcoded: the initial
parameters and hyperparameters have to be freely configurable. That is only
trustworthy if it is checked, so these tests assert three things:

1. every field of every configuration dataclass is either reachable through a
   command-line flag or listed in ``*_NOT_EXPOSED`` **with a reason**;
2. every flag the mapping table names actually exists in ``scripts/train.py``
   (no dead mapping that silently does nothing);
3. the menu's parameter editor exposes all of them, so ``run.py`` really is a
   complete surface rather than a hand-picked subset.
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


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    """Keep menu state out of the developer's real ``run_settings.json``."""
    monkeypatch.setattr(menu, "SETTINGS_FILE", tmp_path / "run_settings.json")
    saved_state, saved_custom = dict(menu.STATE), dict(menu.CUSTOM)
    menu.STATE.update({"speed": "1", "bias": "b-gaussian", "optimizer": "egd"})
    menu.CUSTOM.clear()
    yield menu
    menu.STATE.clear()
    menu.STATE.update(saved_state)
    menu.CUSTOM.clear()
    menu.CUSTOM.update(saved_custom)


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
# 4. the in-menu parameter editor is a complete surface
# --------------------------------------------------------------------------- #
def test_the_hidden_flags_are_only_informational_or_forbidden():
    """The editor may hide flags, but only for a stated reason -- and only these."""
    assert menu.HIDDEN_FLAGS == {"-h", "--help", "--no_plot", "--list_models"}


def test_every_flag_is_reachable_from_the_menu_editor(parser):
    """Nothing is hardcoded: every flag train.py accepts is editable in menu 3)."""
    known = {action.option_strings[0] for action in parser._actions  # noqa: SLF001
             if action.option_strings}
    editable = set(menu.all_flags()) | set(menu.HIDDEN_FLAGS)
    assert known - editable == set(), "flags unreachable from the menu: %s" % (
        sorted(known - editable)
    )
    assert len(menu.all_flags()) > 100, "the training script should expose many options"


def test_every_editable_flag_has_a_live_argparse_action(parser):
    """Each row the editor can show must map back to a real action."""
    index = menu._action_index(parser)
    for flag in menu.all_flags():
        assert flag in index, f"{flag} is listed but train.py does not accept it"


def test_the_common_screen_only_lists_real_flags(parser):
    index = menu._action_index(parser)
    missing = [flag for flag in menu.COMMON_FLAGS if flag not in index]
    assert not missing, f"menu 3) advertises flags train.py does not have: {missing}"


def test_the_common_screen_offers_the_settings_the_user_asked_about():
    """The knobs that matter for this project must be one tap away, not hidden."""
    for flag in ("--model", "--dataset_preset", "--bias_preset", "--optimizer",
                 "--egd_lr", "--batch_size", "--epochs", "--max_steps", "--seed"):
        assert flag in menu.COMMON_FLAGS, f"{flag} should be on the common screen"


def test_the_menu_never_disables_the_loss_curve(isolated_settings):
    """The curve is a hard requirement, so the foolproof path always plots."""
    argv = menu.build_train_command(menu.effective_flags())
    assert "--no_plot" not in argv


# --------------------------------------------------------------------------- #
# 5. custom settings: they win over the bundle, and they are remembered
# --------------------------------------------------------------------------- #
def test_custom_settings_override_the_bundle(isolated_settings):
    menu.STATE["speed"] = "2"
    menu.CUSTOM["--n_embd"] = 64
    flags = menu.effective_flags()
    assert flags["--n_embd"] == 64                          # the edit wins
    assert flags["--model"] == menu.SPEEDS["2"]["model"]    # bundle otherwise intact
    assert flags["--dataset_preset"] == menu.SPEEDS["2"]["dataset_preset"]


def test_custom_settings_survive_a_restart(isolated_settings):
    menu.CUSTOM["--n_embd"] = 96
    menu.CUSTOM["--no_tie_output_embedding"] = True
    menu.STATE["speed"] = "3"
    menu.save_settings()

    menu.CUSTOM.clear()
    menu.STATE["speed"] = "1"
    menu.load_settings()

    assert menu.CUSTOM == {"--n_embd": 96, "--no_tie_output_embedding": True}
    assert menu.STATE["speed"] == "3"
    assert menu.effective_flags()["--n_embd"] == 96


def test_a_broken_settings_file_is_ignored(isolated_settings):
    menu.SETTINGS_FILE.write_text("{not json at all", encoding="utf-8")
    menu.load_settings()                     # must not raise
    assert menu.CUSTOM == {}
    assert menu.STATE["speed"] == "1"


def test_a_settings_file_with_dead_choices_falls_back(isolated_settings):
    menu.SETTINGS_FILE.write_text(
        '{"speed": "99", "bias": "not-a-bias", "optimizer": "nope", "custom": {}}',
        encoding="utf-8",
    )
    menu.load_settings()
    assert menu.STATE == {"speed": "1", "bias": "b-gaussian", "optimizer": "egd"}


def test_editing_a_number_in_the_menu_changes_the_command(isolated_settings, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "64")
    menu.ask_value("--n_embd", menu._action_index()["--n_embd"])
    assert menu.CUSTOM["--n_embd"] == 64

    argv = menu.build_train_command(menu.effective_flags())
    assert argv[argv.index("--n_embd") + 1] == "64"


def test_editing_a_choice_in_the_menu_stores_the_chosen_value(isolated_settings, monkeypatch):
    """A flag with fixed choices is edited by number, so only valid values land."""
    monkeypatch.setattr("builtins.input", lambda *a, **k: "3")   # 3rd choice
    action = menu._action_index()["--model"]
    menu.ask_value("--model", action)
    assert menu.CUSTOM["--model"] == list(action.choices)[2]


def test_an_empty_answer_restores_the_default(isolated_settings, monkeypatch):
    menu.CUSTOM["--n_embd"] = 64
    monkeypatch.setattr("builtins.input", lambda *a, **k: "")
    menu.ask_value("--n_embd", menu._action_index()["--n_embd"])
    assert "--n_embd" not in menu.CUSTOM
    assert "--n_embd" not in menu.build_train_command(menu.effective_flags())


def test_a_bad_number_is_rejected_without_changing_anything(isolated_settings, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a, **k: "not-a-number")
    menu.ask_value("--n_embd", menu._action_index()["--n_embd"])
    assert "--n_embd" not in menu.CUSTOM


def test_new_flags_are_settable_through_the_menu(isolated_settings, parser):
    """The knobs added for this requirement must survive the flag plumbing."""
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
    menu.CUSTOM.update(wanted)
    argv = menu.build_train_command(menu.effective_flags())
    for flag, value in wanted.items():
        assert flag in argv, f"{flag} was dropped by the menu"
        if value is not True:
            assert str(value) in argv, f"{flag}'s value was dropped"
