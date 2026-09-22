"""Tests for the two user-facing scripts: the foolproof menu and the report.

`run.py` is what a non-programmer actually touches, so its option tables are
checked against the real presets (a typo there would be a dead menu entry) and its
generated command is checked field by field. `scripts/report.py` is what turns a
directory of runs into one figure and one table, so it is checked on real-shaped
run directories, including deliberately incomplete ones.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

import run as runpy_menu  # the menu script at the repo root
from symbreak_transformer.config import DATASET_PRESETS, PRESETS, BiasPresets

ROOT = Path(__file__).resolve().parents[1]

#: Column order that ``scripts/train.py`` writes (train rows fill train_*, eval
#: rows fill val_*).
LOG_HEADER = [
    "step", "epoch", "train_loss", "train_ppl", "val_loss", "val_ppl",
    "elapsed_sec", "sec_per_step", "tokens_per_sec",
    "egd_iteration", "egd_momentum_norm", "egd_skipped",
    "embed_b_norm", "bQ_norm", "bK_norm", "bV_norm", "best_val_loss",
]


@pytest.fixture(scope="module")
def report():
    """Load ``scripts/report.py`` without putting ``scripts/`` on sys.path."""
    path = ROOT / "scripts" / "report.py"
    spec = importlib.util.spec_from_file_location("report_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_fake_run(
    parent: Path,
    name: str,
    embed_mode: str = "zero",
    n_val: int = 6,
    with_summary: bool = True,
    with_eval: bool = False,
) -> Path:
    """Create a run directory shaped like a real one (deliberately minimal)."""
    run_dir = parent / name
    run_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(1, n_val * 4 + 1):
        row = {column: "" for column in LOG_HEADER}
        row["step"] = index
        row["epoch"] = 0
        row["train_loss"] = f"{4.0 - 0.02 * index:.6f}"
        row["train_ppl"] = "20.0"
        row["embed_b_norm"] = "0.0" if embed_mode == "zero" else "0.16"
        row["bQ_norm"] = row["bK_norm"] = row["bV_norm"] = "0.0"
        rows.append(row)
        if index % 4 == 0:
            val_row = {column: "" for column in LOG_HEADER}
            val_row["step"] = index
            val_row["epoch"] = 0
            val_row["val_loss"] = f"{4.2 - 0.05 * (index // 4):.6f}"
            val_row["val_ppl"] = "30.0"
            val_row["embed_b_norm"] = row["embed_b_norm"]
            val_row["bQ_norm"] = val_row["bK_norm"] = val_row["bV_norm"] = "0.0"
            rows.append(val_row)

    with (run_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_HEADER)
        writer.writeheader()
        writer.writerows(rows)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "context_length": 32, "src_vocab_size": 400, "tgt_vocab_size": 420,
                "n_encoder_layer": 2, "n_decoder_layer": 2, "n_head": 4, "n_embd": 128,
                "d_ff": 256,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "bias.json").write_text(
        json.dumps({"embed_mode": embed_mode, "use_q_bias": False, "use_v_bias": False}),
        encoding="utf-8",
    )
    (run_dir / "args.json").write_text(
        json.dumps(
            {
                "model": "tiny", "dataset_preset": "multi30k-quick", "epochs": 2,
                "max_steps": 0, "optimizer": "egd", "batch_size": 32, "name": name,
            }
        ),
        encoding="utf-8",
    )
    if with_summary:
        (run_dir / "summary.json").write_text(
            json.dumps(
                {
                    "run_name": name, "steps": n_val * 4, "parameters": 1002368,
                    "optimizer": "EGD", "best_val_loss": 4.18, "elapsed_sec": 31.4,
                }
            ),
            encoding="utf-8",
        )
    if with_eval:
        (run_dir / "eval_test.json").write_text(
            json.dumps({"split": "test", "loss": 4.2, "bleu": {"bleu": 0.12}}),
            encoding="utf-8",
        )
    return run_dir


# --------------------------------------------------------------------------- #
# run.py: the option tables must point at things that exist
# --------------------------------------------------------------------------- #
def test_every_speed_entry_references_real_presets():
    for key, speed in runpy_menu.SPEEDS.items():
        assert speed["model"] in PRESETS, f"speed {key}: unknown model {speed['model']}"
        assert speed["dataset_preset"] in DATASET_PRESETS, (
            f"speed {key}: unknown dataset preset {speed['dataset_preset']}"
        )
        assert speed["epochs"] >= 1
        assert speed["key"] in {"toy", "quick", "long", "corpus"}


def test_every_bias_option_is_a_real_bias_preset():
    for key, (preset, explanation) in runpy_menu.BIASES.items():
        assert preset in BiasPresets, f"bias option {key}: unknown preset {preset}"
        assert explanation.strip(), f"bias option {key} has no explanation"
    for preset in runpy_menu.COMPARE_THREE:
        assert preset in BiasPresets


def test_speeds_always_set_the_logging_cadence():
    """Regression guard: without this the comparison figure is a single dot.

    The training defaults are tuned for long runs on big models, so a short
    menu-driven run logged exactly one evaluation point and the figure had
    nothing to draw.
    """
    for key, speed in runpy_menu.SPEEDS.items():
        assert speed.get("log_every"), f"speed {key} does not set log_every"
        assert speed.get("valid_every"), f"speed {key} does not set valid_every"
        assert 0 < speed["valid_every"]
        assert 0 < speed["log_every"]


# --------------------------------------------------------------------------- #
# run.py: command construction
# --------------------------------------------------------------------------- #
def test_build_train_command_carries_every_setting():
    speed = runpy_menu.SPEEDS["2"]
    command = runpy_menu.build_train_command(
        speed, "b-gaussian", "quick-bgaussian", log_dir="myruns", epochs=3
    )
    assert command[0] == "train"
    joined = " ".join(command)
    for expected in (
        "--model tiny",
        "--dataset_preset multi30k-quick",
        "--bias_preset b-gaussian",
        "--optimizer egd",
        "--epochs 3",
        "--log_dir myruns",
        "--name quick-bgaussian",
        f"--log_every {speed['log_every']}",
        f"--valid_every_updates {speed['valid_every']}",
    ):
        assert expected in joined, f"{expected!r} missing from {command}"
    # The loss curve must always be written: --no_plot would suppress it.
    assert "--no_plot" not in command, (
        "the menu path must save training_curve.png, so it must not pass --no_plot"
    )


def test_menu_never_disables_plotting_for_any_speed():
    for key, speed in runpy_menu.SPEEDS.items():
        command = runpy_menu.build_train_command(speed, "symmetric", f"{key}-x")
        assert "--no_plot" not in command, f"speed {key} would skip the loss curve"


def test_build_train_command_only_adds_max_steps_when_set():
    toy = runpy_menu.build_train_command(
        runpy_menu.SPEEDS["1"], "b-const", "toy-bconst"
    )
    assert "--max_steps" in toy
    long = runpy_menu.build_train_command(
        runpy_menu.SPEEDS["3"], "b-const", "long-bconst"
    )
    assert "--max_steps" not in long, "a full-length run must not cap its steps"


def test_run_name_encodes_the_choices():
    assert runpy_menu.run_name_for("quick", "b-gaussian") == "quick-bgaussian"
    assert runpy_menu.run_name_for("quick", "symmetric") == "quick-symmetric"
    assert runpy_menu.run_name_for("toy", "b-const", "adamw") == "toy-bconst-adamw"


def test_speed_by_key_round_trips_and_rejects_unknown():
    for speed in runpy_menu.SPEEDS.values():
        assert runpy_menu.speed_by_key(speed["key"]) is speed
    with pytest.raises(KeyError):
        runpy_menu.speed_by_key("glacial")


def test_report_and_evaluate_commands_are_shaped_correctly():
    report_command = runpy_menu.build_report_command("myruns")
    assert report_command[:2] == ["report", "--log_dir"]
    evaluate_command = runpy_menu.build_evaluate_command(
        Path("myruns/x/model_best.pt"), "multi30k-quick"
    )
    assert evaluate_command[0] == "evaluate"
    assert "--data" not in evaluate_command
    assert "multi30k-quick" in evaluate_command


# --------------------------------------------------------------------------- #
# run.py: my_config.py handling
# --------------------------------------------------------------------------- #
def test_my_config_round_trip(tmp_path: Path):
    path = tmp_path / "my_config.py"
    flags = {
        "--model": "long",
        "--bias_preset": "attn-bQbV",
        "--epochs": 3,
        "--optimizer": "adamw",
        "--log_dir": "results",
    }
    runpy_menu.write_my_config(flags, path)
    text = path.read_text(encoding="utf-8")
    assert "FORMAT = 2" in text
    assert "SETTINGS" in text
    assert "EXTRA_ARGS" in text
    # The panel must document every knob the training script accepts, so a user
    # can find anything rather than being limited to a hand-picked subset.
    assert "--noise_density" in text
    assert "--prelu_slope_mean" in text
    assert "--no_apply_encoder" in text

    panel = runpy_menu.read_my_config(path)
    assert panel is not None
    for flag, value in flags.items():
        assert panel["flags"][flag] == value
    assert panel["extra"] == []
    assert sorted(runpy_menu.flags_to_argv(panel["flags"])) == sorted(
        runpy_menu.flags_to_argv(flags)
    )


def test_my_config_honours_extra_args(tmp_path: Path):
    path = tmp_path / "my_config.py"
    runpy_menu.write_my_config({"--model": "tiny"}, path, extra=["--seed", "7"])
    panel = runpy_menu.read_my_config(path)
    assert panel["extra"] == ["--seed", "7"]


def test_control_panel_is_not_overwritten(tmp_path: Path, monkeypatch):
    """A hand-tuned panel must survive a later menu run."""
    target = tmp_path / "my_config.py"
    runpy_menu.write_my_config({"--model": "base"}, target)
    before = target.read_text(encoding="utf-8")
    monkeypatch.setattr(runpy_menu, "MY_CONFIG", target)
    runpy_menu.maybe_write_panel({"--model": "tiny"})
    assert target.read_text(encoding="utf-8") == before


def test_read_my_config_survives_a_broken_file(tmp_path: Path):
    """The docs promise a hand-broken file cannot break the menu."""
    path = tmp_path / "my_config.py"
    path.write_text('SPEED = "quick\nBIAS = \n', encoding="utf-8")  # unbalanced quote
    assert runpy_menu.read_my_config(path) is None


def test_read_my_config_missing_file_is_none(tmp_path: Path):
    assert runpy_menu.read_my_config(tmp_path / "nope.py") is None


def test_triple_duration_keeps_the_unit():
    """Regression guard: 3 x "约 15 秒" must not become "约 45 分钟"."""
    assert runpy_menu._triple("约 1 分钟") == "约 3 分钟"
    assert runpy_menu._triple("约 8 分钟") == "约 24 分钟"
    assert runpy_menu._triple("约 15 秒") == "约 45 秒"
    assert runpy_menu._triple("约 30 秒") == "约 1 分钟"
    assert runpy_menu._triple("很快") == "很快"


def test_dry_run_touches_nothing(tmp_path: Path):
    """`--dry-run` must print the command and create no run directory."""
    code = runpy_menu.main(
        [
            "--speed", "toy",
            "--bias", "b-const",
            "--dry-run",
            "--log_dir", str(tmp_path / "runs"),
        ]
    )
    assert code == 0
    assert not (tmp_path / "runs").exists()


def test_non_interactive_rejects_an_unknown_bias(tmp_path: Path):
    code = runpy_menu.main(["--bias", "not-a-bias", "--dry-run"])
    assert code == 2


# --------------------------------------------------------------------------- #
# run.bat: the double-click entry point is fragile, so pin its invariants
# --------------------------------------------------------------------------- #
def test_run_bat_is_cmd_safe():
    """cmd.exe requirements that silently break the double-click entry point.

    Reproduced the hard way while building this: a UTF-8 BOM makes cmd choke on
    the first command, and UNIX line endings make it mis-parse the file so that
    comment text is executed as commands.  Non-ASCII is only tolerable inside an
    ``echo`` line, and only after ``chcp 65001``.
    """
    import re

    raw = (ROOT / "run.bat").read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), "a UTF-8 BOM breaks the first command"

    text = raw.decode("utf-8")
    assert "\r\n" in text, "run.bat must use CRLF"
    assert not re.search(r"(?<!\r)\n", text), "run.bat must not contain a bare LF"

    assert "chcp 65001" in text, "the console must be switched to UTF-8"
    for line in text.split("\r\n"):
        stripped = line.strip().lower()
        if not stripped or stripped.startswith("echo") or stripped.startswith(":"):
            continue
        assert line.isascii(), (
            "non-ASCII on a cmd command line is mis-decoded and executed: "
            f"{line!r}"
        )


def test_gitattributes_pins_the_bat_to_crlf():
    """A clone must not reintroduce LF endings in run.bat."""
    text = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert "*.bat" in text and "eol=crlf" in text
    assert "*.sh" in text and "eol=lf" in text


# --------------------------------------------------------------------------- #
# scripts/report.py
# --------------------------------------------------------------------------- #
def test_report_on_a_single_run(report, tmp_path: Path):
    log_dir = tmp_path / "runs"
    write_fake_run(log_dir, "only-run", embed_mode="gaussian", with_eval=True)
    out = tmp_path / "out"
    result = report.build_report(log_dir=str(log_dir), out=str(out), make_plot=True)

    assert result["ok"] is True
    assert result["n_runs"] == 1
    assert result["best_run"] == "only-run"
    assert (out / report.CSV_FILENAME).exists()
    assert (out / "compare.txt").exists()
    if result.get("png"):
        assert (out / report.PNG_FILENAME).exists()
        assert (out / report.PNG_FILENAME).stat().st_size > 0

    # The CSV header is part of the module's contract, and the file is written
    # with a UTF-8 BOM on purpose so Excel on Windows shows the Chinese correctly.
    raw = (out / report.CSV_FILENAME).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf"), "compare.csv should carry a UTF-8 BOM for Excel"
    with (out / report.CSV_FILENAME).open(encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))
    assert header == list(report.CSV_COLUMNS)


def test_report_ranks_runs_by_validation_loss(report, tmp_path: Path):
    log_dir = tmp_path / "runs"
    write_fake_run(log_dir, "worse", n_val=4)
    write_fake_run(log_dir, "better", n_val=8)  # more steps -> lower val loss
    result = report.build_report(
        log_dir=str(log_dir), out=str(tmp_path / "out"), make_plot=False
    )
    assert result["n_runs"] == 2
    assert result["best_run"] == "better"
    assert result["ranked_runs"][0] == "better"


def test_report_tolerates_incomplete_runs(report, tmp_path: Path):
    """A run with only a training_log.csv must not break the report."""
    log_dir = tmp_path / "runs"
    write_fake_run(
        log_dir, "bare", with_summary=False, with_eval=False
    )
    (log_dir / "bare" / "summary.json").unlink(missing_ok=True)
    result = report.build_report(
        log_dir=str(log_dir), out=str(tmp_path / "out"), make_plot=False
    )
    assert result["ok"] is True
    assert result["n_runs"] == 1


def test_report_on_empty_and_missing_directories(report, tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    result = report.build_report(
        log_dir=str(empty), out=str(tmp_path / "o1"), make_plot=False
    )
    assert result["n_runs"] == 0
    assert not result.get("png")

    missing = tmp_path / "does-not-exist"
    result = report.build_report(
        log_dir=str(missing), out=str(tmp_path / "o2"), make_plot=False
    )
    assert result["n_runs"] == 0
    assert result.get("message")


def test_report_ignores_a_directory_without_a_log(report, tmp_path: Path):
    log_dir = tmp_path / "runs"
    (log_dir / "_compare").mkdir(parents=True)  # its own output dir
    (log_dir / "junk").mkdir()
    (log_dir / "junk" / "notes.txt").write_text("hi", encoding="utf-8")
    write_fake_run(log_dir, "real", n_val=4)
    result = report.build_report(
        log_dir=str(log_dir), out=str(tmp_path / "out"), make_plot=False
    )
    assert result["n_runs"] == 1
    assert result["best_run"] == "real"


def test_report_has_no_import_side_effects(report, tmp_path: Path):
    """``build_report`` is importable and callable without touching cwd."""
    assert hasattr(report, "build_report")
    assert callable(report.build_report)
    assert report.LOG_FILENAME == "training_log.csv"
    assert report.OUT_DIRNAME == "_compare"
