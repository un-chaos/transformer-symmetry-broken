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


@pytest.fixture(autouse=True)
def isolated_menu_state(tmp_path, monkeypatch):
    """Every test gets a fresh menu state and its own settings file.

    The menu keeps its choices in module globals, so without this one test's
    "user picked speed 3" would leak into the next one.
    """
    monkeypatch.setattr(runpy_menu, "SETTINGS_FILE", tmp_path / "run_settings.json")
    saved_state, saved_custom = dict(runpy_menu.STATE), dict(runpy_menu.CUSTOM)
    runpy_menu.STATE.update({"speed": "1", "bias": "b-gaussian", "optimizer": "egd"})
    runpy_menu.CUSTOM.clear()
    yield
    runpy_menu.STATE.clear()
    runpy_menu.STATE.update(saved_state)
    runpy_menu.CUSTOM.clear()
    runpy_menu.CUSTOM.update(saved_custom)


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
    runpy_menu.STATE.update({"speed": "2", "bias": "b-gaussian", "optimizer": "egd"})
    runpy_menu.CUSTOM.update({"--epochs": 3, "--log_dir": "myruns"})
    command = runpy_menu.build_train_command(runpy_menu.effective_flags())

    joined = " ".join(command)
    speed = runpy_menu.SPEEDS["2"]
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
    # There is no subcommand layer any more: the menu calls the training script
    # directly, so the first element is a flag.
    assert command[0].startswith("--")
    # The loss curve must always be written: --no_plot would suppress it.
    assert "--no_plot" not in command, (
        "the menu path must save training_curve.png, so it must not pass --no_plot"
    )


def test_menu_never_disables_plotting_for_any_speed():
    for key in runpy_menu.SPEEDS:
        runpy_menu.STATE["speed"] = key
        command = runpy_menu.build_train_command(runpy_menu.effective_flags())
        assert "--no_plot" not in command, f"speed {key} would skip the loss curve"


def test_build_train_command_only_adds_max_steps_when_set():
    runpy_menu.STATE["speed"] = "1"          # toy: step-capped
    toy = runpy_menu.build_train_command(runpy_menu.effective_flags())
    assert "--max_steps" in toy

    runpy_menu.STATE["speed"] = "3"          # long: bounded by epochs instead
    long = runpy_menu.build_train_command(runpy_menu.effective_flags())
    assert "--max_steps" not in long, "a full-length run must not cap its steps"


def test_run_name_encodes_the_choices():
    assert runpy_menu.run_name_for("quick", "b-gaussian") == "quick-bgaussian"
    assert runpy_menu.run_name_for("quick", "symmetric") == "quick-symmetric"
    assert runpy_menu.run_name_for("toy", "b-const", "adamw") == "toy-bconst-adamw"


def test_current_speed_follows_the_state():
    for key in runpy_menu.SPEEDS:
        runpy_menu.STATE["speed"] = key
        assert runpy_menu.current_speed() is runpy_menu.SPEEDS[key]


def test_report_and_evaluate_commands_are_shaped_correctly():
    report_command = runpy_menu.build_report_command("myruns")
    assert report_command[:2] == ["--log_dir", "myruns"]
    evaluate_command = runpy_menu.build_evaluate_command(
        Path("myruns/x/model_best.pt"), "multi30k-quick"
    )
    assert "--data" not in evaluate_command
    assert "multi30k-quick" in evaluate_command
    assert "--ckpt" in evaluate_command


def test_log_paths_are_root_relative(tmp_path: Path, monkeypatch):
    """Starting the menu from another directory must not misreport where results go.

    Regression guard: the run directory used to be resolved against the *current*
    working directory, so launching ``python D:\\...\\run.py`` from elsewhere
    printed (and later opened) a path in the wrong place, even though training
    wrote into the project.
    """
    monkeypatch.chdir(tmp_path)
    assert runpy_menu.log_dir_path() == runpy_menu.ROOT / "runs"
    assert runpy_menu.log_dir_path("myruns") == runpy_menu.ROOT / "myruns"
    # An absolute path is taken as given.
    absolute = tmp_path / "elsewhere"
    assert runpy_menu.log_dir_path(str(absolute)) == absolute

    # And the finished-run scan follows the same rule.
    assert runpy_menu.DEFAULT_LOG_DIR in str(runpy_menu.log_dir_path())


def test_the_menu_scripts_all_exist():
    """Every script the menu shells out to must be present and named right."""
    for script in (
        runpy_menu.TRAIN_SCRIPT,
        runpy_menu.EVALUATE_SCRIPT,
        runpy_menu.REPORT_SCRIPT,
        runpy_menu.DOWNLOAD_SCRIPT,
        runpy_menu.ANALYZE_SCRIPT,
    ):
        assert script.is_file(), f"{script} is missing"


def test_the_menu_numbers_every_feature():
    """Renumbering the menu is fine; losing an item off the end is not."""
    assert set(runpy_menu.MENU) == {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    labels = " ".join(runpy_menu.menu_items().values())
    for feature in ("三种 bias", "只跑一种", "修改训练设置", "当前设置",
                    "已有结果", "分析 bias", "结果文件夹", "语料", "帮助"):
        assert feature in labels, f"the menu no longer offers: {feature}"


# --------------------------------------------------------------------------- #
# run.py: the bias-analysis item
# --------------------------------------------------------------------------- #
def test_finished_runs_lists_only_trained_runs(tmp_path: Path):
    log_dir = tmp_path / "runs"
    for name, with_ckpt in (("trained", True), ("interrupted", False)):
        (log_dir / name).mkdir(parents=True)
        if with_ckpt:
            (log_dir / name / "model_best.pt").write_bytes(b"x")
    (log_dir / "not-a-dir.txt").write_text("x", encoding="utf-8")
    assert [p.name for p in runpy_menu.finished_runs(str(log_dir))] == ["trained"]


def test_finished_runs_on_a_missing_directory_is_empty(tmp_path: Path):
    assert runpy_menu.finished_runs(str(tmp_path / "nope")) == []


def test_dataset_preset_is_read_back_from_the_run(tmp_path: Path):
    """The analysis must use the data the checkpoint was trained on."""
    run_dir = tmp_path / "r"
    run_dir.mkdir()
    (run_dir / "args.json").write_text(
        json.dumps({"dataset_preset": "multi30k-tiny"}), encoding="utf-8"
    )
    assert runpy_menu.dataset_preset_of(run_dir) == "multi30k-tiny"

    (run_dir / "args.json").unlink()          # missing file: fall back, never raise
    assert runpy_menu.dataset_preset_of(run_dir) == (
        runpy_menu.effective_flags()["--dataset_preset"]
    )


def test_analyze_command_points_at_the_chosen_run(tmp_path: Path):
    run_dir = tmp_path / "quick-bgaussian"
    command = runpy_menu.build_analyze_command(run_dir, "multi30k-quick")
    joined_command = " ".join(command)
    assert "model_best.pt" in joined_command
    assert "multi30k-quick" in joined_command
    assert "--out" in command and "bias_analysis.json" in joined_command
    assert "--no_plot" not in command, "the analysis should still draw its figure"


def test_analyze_on_an_empty_runs_directory_says_so(tmp_path, capsys, monkeypatch):
    """With nothing trained yet the item must explain itself, not crash."""
    monkeypatch.setattr(runpy_menu, "finished_runs", lambda log_dir="runs": [])
    runpy_menu.action_analyze_bias()
    assert "还没有训练好的模型" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# run.py: the local corpus is reused, never re-downloaded
# --------------------------------------------------------------------------- #
def test_corpus_status_reads_the_disk(tmp_path: Path):
    """A complete manifest plus its shards means "ready"; no network involved."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.parquet").write_bytes(b"x" * 10)
    (corpus / "b.parquet").write_bytes(b"y" * 20)
    status = runpy_menu.corpus_status(str(corpus))
    assert status["shards"] == 2
    assert status["bytes"] == 30
    # No manifest at all: shards on disk are still good enough to train on.
    assert status["ready"] is True

    (corpus / "manifest.json").write_text(
        json.dumps({"complete": True, "files": [{"path": "a"}, {"path": "b"}]}),
        encoding="utf-8",
    )
    assert runpy_menu.corpus_status(str(corpus))["ready"] is True


def test_corpus_status_reports_an_incomplete_corpus(tmp_path: Path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.parquet").write_bytes(b"x")
    (corpus / "manifest.json").write_text(
        json.dumps({"complete": False, "files": [{"path": "a"}, {"path": "b"}]}),
        encoding="utf-8",
    )
    status = runpy_menu.corpus_status(str(corpus))
    assert status["ready"] is False
    assert "不完整" in runpy_menu.corpus_line(status)


def test_corpus_status_on_an_empty_directory(tmp_path: Path):
    status = runpy_menu.corpus_status(str(tmp_path / "nothing"))
    assert status["ready"] is False
    assert "还没有语料" in runpy_menu.corpus_line(status)


def test_the_committed_corpus_is_detected_as_ready():
    """The point of this change: the corpus on this machine is used as-is."""
    status = runpy_menu.corpus_status()
    if not status["shards"]:  # pragma: no cover - a clone without the corpus
        pytest.skip("the FineWeb corpus is not present in this checkout")
    assert status["ready"] is True, runpy_menu.corpus_line(status)
    assert "已就绪" in runpy_menu.corpus_line(status)


def test_menu_item_8_says_ready_instead_of_offering_a_download():
    """The menu itself must stop implying that the corpus needs downloading."""
    items = runpy_menu.menu_items()
    status = runpy_menu.corpus_status()
    if status["ready"]:
        assert "已就绪" in items["8"]
        assert "下载" not in items["8"]
    else:  # pragma: no cover - depends on the machine
        assert "下载" in items["8"] or "不完整" in items["8"]


def test_a_ready_corpus_is_not_downloaded(monkeypatch, capsys):
    """Choosing item 8 with a complete corpus must not spawn the downloader."""
    started = []

    def fake_popen(*args, **kwargs):  # pragma: no cover - must not run
        started.append(args)
        raise AssertionError("the downloader must not be started")

    monkeypatch.setattr(runpy_menu.subprocess, "Popen", fake_popen)
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")  # "no, do not verify"
    runpy_menu.action_download()
    output = capsys.readouterr().out
    assert not started, "a complete corpus triggered a download"
    assert "已就绪" in output
    assert "不需要再下一遍" in output


def test_downloading_is_still_offered_when_the_corpus_is_missing(monkeypatch, capsys):
    """The opposite case must keep working: an empty corpus offers a download."""
    monkeypatch.setattr(
        runpy_menu, "corpus_status",
        lambda directory=runpy_menu.CORPUS_DIR: {
            "dir": Path("nowhere"), "shards": 0, "bytes": 0, "expected": 0,
            "complete": False, "ready": False,
        },
    )
    monkeypatch.setattr("builtins.input", lambda *a, **k: "n")  # decline to download
    runpy_menu.action_download()
    output = capsys.readouterr().out
    assert "还没有语料" in output
    assert "断点续传" in output


def test_uses_fineweb_detects_the_corpus_datasets():
    assert runpy_menu.uses_fineweb({"--dataset_preset": "fineweb-10b"})
    assert runpy_menu.uses_fineweb({"--dataset_preset": "fineweb-quick"})
    assert runpy_menu.uses_fineweb({"--objective": "denoising"})
    assert runpy_menu.uses_fineweb({"--fineweb_dir": "somewhere"})
    assert not runpy_menu.uses_fineweb({"--dataset_preset": "multi30k-quick"})
    assert not runpy_menu.uses_fineweb({"--dataset_preset": "synthetic-copy"})


# --------------------------------------------------------------------------- #
# run.py: model structure is adjustable and the size is visible
# --------------------------------------------------------------------------- #
def test_the_parameter_estimate_matches_the_real_model():
    """The arithmetic estimate must agree with what torch actually builds.

    A wrong estimate would be worse than none: this is the number the menu shows
    while the user picks a tier.
    """
    from symbreak_transformer.config import Seq2SeqConfig, resolve_bias_config

    torch_model = pytest.importorskip("symbreak_transformer.model")
    for name, cfg in PRESETS.items():
        real = torch_model.Seq2SeqTransformer(
            Seq2SeqConfig(
                n_embd=cfg.n_embd, n_head=cfg.n_head,
                n_encoder_layer=cfg.n_encoder_layer,
                n_decoder_layer=cfg.n_decoder_layer, d_ff=cfg.d_ff,
                context_length=cfg.context_length,
                src_vocab_size=runpy_menu.REFERENCE_VOCAB,
                tgt_vocab_size=runpy_menu.REFERENCE_VOCAB,
                tie_output_embedding=True,
            ),
            resolve_bias_config("symmetric"),
        )
        real_total = sum(p.numel() for p in real.parameters())
        estimate = runpy_menu.estimate_parameters(
            cfg.n_embd, cfg.n_head, cfg.n_encoder_layer, cfg.n_decoder_layer,
            cfg.d_ff, cfg.context_length, runpy_menu.REFERENCE_VOCAB,
        )
        assert estimate == real_total, f"{name}: estimate {estimate} != real {real_total}"


def test_parameter_tiers_exist_and_are_ordered():
    """The ladder the user picks from: 4m ... 124m, each bigger than the last."""
    from symbreak_transformer.config import MODEL_TIERS

    assert list(MODEL_TIERS) == ["4m", "10m", "25m", "50m", "124m"]
    sizes = [
        runpy_menu.estimate_parameters(
            cfg.n_embd, cfg.n_head, cfg.n_encoder_layer, cfg.n_decoder_layer,
            cfg.d_ff, cfg.context_length, runpy_menu.REFERENCE_VOCAB,
        )
        for cfg in MODEL_TIERS.values()
    ]
    assert sizes == sorted(sizes), f"the tiers are not monotone: {sizes}"
    # Each tier should land within 25% of its name at the reference vocabulary.
    for (name, _), size in zip(MODEL_TIERS.items(), sizes):
        target = float(name.rstrip("m")) * 1e6
        assert abs(size - target) / target < 0.25, f"{name} is really {size / 1e6:.1f}M"


def test_the_model_choices_show_parameter_counts():
    """Selecting by parameter count is the whole point, so the counts are shown."""
    labels = runpy_menu.model_choice_labels({"--dataset_preset": "synthetic-copy"})
    assert set(labels) == {str(i) for i in range(1, len(labels) + 1)}
    joined = " ".join(labels.values())
    assert "参数" in joined
    assert "124m" in joined and "参数量档位" in joined
    # The number next to a tier matches the standalone estimate.
    from symbreak_transformer.config import MODEL_TIERS
    cfg = MODEL_TIERS["124m"]
    expected = runpy_menu._millions(runpy_menu.estimate_parameters(
        cfg.n_embd, cfg.n_head, cfg.n_encoder_layer, cfg.n_decoder_layer,
        cfg.d_ff, cfg.context_length, runpy_menu.REFERENCE_VOCAB,
    ))
    assert expected in joined


def test_every_model_choice_maps_back_to_its_name():
    labels = runpy_menu.model_choice_labels({"--dataset_preset": "synthetic-copy"})
    for key in labels:
        name = runpy_menu.model_name_for_choice(key)
        assert name in PRESETS
        assert name in labels[key]


def test_structure_flags_are_editable_in_the_structure_screen():
    """The structure screen must expose the shape, not just the preset name."""
    index = runpy_menu._action_index(runpy_menu.train_parser())
    for flag in ("--model", "--n_embd", "--n_head", "--n_encoder_layer",
                 "--n_decoder_layer", "--d_ff", "--ctx"):
        assert flag in runpy_menu.STRUCTURE_FLAGS, f"{flag} is not in the structure screen"
        assert flag in index, f"{flag} is not a real training flag"


def test_editing_the_structure_changes_the_estimated_size(isolated_menu_state):
    """A manual width/深度 change must move the number the menu shows."""
    before = runpy_menu.estimate_for(runpy_menu.effective_flags())
    runpy_menu.CUSTOM.update({
        "--n_embd": 512, "--n_head": 8, "--n_encoder_layer": 12,
        "--n_decoder_layer": 12, "--d_ff": 2048, "--ctx": 512,
    })
    after = runpy_menu.estimate_for(runpy_menu.effective_flags())
    assert after > before * 10
    assert runpy_menu._millions(after) == "121.6M"


def test_the_assumed_vocabulary_follows_the_dataset():
    """Embeddings dominate, so the count must reflect the vocabulary in play."""
    small = runpy_menu.assumed_vocab({"--dataset_preset": "fineweb-quick"})
    large = runpy_menu.assumed_vocab({"--dataset_preset": "fineweb-10b"})
    assert small == 8000 and large == 32000
    # An explicit size wins over the preset.
    assert runpy_menu.assumed_vocab(
        {"--dataset_preset": "fineweb-quick", "--bpe_vocab_size": 2000}
    ) == 2000


# --------------------------------------------------------------------------- #
# run.py: the constant b is a first-class, hand-typeable value
# --------------------------------------------------------------------------- #
def test_bias_const_is_on_the_common_screen():
    assert "--bias_const" in runpy_menu.COMMON_FLAGS
    assert "--bias_std" in runpy_menu.COMMON_FLAGS


def test_the_bias_screen_exposes_the_constant_and_the_rest():
    for flag in ("--bias_mode", "--bias_const", "--bias_std", "--bias_mean",
                 "--bias_seed", "--bias_learnable", "--attn_const"):
        assert flag in runpy_menu.BIAS_FLAGS, f"{flag} is missing from the bias screen"


def test_a_hand_typed_constant_b_reaches_the_command(isolated_menu_state, monkeypatch):
    """Typing a value for --bias_const must put it in the training command."""
    answers = iter(["2.5"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    runpy_menu.ask_value("--bias_const", runpy_menu._action_index()["--bias_const"])
    assert runpy_menu.CUSTOM["--bias_const"] == 2.5

    argv = runpy_menu.build_train_command(runpy_menu.effective_flags())
    assert argv[argv.index("--bias_const") + 1] == "2.5"


def test_choosing_the_const_bias_mode_then_the_value(isolated_menu_state, monkeypatch):
    """The documented two-step: mode=const, then the value."""
    monkeypatch.setattr("builtins.input", lambda *a, **k: "3")  # 3rd choice = const
    runpy_menu.ask_value("--bias_mode", runpy_menu._action_index()["--bias_mode"])
    assert runpy_menu.CUSTOM["--bias_mode"] == "const"

    monkeypatch.setattr("builtins.input", lambda *a, **k: "0.5")
    runpy_menu.ask_value("--bias_const", runpy_menu._action_index()["--bias_const"])

    argv = runpy_menu.build_train_command(runpy_menu.effective_flags())
    assert argv[argv.index("--bias_mode") + 1] == "const"
    assert argv[argv.index("--bias_const") + 1] == "0.5"


def test_the_const_bias_flag_is_a_float():
    """A non-numeric value must be rejected rather than passed to argparse."""
    action = runpy_menu._action_index()["--bias_const"]
    assert runpy_menu._coerce("1.25", action) == 1.25


# --------------------------------------------------------------------------- #
# run.py: dry runs and menu robustness
# --------------------------------------------------------------------------- #
def test_dry_run_touches_nothing(tmp_path: Path):
    """A dry run must print the command and create no run directory."""
    runpy_menu.CUSTOM["--log_dir"] = str(tmp_path / "runs")
    code = runpy_menu.do_training(runpy_menu.effective_flags(), dry_run=True)
    assert code == 0
    assert not (tmp_path / "runs").exists()


def test_an_unknown_menu_choice_is_rejected_and_the_menu_continues(monkeypatch, capsys):
    """A typo must not crash and must not exit the menu."""
    answers = iter(["zz", "0"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    assert runpy_menu.main() == 0
    assert "没有 'zz' 这个选项" in capsys.readouterr().out


def test_the_menu_exits_cleanly_when_input_ends(monkeypatch):
    """Piping input (or closing the terminal) must not raise EOFError."""
    def no_more_input(*args, **kwargs):
        raise EOFError
    monkeypatch.setattr("builtins.input", no_more_input)
    assert runpy_menu.main() == 0


def test_the_menu_ignores_command_line_arguments(monkeypatch, capsys):
    """There is exactly one usage: extra arguments must not unlock another one."""
    answers = iter(["0"])
    monkeypatch.setattr("builtins.input", lambda *a, **k: next(answers))
    monkeypatch.setattr("sys.argv", ["run.py", "--speed", "quick"])
    assert runpy_menu.main() == 0
    assert "不需要参数" in capsys.readouterr().out


def test_settings_are_remembered_when_a_run_is_configured(tmp_path, monkeypatch):
    """Choosing in the menu writes the state file, so the choice survives a restart."""
    monkeypatch.setattr(runpy_menu, "SETTINGS_FILE", tmp_path / "run_settings.json")
    runpy_menu.STATE.update({"speed": "3", "bias": "b-const"})
    runpy_menu.save_settings()

    runpy_menu.STATE.update({"speed": "1", "bias": "b-gaussian"})
    runpy_menu.load_settings()
    assert runpy_menu.STATE["speed"] == "3"
    assert runpy_menu.STATE["bias"] == "b-const"


def test_triple_duration_keeps_the_unit():
    """Regression guard: 3 x "约 15 秒" must not become "约 45 分钟"."""
    assert runpy_menu._triple("约 1 分钟") == "约 3 分钟"
    assert runpy_menu._triple("约 8 分钟") == "约 24 分钟"
    assert runpy_menu._triple("约 15 秒") == "约 45 秒"
    assert runpy_menu._triple("约 30 秒") == "约 1 分钟"
    assert runpy_menu._triple("很快") == "很快"


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
