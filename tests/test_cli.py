"""End-to-end tests through the command line.

These drive the real entry point (``main.py``) in a subprocess, which is the
only way to test a CLI-centric project honestly: argparse wiring, the run
directory layout, the CSV header and the checkpoint round trip are all exercised
exactly as a user would exercise them.

Kept deliberately small: one shared training run is reused by the evaluate /
analyze / plot tests, and the bias-mode comparison uses 4-step runs.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "main.py"
EXAMPLES = ROOT / "examples"

#: A tiny, offline, CPU-cheap run used repeatedly below.
SMOKE = [
    "--model", "smoke",
    "--dataset_preset", "synthetic-copy",
    "--batch_size", "16",
    "--max_steps", "12",
    "--valid_every_updates", "6",
    "--log_every", "4",
    "--no_plot",
]


def run_cli(*args, timeout: int = 900) -> subprocess.CompletedProcess:
    """Invoke ``main.py`` with the given arguments and capture its output."""
    return subprocess.run(
        [sys.executable, str(MAIN), *[str(a) for a in args]],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def joined(result: subprocess.CompletedProcess) -> str:
    return (result.stdout or "") + (result.stderr or "")


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory) -> Path:
    """One real training run (embedding bias + attention biases) shared below."""
    log_dir = tmp_path_factory.mktemp("cli_runs")
    result = run_cli(
        "train", *SMOKE,
        "--bias_preset", "b-gaussian",
        "--use_q_bias", "--use_v_bias",
        "--log_dir", str(log_dir),
        "--name", "shared",
    )
    assert result.returncode == 0, joined(result)
    run_dir = log_dir / "shared"
    assert run_dir.is_dir()
    return run_dir


# --------------------------------------------------------------------------- #
# Dispatch / help
# --------------------------------------------------------------------------- #
def test_help_lists_every_command():
    result = run_cli("--help")
    assert result.returncode == 0, joined(result)
    for command in ("train", "evaluate", "analyze-bias", "plot"):
        assert command in result.stdout


def test_unknown_command_is_rejected():
    result = run_cli("frobnicate")
    assert result.returncode == 2
    assert "unknown command" in joined(result)


def test_train_list_models_prints_the_presets():
    result = run_cli("train", "--list_models")
    assert result.returncode == 0, joined(result)
    for name in ("smoke", "tiny", "small", "base", "large"):
        assert name in result.stdout
    for name in ("symmetric", "b-gaussian", "b-const"):
        assert name in result.stdout


def test_every_flag_used_in_examples_exists():
    """Guard against flag drift between examples/*.sh and scripts/train.py."""
    help_text = joined(run_cli("train", "--help"))
    documented = set(re.findall(r"--[A-Za-z0-9_]+", help_text))
    assert "--model" in documented

    scripts = sorted(EXAMPLES.glob("*.sh"))
    assert scripts, "no example scripts found"
    problems = []
    for script in scripts:
        used = set(re.findall(r"--[A-Za-z0-9_]+", script.read_text(encoding="utf-8")))
        missing = sorted(used - documented)
        if missing:
            problems.append(f"{script.name}: {missing}")
    assert not problems, "examples use flags train.py does not accept: " + "; ".join(problems)


# --------------------------------------------------------------------------- #
# Training outputs
# --------------------------------------------------------------------------- #
def test_training_writes_the_expected_artifacts(trained_run: Path):
    required = [
        "training_log.csv",
        "model_best.pt",
        "model_final.pt",
        "tokenizer_src.json",
        "tokenizer_tgt.json",
    ]
    listing = sorted(p.name for p in trained_run.iterdir())
    for name in required:
        assert (trained_run / name).exists(), f"missing {name}; dir has {listing}"

    # Some record of the resolved configuration must be persisted.
    assert any(p.suffix == ".json" for p in trained_run.iterdir()), listing


def test_training_log_has_the_bias_and_optimizer_columns(trained_run: Path):
    header = (trained_run / "training_log.csv").read_text(encoding="utf-8").splitlines()[0]
    columns = header.split(",")
    for expected in (
        "train_loss",
        "val_loss",
        "embed_b_norm",
        "bQ_norm",
        "bK_norm",
        "bV_norm",
        "egd_iteration",
        "egd_momentum_norm",
        "egd_skipped",
    ):
        assert expected in columns, f"{expected} not in CSV header {columns}"


def test_checkpoint_carries_dict_configs(trained_run: Path):
    import torch

    ckpt = torch.load(trained_run / "model_best.pt", map_location="cpu")
    assert isinstance(ckpt.get("config"), dict), "config should be a plain dict"
    assert isinstance(ckpt.get("bias"), dict), "bias should be a plain dict"
    from symbreak_transformer.config import BiasConfig, Seq2SeqConfig

    cfg = Seq2SeqConfig.from_dict(ckpt["config"])
    bias = BiasConfig.from_dict(ckpt["bias"])
    assert cfg.n_embd > 0
    assert bias.embed_mode == "gaussian"


# --------------------------------------------------------------------------- #
# The other subcommands
# --------------------------------------------------------------------------- #
def test_evaluate_scores_a_checkpoint(trained_run: Path):
    result = run_cli(
        "evaluate",
        "--ckpt", trained_run / "model_best.pt",
        "--split", "test",
        "--max_samples", "32",
        "--dataset_preset", "synthetic-copy",
    )
    assert result.returncode == 0, joined(result)
    assert "BLEU" in result.stdout
    out = trained_run / "eval_test.json"
    assert out.exists(), joined(result)
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["split"] == "test"
    assert "bleu" in payload and "loss" in payload


def test_analyze_bias_measures_a_non_zero_effect(trained_run: Path, tmp_path: Path):
    out = tmp_path / "bias_comparison.json"
    result = run_cli(
        "analyze-bias",
        "--ckpt", trained_run / "model_best.pt",
        "--split", "test",
        "--batches", "2",
        "--max_samples", "64",
        "--dataset_preset", "synthetic-copy",
        "--no_plot",
        "--out", out,
    )
    assert result.returncode == 0, joined(result)
    assert out.exists(), joined(result)
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert len(rows) == 1
    row = rows[0]
    # The embedding bias is a non-zero gaussian, so zeroing it must change the
    # model output -- this is the whole point of the analysis.
    assert row["inventory"]["embed"]["norm"] > 0.0
    assert row["inventory"]["embed"]["is_exactly_zero"] is False
    assert row["effect"]["logit_rel_change"] > 0.0
    assert row["inventory"]["attention_sectors"], "attention biases were enabled"


def test_plot_writes_a_png(trained_run: Path):
    result = run_cli("plot", "--run", trained_run)
    assert result.returncode == 0, joined(result)
    png = trained_run / "training_curve.png"
    if not png.exists():
        pytest.skip(f"matplotlib unavailable: {joined(result)[-400:]}")
    assert png.stat().st_size > 0


def test_report_compares_the_finished_runs(trained_run: Path):
    """`main.py report` turns a log directory into one report + one figure."""
    log_dir = trained_run.parent
    result = run_cli("report", "--log_dir", log_dir)
    assert result.returncode == 0, joined(result)

    out = log_dir / "_compare"
    assert (out / "compare.txt").exists(), joined(result)
    assert (out / "compare.csv").exists(), joined(result)
    text = (out / "compare.txt").read_text(encoding="utf-8")
    assert trained_run.name in text, "the finished run is missing from the report"
    if (out / "compare.png").exists():
        assert (out / "compare.png").stat().st_size > 0


# --------------------------------------------------------------------------- #
# The three bias modes, end to end
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("preset", "expect_zero"),
    [("symmetric", True), ("b-gaussian", False), ("b-const", False)],
)
def test_bias_modes_train_and_log_the_expected_bias(
    tmp_path: Path, preset: str, expect_zero: bool
):
    log_dir = tmp_path / "runs"
    result = run_cli(
        "train",
        "--model", "smoke",
        "--dataset_preset", "synthetic-copy",
        "--batch_size", "16",
        "--max_steps", "4",
        "--valid_every_updates", "4",
        "--log_every", "2",
        "--bias_preset", preset,
        "--log_dir", str(log_dir),
        "--name", preset,
        "--no_plot",
    )
    assert result.returncode == 0, joined(result)

    run_dir = log_dir / preset
    assert (run_dir / "model_final.pt").exists()
    rows = [
        line.split(",")
        for line in (run_dir / "training_log.csv").read_text(encoding="utf-8").splitlines()
    ]
    header, body = rows[0], rows[1:]
    assert header, "empty CSV"
    index = header.index("embed_b_norm")
    values = [float(r[index]) for r in body if len(r) > index and r[index] not in ("", "nan")]
    assert values, f"no embed_b_norm rows logged; header={header}"
    if expect_zero:
        assert all(v == 0.0 for v in values), f"{preset} should log a zero embedding bias"
    else:
        assert any(v > 0.0 for v in values), f"{preset} should log a non-zero embedding bias"


def test_attention_bias_presets_train_end_to_end(tmp_path: Path):
    log_dir = tmp_path / "runs"
    result = run_cli(
        "train",
        "--model", "smoke",
        "--dataset_preset", "synthetic-copy",
        "--batch_size", "16",
        "--max_steps", "4",
        "--valid_every_updates", "4",
        "--log_every", "2",
        "--bias_preset", "attn-bQbV",
        "--log_dir", str(log_dir),
        "--name", "attn",
        "--no_plot",
    )
    assert result.returncode == 0, joined(result)
    csv = (log_dir / "attn" / "training_log.csv").read_text(encoding="utf-8")
    header, *body = csv.splitlines()
    columns = header.split(",")
    bq = columns.index("bQ_norm")
    bk = columns.index("bK_norm")
    bv = columns.index("bV_norm")
    last = body[-1].split(",")
    assert float(last[bq]) > 0.0, "bQ should be active"
    assert float(last[bv]) > 0.0, "bV should be active"
    assert float(last[bk]) == 0.0, "bK is off by default"


# --------------------------------------------------------------------------- #
# Resuming
# --------------------------------------------------------------------------- #
def test_resume_continues_from_a_checkpoint(tmp_path: Path):
    """`--resume` restores model + optimizer and continues to the requested step."""
    log_dir = tmp_path / "runs"
    base = [
        "--model", "smoke",
        "--dataset_preset", "synthetic-copy",
        "--batch_size", "16",
        "--valid_every_updates", "6",
        "--log_every", "6",
        "--bias_preset", "b-gaussian",
        "--log_dir", str(log_dir),
        "--name", "resumed",
        "--no_plot",
    ]

    first = run_cli("train", *base, "--max_steps", "6")
    assert first.returncode == 0, joined(first)
    ckpt = log_dir / "resumed" / "model_final.pt"
    assert ckpt.exists(), sorted(p.name for p in (log_dir / "resumed").iterdir())

    second = run_cli("train", *base, "--max_steps", "12", "--resume", ckpt)
    assert second.returncode == 0, joined(second)
    assert "[resume]" in joined(second), "the resume was not reported"

    summary = json.loads(
        (log_dir / "resumed" / "summary.json").read_text(encoding="utf-8")
    )
    assert summary["steps"] == 12, f"resume did not advance to 12: {summary['steps']}"

    # The log is appended to rather than truncated: header + rows from both runs.
    rows = (log_dir / "resumed" / "training_log.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) >= 5, f"the log was truncated on resume: {rows}"


def test_resume_warns_but_continues_on_a_bad_checkpoint(tmp_path: Path):
    """A missing or incompatible checkpoint must not abort the run."""
    log_dir = tmp_path / "runs"
    missing = tmp_path / "not-a-checkpoint.pt"
    result = run_cli(
        "train",
        "--model", "smoke",
        "--dataset_preset", "synthetic-copy",
        "--batch_size", "16",
        "--max_steps", "2",
        "--valid_every_updates", "2",
        "--log_every", "1",
        "--bias_preset", "symmetric",
        "--log_dir", str(log_dir),
        "--name", "badresume",
        "--resume", missing,
        "--no_plot",
    )
    assert result.returncode == 0, joined(result)
    assert (log_dir / "badresume" / "model_final.pt").exists()
