"""Tests for the evaluation helpers and the training entry point.

These cover the metrics in ``symbreak_transformer/evaluate.py`` and the
end-to-end behaviour of ``scripts/train.py`` driven through ``main.py``.  The
BLEU expectations are hand-computed, not generated from the implementation:

* ``perfect``   : p1=p2=p3=p4=1, lengths equal -> BP=1, BLEU=100
* ``partial``   : p1=1/2, p2=1/3, p3=0, p4=0 with equal lengths and add-1
                  smoothing -> 100 * (1/2 * 1/3 * 1/3 * 1/2)^(1/4) = 40.8248
* ``short_hyp`` : all four precisions 1 but hyp_len=4, ref_len=6
                  -> BP = exp(1 - 3/2) = 0.606531, BLEU = 60.6531

The training loop lives in the script ``scripts/train.py``, so the end-to-end
tests run ``python main.py train ...`` as a **real subprocess** and assert on the
run directory it leaves behind.  That is the honest way to test a CLI-centric
project: it exercises argparse, config resolution, the data pipeline, the
optimizer and the checkpoint/log writing exactly as a user does.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from symbreak_transformer.config import (
    BiasConfig,
    DataConfig,
    Seq2SeqConfig,
    resolve_bias_config,
    resolve_config,
)
from symbreak_transformer.data import PAD_ID, Tokenizer
from symbreak_transformer.evaluate import corpus_bleu, evaluate_loss, greedy_translate

#: Repository root (``conftest.py`` already puts it on ``sys.path``).
ROOT = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------- #
# BLEU
# --------------------------------------------------------------------------- #
def test_bleu_perfect_match_is_100():
    hyp = [["a", "b", "c", "d"]]
    ref = [["a", "b", "c", "d"]]
    result = corpus_bleu(hyp, ref)
    assert result["bleu"] == pytest.approx(100.0, abs=1e-9)
    assert result["bleu_strict"] == pytest.approx(100.0, abs=1e-9)
    assert result["bp"] == pytest.approx(1.0)
    assert result["precisions"] == pytest.approx([100.0, 100.0, 100.0, 100.0])


def test_bleu_partial_match_matches_hand_computation():
    hyp = [["a", "b", "c", "d"]]
    ref = [["a", "b", "x", "y"]]
    result = corpus_bleu(hyp, ref)
    # clipped 1-grams: a, b -> 2/4; 2-grams: (a,b) -> 1/3; 3- and 4-grams: none.
    assert result["precisions"] == pytest.approx([50.0, 100.0 / 3.0, 0.0, 0.0])
    # Strict BLEU is zero because an order has zero precision...
    assert result["bleu_strict"] == pytest.approx(0.0)
    # ...while add-1 smoothing replaces 0 by 1/(total_n + 1).
    expected = 100.0 * (0.5 * (1.0 / 3.0) * (1.0 / 3.0) * 0.5) ** 0.25
    assert result["bleu"] == pytest.approx(expected, abs=1e-6)
    assert result["bleu"] == pytest.approx(40.8248, abs=1e-3)


def test_bleu_brevity_penalty():
    hyp = [["a", "b", "c", "d"]]
    ref = [["a", "b", "c", "d", "e", "f"]]
    result = corpus_bleu(hyp, ref)
    expected_bp = math.exp(1.0 - 6.0 / 4.0)
    assert result["bp"] == pytest.approx(expected_bp, abs=1e-9)
    assert result["bleu"] == pytest.approx(100.0 * expected_bp, abs=1e-6)
    assert result["bleu"] == pytest.approx(60.6531, abs=1e-3)
    assert result["ratio"] == pytest.approx(4.0 / 6.0)


def test_bleu_smoothing_none_equals_strict():
    hyp = [["a", "b", "c", "d"]]
    ref = [["a", "b", "x", "y"]]
    result = corpus_bleu(hyp, ref, smoothing="none")
    assert result["bleu"] == result["bleu_strict"]


def test_bleu_rejects_bad_input():
    with pytest.raises(ValueError):
        corpus_bleu([["a"]], [["a"], ["b"]])
    with pytest.raises(ValueError):
        corpus_bleu([["a"]], [["a"]], smoothing="bogus")


def test_bleu_is_bounded_and_reports_counts():
    hyp = [["x", "y"], ["a", "b", "c", "d"]]
    ref = [["a", "b"], ["a", "b", "c", "d"]]
    result = corpus_bleu(hyp, ref)
    assert 0.0 <= result["bleu"] <= 100.0
    assert result["n_samples"] == 2
    assert result["hyp_len"] == 6 and result["ref_len"] == 6
    assert len(result["precisions"]) == 4


# --------------------------------------------------------------------------- #
# Fixtures: a tiny synthetic setup
# --------------------------------------------------------------------------- #
@dataclass
class TinyConfig:
    """The three configs one tiny experiment needs, in one object.

    The old single ``ExperimentConfig`` (and the YAML loader that built it from a
    nested dict) is gone: the model shape comes from a preset plus overrides via
    :func:`resolve_config`, the bias from :func:`resolve_bias_config`, and the
    data from a plain :class:`DataConfig`.  The training knobs that used to sit in
    ``cfg.train`` are command-line arguments of ``scripts/train.py`` now, so the
    subprocess tests pass them as flags.
    """

    model: Seq2SeqConfig
    bias: BiasConfig
    data: DataConfig
    #: ``scripts/train.py --seed``; also seeds the shuffled dataloaders.
    seed: int = 7


def tiny_cfg(**overrides) -> TinyConfig:
    """A small, fast config on the built-in synthetic task.

    ``overrides`` are dotted paths in the **new** field names, e.g.
    ``model__n_embd=64``, ``bias__embed_mode="gaussian"`` or
    ``data__synthetic_task="copy"``.
    """
    run = TinyConfig(
        model=resolve_config(
            "smoke",
            context_length=12,
            n_embd=32,
            n_head=2,
            n_encoder_layer=1,
            n_decoder_layer=1,
            d_ff=64,
            dropout=0.0,
            attention_dropout=0.0,
        ),
        bias=resolve_bias_config(None),
        data=DataConfig(
            source="synthetic",
            synthetic_task="reverse",
            synthetic_train_size=256,
            synthetic_val_size=64,
            synthetic_test_size=64,
            synthetic_vocab=12,
            synthetic_len=6,
            min_freq=1,
            max_vocab=64,
        ).validate(),
    )
    for dotted, value in overrides.items():
        section, _, field = dotted.partition("__")
        if not section or not field:
            raise ValueError(f"override {dotted!r} must look like 'model__n_embd'")
        target = getattr(run, section)
        if not hasattr(target, field):
            raise ValueError(f"unknown override {dotted!r}")
        setattr(target, field, value)
    return run


def build_model_and_loader(cfg: TinyConfig, batch_size: int = 32):
    """Build a model plus loaders through the real pipeline."""
    from symbreak_transformer.data import (
        build_dataloaders,
        build_tokenizers,
        load_all_splits,
    )
    from symbreak_transformer.model import Seq2SeqTransformer

    pairs = load_all_splits(cfg.data, ("train", "val", "test"))
    src_tok, tgt_tok = build_tokenizers(cfg.data, pairs)
    model = Seq2SeqTransformer(
        cfg.model,
        cfg.bias,
        src_vocab_size=src_tok.vocab_size,
        tgt_vocab_size=tgt_tok.vocab_size,
        pad_id=PAD_ID,
        bos_id=Tokenizer.bos_id,
        eos_id=Tokenizer.eos_id,
    )
    loaders = build_dataloaders(
        cfg.data,
        src_tok,
        tgt_tok,
        splits=("train", "val"),
        batch_size=batch_size,
        seed=cfg.seed,
    )
    return model, loaders, pairs, src_tok, tgt_tok


# --------------------------------------------------------------------------- #
# Evaluation helpers
# --------------------------------------------------------------------------- #
def test_evaluate_loss_is_finite():
    cfg = tiny_cfg()
    model, loaders, _, _, _ = build_model_and_loader(cfg)
    loss = evaluate_loss(model, loaders["train"], torch.device("cpu"), pad_id=PAD_ID)
    assert math.isfinite(loss)
    assert loss > 0.0


def test_evaluate_loss_restores_training_mode():
    cfg = tiny_cfg()
    model, loaders, _, _, _ = build_model_and_loader(cfg)
    model.train()
    evaluate_loss(model, loaders["val"], torch.device("cpu"), pad_id=PAD_ID)
    assert model.training is True


def test_greedy_translate_returns_one_string_per_input():
    cfg = tiny_cfg()
    model, _, pairs, src_tok, tgt_tok = build_model_and_loader(cfg)
    texts = [src for src, _ in pairs["val"][:5]]
    out = greedy_translate(
        model, texts, src_tok, tgt_tok, torch.device("cpu"),
        max_len=cfg.model.context_length,
    )
    assert len(out) == len(texts)
    assert all(isinstance(s, str) for s in out)


def test_plot_curve_writes_png(tmp_path: Path):
    """``plot_curve`` now lives in ``scripts/plot_curve.py``, not ``evaluate``."""
    from scripts.plot_curve import plot_curve

    # A minimal CSV with the columns the plotter expects.
    csv_path = tmp_path / "training_log.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "step", "epoch", "train_loss", "val_loss",
                "embed_b_norm", "bQ_norm", "bK_norm", "bV_norm",
            ],
        )
        writer.writeheader()
        for step in range(1, 6):
            writer.writerow(
                {
                    "step": step, "epoch": 0, "train_loss": 3.0 - 0.1 * step,
                    "val_loss": 3.1 - 0.1 * step, "embed_b_norm": 0.1,
                    "bQ_norm": 0.0, "bK_norm": 0.0, "bV_norm": 0.2,
                }
            )
    out = plot_curve(csv_path, tmp_path / "curve.png", title="probe")
    if out is None:
        pytest.skip("matplotlib unavailable")
    assert out.exists() and out.stat().st_size > 0


# --------------------------------------------------------------------------- #
# Training entry point: one real end-to-end run per optimizer, as a subprocess
# --------------------------------------------------------------------------- #
#: The tiny, offline flag set shared by both runs: the ``smoke`` preset on the
#: synthetic copy task, with a 256-pair training split so a full run costs a few
#: seconds of actual training.  ``--no_plot`` keeps matplotlib out of the path.
TRAIN_BASE_FLAGS = [
    "--model", "smoke",
    "--dataset_preset", "synthetic-copy",
    "--synthetic_train_size", "256",
    "--synthetic_val_size", "32",
    "--synthetic_test_size", "32",
    "--no_plot",
]

#: Generous: a run takes ~20 s here, and this only exists to stop a hang.
TRAIN_TIMEOUT = 600.0

#: The two optimizers, with the tiny copy-task shapes the old in-process tests
#: used (``synthetic_vocab=6``, ``synthetic_len=4``).
EGD_RUN_FLAGS = [
    "--max_steps", "100",
    "--valid_every_updates", "50",
    "--log_every", "20",
    "--egd_lr", "0.1",
    "--synthetic_vocab", "6",
    "--synthetic_len", "4",
    # A non-zero embedding bias and bQ + bV, so the logged bias norms are non-zero
    # (bK stays off, as it does by default).
    "--bias_preset", "b-gaussian",
    "--use_q_bias",
    "--use_v_bias",
]

ADAMW_RUN_FLAGS = [
    "--optimizer", "adamw",
    "--adam_lr", "3.0e-3",
    "--max_steps", "100",
    "--valid_every_updates", "0",
    "--log_every", "25",
    "--synthetic_vocab", "6",
    "--synthetic_len", "4",
]


def train_command(tmp_path: Path, name: str, flags) -> list:
    """The ``python main.py train ...`` command line for one subprocess run."""
    return [
        sys.executable, str(ROOT / "main.py"), "train",
        *TRAIN_BASE_FLAGS,
        "--log_dir", str(tmp_path),
        "--name", name,
        *flags,
    ]


def read_losses(run_dir: Path) -> list:
    """The per-step ``train_loss`` column of the run's two-column ``losses.csv``."""
    with (run_dir / "losses.csv").open(encoding="utf-8") as fh:
        return [
            float(row["loss"])
            for row in csv.DictReader(fh)
            if row.get("tag") == "train_loss" and row.get("loss")
        ]


def test_train_end_to_end_runs(tmp_path: Path):
    """Drive ``main.py train`` for real, once per optimizer, and check the artifacts.

    Both runs are launched **concurrently**: this repository's smoke run costs
    ~10 s of pure ``torch``/``wandb`` import before it does any work, so running
    the two back to back would double an already generous test budget.  The
    assertions preserve the old in-process ``train(cfg)`` tests:

    * the EGD run writes the run directory artifacts and a non-zero/non-finite-free
      bias report, and its four bias-norm CSV columns carry the right numbers;
    * the EGD run's loss goes down (``losses.csv``), i.e. EGD really trains;
    * the AdamW run's loss drops decisively on the copy task (the old 0.5x bound);
    * ``--no_plot`` leaves no ``training_curve.png`` behind.
    """
    egd_name, adamw_name = "egd_bias", "adamw_learns"
    commands = [
        (egd_name, train_command(tmp_path, egd_name, EGD_RUN_FLAGS)),
        (adamw_name, train_command(tmp_path, adamw_name, ADAMW_RUN_FLAGS)),
    ]
    procs = [
        (
            name,
            subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            ),
        )
        for name, cmd in commands
    ]

    logs = {}
    for name, proc in procs:
        try:
            out, _ = proc.communicate(timeout=TRAIN_TIMEOUT)
        except subprocess.TimeoutExpired:  # pragma: no cover - only on a hang
            proc.kill()
            out, _ = proc.communicate()
            raise AssertionError(f"main.py train ({name}) timed out:\n{out[-4000:]}")
        logs[name] = out
        assert proc.returncode == 0, (
            f"main.py train ({name}) exited with {proc.returncode}:\n{out[-6000:]}"
        )

    # ---------------- the EGD run: artifacts + logged bias norms ---------------- #
    run_dir = tmp_path / egd_name
    assert run_dir.is_dir()
    for name in (
        "args.json",
        "config.json",
        "bias.json",
        "tokenizer_src.json",
        "tokenizer_tgt.json",
        "training_log.csv",
        "losses.csv",
        "summary.json",
        "model_final.pt",
        "model_best.pt",
    ):
        assert (run_dir / name).exists(), f"missing artifact {name}"
    # ``--no_plot`` must leave no curve behind (the old plot=False case).
    assert not (run_dir / "training_curve.png").exists()

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_name"] == egd_name
    assert summary["steps"] == 100
    assert summary["parameters"] > 0
    assert summary["optimizer"] == "EGD"
    assert math.isfinite(summary["best_val_loss"])

    # The resolved configs are persisted with the preset's defaults filled in.
    assert json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["n_embd"] == 64
    assert json.loads((run_dir / "bias.json").read_text(encoding="utf-8"))["embed_mode"] == "gaussian"
    assert json.loads((run_dir / "args.json").read_text(encoding="utf-8"))["optimizer"] == "egd"

    with (run_dir / "training_log.csv").open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = set(reader.fieldnames or [])
        rows = list(reader)
    assert {"embed_b_norm", "bQ_norm", "bK_norm", "bV_norm"} <= header
    logged = [r for r in rows if r.get("embed_b_norm") not in ("", None)]
    assert logged, "no rows carried a bias norm"
    # A non-zero gaussian b must be logged as non-zero, bQ/bV too, and bK stays off.
    assert float(logged[-1]["embed_b_norm"]) > 0.0
    assert float(logged[-1]["bQ_norm"]) > 0.0
    assert float(logged[-1]["bK_norm"]) == 0.0
    assert float(logged[-1]["bV_norm"]) > 0.0

    # ------------------------------- EGD learns -------------------------------- #
    # Measured on this setup: 100 EGD steps (lr 0.1) take the copy-task loss from
    # ~2.30 to ~1.9, so 0.95x has margin (the old 300-step run asserted 0.85x).
    egd_losses = read_losses(run_dir)
    assert len(egd_losses) >= 2
    assert egd_losses[-1] < 0.95 * egd_losses[0], (
        f"EGD barely learned: {egd_losses[0]} -> {egd_losses[-1]}"
    )

    # ------------------------------- AdamW learns ------------------------------ #
    # The old in-process test measured AdamW (lr 3e-3, 250 steps) going from ~2.07
    # to ~0.10 and asserted a 0.5x bound; 100 steps of the same schedule reach
    # ~0.36x here, so the bound is unchanged and still fails loudly if the loss
    # plumbing, masking or label shifting breaks.
    adamw_dir = tmp_path / adamw_name
    adamw_summary = json.loads((adamw_dir / "summary.json").read_text(encoding="utf-8"))
    assert adamw_summary["steps"] == 100
    assert adamw_summary["optimizer"] == "AdamW"
    assert math.isfinite(adamw_summary["best_val_loss"])
    adamw_losses = read_losses(adamw_dir)
    assert len(adamw_losses) >= 2
    assert adamw_losses[-1] < 0.5 * adamw_losses[0], (
        f"AdamW barely learned: {adamw_losses[0]} -> {adamw_losses[-1]}"
    )


# --------------------------------------------------------------------------- #
# Run identity
# --------------------------------------------------------------------------- #
def test_run_name_encodes_bias_and_optimizer():
    """The generated run name must encode both the bias and the optimizer."""
    from scripts.train import bias_tag, run_name_for

    assert bias_tag(resolve_bias_config(None)) == "symmetric"
    name = run_name_for(
        "smoke", "egd", resolve_bias_config(None, embed_mode="gaussian"), 7
    )
    assert "egd" in name
    assert "gaussian" in name
    assert "seed7" in name


# --------------------------------------------------------------------------- #
# The three bias modes -- the reason this project exists
# --------------------------------------------------------------------------- #
def test_embedding_bias_zero_mode_is_exact_zero():
    from symbreak_transformer.bias import EmbeddingBias

    bias = EmbeddingBias(16, mode="zero")
    assert torch.equal(bias.b, torch.zeros(16))
    assert bias.is_zero and not bias.active
    # Adding it is an exact no-op, so the O(n_embd) symmetry is untouched.
    x = torch.randn(2, 5, 16)
    assert torch.equal(bias(x), x)


def test_embedding_bias_const_mode_is_isotropic():
    from symbreak_transformer.bias import EmbeddingBias

    bias = EmbeddingBias(16, mode="const", const_value=0.5)
    assert torch.allclose(bias.b, torch.full((16,), 0.5))
    # ||b|| = 0.5 * sqrt(16) = 2.0, a rank-one direction, hence symmetry breaking.
    assert float(bias.b.norm()) == pytest.approx(2.0, abs=1e-6)
    assert bias.active and not bias.is_zero


def test_embedding_bias_gaussian_is_seed_reproducible_and_nonzero():
    from symbreak_transformer.bias import EmbeddingBias

    first = EmbeddingBias(32, mode="gaussian", mean=0.0, std=0.02, seed=99).b
    second = EmbeddingBias(32, mode="gaussian", mean=0.0, std=0.02, seed=99).b
    assert torch.equal(first, second), "same seed must give the same b"
    other = EmbeddingBias(32, mode="gaussian", std=0.02, seed=100).b
    assert not torch.equal(first, other), "a different seed must give a different b"
    # ||b|| ~ std * sqrt(n_embd) = 0.02 * sqrt(32) = 0.113
    assert float(first.norm()) == pytest.approx(0.113, abs=0.05)
    assert not torch.equal(first, torch.zeros(32))


def test_attention_bias_sectors_are_independent_but_reproducible():
    from symbreak_transformer.bias import AttentionBias

    cfg = resolve_bias_config(
        None,
        attn_mode="gaussian",
        use_q_bias=True,
        use_k_bias=True,
        use_v_bias=True,
    )
    bias = AttentionBias(4, 8, cfg)
    q, k, v = bias.q.b, bias.k.b, bias.v.b
    # Distinct RNG streams: bQ/bK/bV must not be three copies of one vector.
    assert not torch.equal(q, v)
    assert not torch.equal(q, k)
    again = AttentionBias(4, 8, cfg)
    assert torch.equal(q, again.q.b) and torch.equal(v, again.v.b)
    # bK is off by default, so a q+v config yields exactly zero there.
    default = AttentionBias(
        4,
        8,
        resolve_bias_config(
            None, attn_mode="gaussian", use_q_bias=True, use_v_bias=True
        ),
    )
    assert torch.equal(default.k.b, torch.zeros(4, 8))
    assert float(default.q.b.norm()) > 0 and float(default.v.b.norm()) > 0


def test_bias_resample_per_step_only_moves_nonzero_modes():
    from symbreak_transformer.bias import EmbeddingBias

    gauss = EmbeddingBias(16, mode="gaussian", std=0.05, resample="per_step")
    before = gauss.b.clone()
    gauss.resample()
    assert not torch.equal(before, gauss.b), "per_step must redraw"

    fixed = EmbeddingBias(16, mode="gaussian", resample="fixed")
    before_fixed = fixed.b.clone()
    fixed.resample()
    assert torch.equal(before_fixed, fixed.b), "fixed must not redraw"

    zero = EmbeddingBias(16, mode="zero", resample="per_step")
    zero.resample()
    assert torch.equal(zero.b, torch.zeros(16)), "zero mode must stay exactly zero"
