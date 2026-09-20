"""Tests for the integration layer: BLEU, evaluation, and the training loop.

These cover the modules no subagent owned (``evaluate.py``, ``train.py``,
``main.py``).  The BLEU expectations are hand-computed, not generated from the
implementation:

* ``perfect``   : p1=p2=p3=p4=1, lengths equal -> BP=1, BLEU=100
* ``partial``   : p1=1/2, p2=1/3, p3=0, p4=0 with equal lengths and add-1
                  smoothing -> 100 * (1/2 * 1/3 * 1/3 * 1/2)^(1/4) = 40.8248
* ``short_hyp`` : all four precisions 1 but hyp_len=4, ref_len=6
                  -> BP = exp(1 - 3/2) = 0.606531, BLEU = 60.6531
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest
import torch

from transformer_sym.config import config_from_dict
from transformer_sym.data.tokenizer import PAD_ID, Tokenizer
from transformer_sym.evaluate import corpus_bleu, evaluate_loss, greedy_translate, plot_curve
from transformer_sym.train import auto_run_name, train

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
def tiny_cfg(**overrides):
    """A small, fast config on the built-in synthetic task."""
    raw = {
        "model": {
            "d_model": 32,
            "n_heads": 2,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "d_ff": 64,
            "max_seq_len": 12,
            "dropout": 0.0,
            "attention_dropout": 0.0,
        },
        "bias": {"embed": {"mode": "zero"}, "attention": {"enabled": False}},
        "data": {
            "source": "synthetic",
            "synthetic_task": "reverse",
            "synthetic_train_size": 256,
            "synthetic_val_size": 64,
            "synthetic_test_size": 64,
            "synthetic_vocab": 12,
            "synthetic_len": 6,
            "min_freq": 1,
            "max_vocab": 64,
        },
        "train": {
            "optimizer": "egd",
            "batch_size": 32,
            "epochs": 1,
            "max_steps": 12,
            "log_every": 4,
            "eval_every": 6,
            "max_eval_batches": 2,
            "seed": 7,
            "device": "cpu",
        },
    }
    for dotted, value in overrides.items():
        target = raw
        parts = dotted.split("__")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return config_from_dict(raw)


def build_model_and_loader(cfg, pairs_split: str = "train", batch_size: int = 32):
    """Build a model plus a loader through the real pipeline."""
    from transformer_sym.data.dataset import build_dataloaders
    from transformer_sym.data.download import load_all_splits
    from transformer_sym.data.tokenizer import build_tokenizers
    from transformer_sym.model import Seq2SeqTransformer

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
    cfg.train.batch_size = batch_size
    loaders = build_dataloaders(cfg, src_tok, tgt_tok, splits=("train", "val"))
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
        model, texts, src_tok, tgt_tok, torch.device("cpu"), max_len=cfg.model.max_seq_len
    )
    assert len(out) == len(texts)
    assert all(isinstance(s, str) for s in out)


def test_plot_curve_writes_png(tmp_path: Path):
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
# Training loop
# --------------------------------------------------------------------------- #
def test_auto_run_name_encodes_bias_and_optimizer():
    cfg = tiny_cfg(**{"bias__embed__mode": "gaussian"})
    name = auto_run_name(cfg)
    assert "embgaussian" in name
    assert name.startswith("egd")


def test_train_writes_expected_artifacts(tmp_path: Path):
    cfg = tiny_cfg(**{"train__save_dir": str(tmp_path), "train__run_name": "artifacts"})
    result = train(cfg, plot=False)

    run_dir = Path(result["run_dir"])
    assert run_dir == tmp_path / "artifacts"
    for name in (
        "config.yaml",
        "tokenizer_src.json",
        "tokenizer_tgt.json",
        "training_log.csv",
        "checkpoint_final.pt",
        "summary.json",
    ):
        assert (run_dir / name).exists(), f"missing artifact {name}"

    assert result["steps"] == 12
    assert math.isfinite(result["best_val_loss"])
    assert result["optimizer"] == "EGD"

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["run_name"] == "artifacts"
    assert summary["parameters"] > 0

    # The resolved config is persisted with defaults filled in.
    saved = (run_dir / "config.yaml").read_text(encoding="utf-8")
    assert "optimizer: egd" in saved


def test_train_logs_bias_norms_for_nonzero_bias(tmp_path: Path):
    cfg = tiny_cfg(
        **{
            "train__save_dir": str(tmp_path),
            "train__run_name": "biaslog",
            "bias__embed__mode": "gaussian",
            "bias__embed__std": 0.05,
            "bias__attention__enabled": True,
            "bias__attention__mode": "gaussian",
        }
    )
    result = train(cfg, plot=False)
    rows = list(
        csv.DictReader((Path(result["run_dir"]) / "training_log.csv").open(encoding="utf-8"))
    )
    assert rows, "training log is empty"
    logged = [r for r in rows if r.get("embed_b_norm") not in ("", None)]
    assert logged, "no rows carried a bias norm"
    # A non-zero gaussian b must be logged as non-zero, and bK stays off.
    assert float(logged[-1]["embed_b_norm"]) > 0.0
    assert float(logged[-1]["bQ_norm"]) > 0.0
    assert float(logged[-1]["bK_norm"]) == 0.0
    assert float(logged[-1]["bV_norm"]) > 0.0


def test_train_decreases_loss_with_egd(tmp_path: Path):
    """EGD must actually reduce the loss end to end (not just run).

    Threshold from a measured sweep on this exact setup: EGD reaches about
    0.71-0.74x the initial loss in 300 steps (lr 0.05-0.2), so 0.85 has margin.
    """
    cfg = tiny_cfg(
        **{
            "train__max_steps": 300,
            "train__run_name": "egd_learns",
            "train__save_dir": str(tmp_path),
            "train__log_every": 25,
            "train__egd__lr": 0.1,
            "data__synthetic_task": "copy",
            "data__synthetic_vocab": 6,
            "data__synthetic_len": 4,
            "model__d_model": 64,
            "model__d_ff": 128,
        }
    )
    result = train(cfg, plot=False)
    history = list(
        csv.DictReader(
            (Path(result["run_dir"]) / "training_log.csv").open(encoding="utf-8")
        )
    )
    losses = [float(r["train_loss"]) for r in history if r.get("train_loss")]
    assert len(losses) >= 2
    initial, final = losses[0], losses[-1]
    assert final < 0.85 * initial, f"EGD barely learned: {initial} -> {final}"
    assert math.isfinite(result["best_val_loss"])


def test_train_decreases_loss_with_adamw(tmp_path: Path):
    """The baseline optimizer must learn the copy task decisively.

    Measured on this setup: AdamW (lr 3e-3, 250 steps) drives the loss from
    ~2.07 to ~0.10, so 0.5x is a generous bound that still fails loudly if the
    loss plumbing, masking or label shifting breaks.
    """
    cfg = tiny_cfg(
        **{
            "train__optimizer": "adamw",
            "train__max_steps": 250,
            "train__run_name": "adamw_learns",
            "train__save_dir": str(tmp_path),
            "train__log_every": 25,
            "train__adamw__lr": 3.0e-3,
            "data__synthetic_task": "copy",
            "data__synthetic_vocab": 6,
            "data__synthetic_len": 4,
        }
    )
    result = train(cfg, plot=False)
    history = list(
        csv.DictReader(
            (Path(result["run_dir"]) / "training_log.csv").open(encoding="utf-8")
        )
    )
    losses = [float(r["train_loss"]) for r in history if r.get("train_loss")]
    assert losses[-1] < 0.5 * losses[0], f"AdamW barely learned: {losses[0]} -> {losses[-1]}"
    assert result["optimizer"] == "AdamW"


def test_train_max_seconds_stops_early(tmp_path: Path):
    cfg = tiny_cfg(
        **{
            "train__max_steps": 100000,
            "train__epochs": 1000,
            "train__max_seconds": 0.0,
            "train__run_name": "stopper",
            "train__save_dir": str(tmp_path),
        }
    )
    cfg.train.max_seconds = 3.0
    result = train(cfg, plot=False)
    assert result["steps"] < 100000
    assert "max_seconds" in result["stop_reason"] or result["stop_reason"] == "completed"


def test_train_resume_continues_from_checkpoint(tmp_path: Path):
    cfg = tiny_cfg(
        **{
            "train__max_steps": 6,
            "train__run_name": "resume",
            "train__save_dir": str(tmp_path),
        }
    )
    first = train(cfg, plot=False)
    ckpt = Path(first["run_dir"]) / "checkpoint_final.pt"

    cfg2 = tiny_cfg(
        **{
            "train__max_steps": 12,
            "train__run_name": "resume2",
            "train__save_dir": str(tmp_path),
        }
    )
    second = train(cfg2, resume=str(ckpt), plot=False)
    assert second["steps"] == 12, "resume did not advance to the requested step"


def test_plot_disabled_leaves_no_png(tmp_path: Path):
    cfg = tiny_cfg(**{"train__save_dir": str(tmp_path), "train__run_name": "noplot"})
    result = train(cfg, plot=False)
    assert not (Path(result["run_dir"]) / "training_curve.png").exists()


# --------------------------------------------------------------------------- #
# The three bias modes -- the reason this project exists
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("mode", "expect_nonzero"),
    [("zero", False), ("gaussian", True), ("const", True)],
)
def test_embedding_bias_mode_round_trip(tmp_path: Path, mode: str, expect_nonzero: bool):
    """Each of the three modes must train end to end with the expected b."""
    cfg = tiny_cfg(
        **{
            "train__save_dir": str(tmp_path),
            "train__run_name": f"mode_{mode}",
            "train__max_steps": 8,
            "train__log_every": 4,
            "train__eval_every": 8,
            "bias__embed__mode": mode,
            "bias__embed__const_value": 0.5,
        }
    )
    result = train(cfg, plot=False)
    report = result["bias_report"]
    norm = float(report["embed.bias_norm"])
    assert math.isfinite(norm)
    if expect_nonzero:
        assert norm > 0.0, f"{mode} mode produced a zero bias"
    else:
        assert norm == 0.0, f"zero mode produced a non-zero bias ({norm})"
    # The reported mode always matches what was asked for.
    assert str(report["embed.bias_mode"]) == mode


def test_embedding_bias_zero_mode_is_exact_zero():
    from transformer_sym.bias import EmbeddingBias
    from transformer_sym.config import EmbeddingBiasConfig

    bias = EmbeddingBias(16, EmbeddingBiasConfig(mode="zero"))
    assert torch.equal(bias.b, torch.zeros(16))
    assert bias.is_zero and not bias.active
    # Adding it is an exact no-op, so the O(d_model) symmetry is untouched.
    x = torch.randn(2, 5, 16)
    assert torch.equal(bias(x), x)


def test_embedding_bias_const_mode_is_isotropic():
    from transformer_sym.bias import EmbeddingBias
    from transformer_sym.config import EmbeddingBiasConfig

    bias = EmbeddingBias(16, EmbeddingBiasConfig(mode="const", const_value=0.5))
    assert torch.allclose(bias.b, torch.full((16,), 0.5))
    # ||b|| = 0.5 * sqrt(16) = 2.0, a rank-one direction, hence symmetry breaking.
    assert float(bias.b.norm()) == pytest.approx(2.0, abs=1e-6)
    assert bias.active and not bias.is_zero


def test_embedding_bias_gaussian_is_seed_reproducible_and_nonzero():
    from transformer_sym.bias import EmbeddingBias
    from transformer_sym.config import EmbeddingBiasConfig

    cfg = EmbeddingBiasConfig(mode="gaussian", mean=0.0, std=0.02, seed=99)
    first = EmbeddingBias(32, cfg).b
    second = EmbeddingBias(32, cfg).b
    assert torch.equal(first, second), "same seed must give the same b"
    other = EmbeddingBias(32, EmbeddingBiasConfig(mode="gaussian", std=0.02, seed=100)).b
    assert not torch.equal(first, other), "a different seed must give a different b"
    # ||b|| ~ std * sqrt(d_model) = 0.02 * sqrt(32) = 0.113
    assert float(first.norm()) == pytest.approx(0.113, abs=0.05)
    assert not torch.equal(first, torch.zeros(32))


def test_attention_bias_sectors_are_independent_but_reproducible():
    from transformer_sym.bias import AttentionBias
    from transformer_sym.config import AttentionBiasConfig

    cfg = AttentionBiasConfig(
        enabled=True, mode="gaussian", q_enabled=True, k_enabled=True, v_enabled=True
    )
    bias = AttentionBias(4, 8, cfg)
    q, k, v = bias.q.b, bias.k.b, bias.v.b
    # Distinct RNG streams: bQ/bK/bV must not be three copies of one vector.
    assert not torch.equal(q, v)
    assert not torch.equal(q, k)
    again = AttentionBias(4, 8, cfg)
    assert torch.equal(q, again.q.b) and torch.equal(v, again.v.b)
    # bK is off by default, so the default config yields exactly zero there.
    default = AttentionBias(4, 8, AttentionBiasConfig(enabled=True, mode="gaussian"))
    assert torch.equal(default.k.b, torch.zeros(4, 8))
    assert float(default.q.b.norm()) > 0 and float(default.v.b.norm()) > 0


def test_bias_resample_per_step_only_moves_nonzero_modes():
    from transformer_sym.bias import EmbeddingBias
    from transformer_sym.config import EmbeddingBiasConfig

    gauss = EmbeddingBias(
        16, EmbeddingBiasConfig(mode="gaussian", std=0.05, resample="per_step")
    )
    before = gauss.b.clone()
    gauss.resample()
    assert not torch.equal(before, gauss.b), "per_step must redraw"

    fixed = EmbeddingBias(16, EmbeddingBiasConfig(mode="gaussian", resample="fixed"))
    before_fixed = fixed.b.clone()
    fixed.resample()
    assert torch.equal(before_fixed, fixed.b), "fixed must not redraw"

    zero = EmbeddingBias(16, EmbeddingBiasConfig(mode="zero", resample="per_step"))
    zero.resample()
    assert torch.equal(zero.b, torch.zeros(16)), "zero mode must stay exactly zero"
