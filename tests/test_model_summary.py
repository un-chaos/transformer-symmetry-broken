"""Tests for the model structure / dimension / parameter-count report.

The user's requirement was that a run must actually tell you what the model is.
The property that makes the report trustworthy is arithmetic: the per-component
rows must add up to the model's real parameter count, including the awkward
configurations (shared embeddings, a tied output projection, learnable biases,
prelu slopes).  Those invariants are asserted here for every variant.
"""

from __future__ import annotations

import pytest
import torch

from symbreak_transformer.config import resolve_bias_config, resolve_config
from symbreak_transformer.model import Seq2SeqTransformer
from symbreak_transformer.model_summary import (
    _merge_for_display,
    architecture_lines,
    buffer_rows,
    component_rows,
    model_summary,
    parameter_totals,
    write_model_summary,
)
from symbreak_transformer.utils import count_parameters

VOCAB = 400


def build(model: str = "smoke", bias: str = "b-gaussian", **overrides) -> Seq2SeqTransformer:
    cfg = resolve_config(
        model, src_vocab_size=VOCAB, tgt_vocab_size=VOCAB, **overrides
    )
    return Seq2SeqTransformer(
        cfg,
        resolve_bias_config(bias),
        src_vocab_size=cfg.src_vocab_size,
        tgt_vocab_size=cfg.tgt_vocab_size,
    )


# --------------------------------------------------------------------------- #
# The arithmetic invariant
# --------------------------------------------------------------------------- #
ALL_VARIANTS = [
    ("default (tied output)", {"model": "small"}),
    ("share_embeddings", {"model": "small", "share_embeddings": True}),
    ("untied output", {"model": "small", "tie_output_embedding": False}),
    ("prelu slopes", {"model": "small", "activation": "prelu", "prelu_random_init": True}),
    ("no scale_embedding", {"model": "small", "scale_embedding": False}),
    ("tiny", {"model": "smoke"}),
]


@pytest.mark.parametrize(("label", "kwargs"), ALL_VARIANTS, ids=[v[0] for v in ALL_VARIANTS])
def test_component_rows_sum_to_the_real_total(label, kwargs):
    model = build(**kwargs)
    totals = parameter_totals(model)
    rows = component_rows(model)

    assert sum(row["params"] for row in rows) == totals["total"], label
    assert totals["total"] == count_parameters(model, trainable_only=False), label
    assert totals["trainable"] == count_parameters(model, trainable_only=True), label
    assert totals["total"] == totals["trainable"] + totals["frozen"], label
    assert abs(sum(row["share"] for row in rows) - 1.0) < 1e-6, label


@pytest.mark.parametrize(
    ("label", "bias"),
    [
        ("learnable attention biases", "attn-learnable"),
        ("learnable embedding bias", "b-learnable"),
        ("symmetric control", "symmetric"),
    ],
)
def test_learnable_and_fixed_biases_are_counted_in_the_right_place(label, bias):
    model = build(model="smoke", bias=bias)
    totals = parameter_totals(model)
    assert sum(row["params"] for row in component_rows(model)) == totals["total"], label
    assert totals["trainable"] == count_parameters(model, trainable_only=True), label


def test_shared_embeddings_are_not_double_counted():
    """One embedding module used twice is one set of weights, not two."""
    shared = build(model="small", share_embeddings=True)
    separate = build(model="small", share_embeddings=False)
    shared_total = parameter_totals(shared)["total"]
    separate_total = parameter_totals(separate)["total"]
    # Sharing removes one embedding table (vocab * n_embd) plus one positional table.
    expected = separate_total - (VOCAB * shared.cfg.n_embd + shared.cfg.context_length * shared.cfg.n_embd)
    assert shared_total == expected


def test_tied_output_projection_has_no_parameters_of_its_own():
    model = build(model="small", tie_output_embedding=True)
    rows = {row["name"]: row for row in component_rows(model)}
    assert rows["output projection"]["params"] == 0
    assert "tied" in rows["output projection"]["dims"].lower() or "shared" in rows["output projection"]["dims"].lower()

    untied = build(model="small", tie_output_embedding=False)
    untied_rows = {row["name"]: row for row in component_rows(untied)}
    # weight (vocab x n_embd) plus the bias (vocab)
    assert untied_rows["output projection"]["params"] == VOCAB * untied.cfg.n_embd + VOCAB


# --------------------------------------------------------------------------- #
# Buffers vs parameters
# --------------------------------------------------------------------------- #
def test_a_fixed_embedding_bias_is_a_buffer_not_a_parameter():
    model = build(model="smoke", bias="b-gaussian")
    totals = parameter_totals(model)
    assert totals["buffers"] == model.cfg.n_embd  # exactly one b vector
    names = [row["name"] for row in buffer_rows(model)]
    assert any(name.endswith("bias.b") or name.endswith(".b") for name in names), names


def test_a_learnable_bias_moves_from_buffers_to_parameters():
    fixed = build(model="smoke", bias="b-gaussian")
    learnable = build(model="smoke", bias="b-learnable")
    assert parameter_totals(learnable)["buffers"] == 0
    assert parameter_totals(fixed)["buffers"] > 0
    assert parameter_totals(learnable)["trainable"] > parameter_totals(fixed)["trainable"]


def test_structural_buffers_are_hidden():
    """The causal mask is noise, not an experiment setting."""
    model = build(model="smoke")
    assert not any("causal" in row["name"] for row in buffer_rows(model))


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def test_architecture_lines_name_the_dimensions():
    model = build(model="small")
    text = "\n".join(architecture_lines(model))
    cfg = model.cfg
    for expected in (
        str(cfg.n_embd),
        str(cfg.n_head),
        str(cfg.n_embd // cfg.n_head),
        str(cfg.n_encoder_layer),
        str(cfg.n_decoder_layer),
        str(cfg.d_ff),
        str(cfg.context_length),
        str(VOCAB),
        "head_dim",
        "symmetry-breaking",
    ):
        assert expected in text, f"{expected!r} missing from {text}"


def test_summary_is_ascii_and_mentions_the_total():
    model = build(model="small")
    text = model_summary(model, width=100)
    assert text == text.strip(), "the summary must not start or end with blank lines"
    assert all(line.isascii() for line in text.splitlines()), "GBK consoles need ASCII"
    total = parameter_totals(model)["total"]
    assert f"{total:,}" in text
    assert "Components" in text and "Where the parameters live" in text
    # Every component name appears, so nothing is silently omitted.
    for row in component_rows(model):
        stem = row["name"].split(" ")[0]
        assert stem in text


def test_summary_mentions_the_bias_configuration():
    model = build(model="small", bias="attn-bQbV")
    assert "attention bQV" in model_summary(model) or "bQV" in model_summary(model)


def test_layer_rows_are_merged_for_display():
    """Six identical layers must read as one line, but still sum correctly."""
    model = build(model="small")
    rows = component_rows(model)
    merged = _merge_for_display(rows)
    assert len(merged) < len(rows), "identical consecutive layers were not merged"
    assert sum(row["params"] for row in merged) == sum(row["params"] for row in rows)
    assert any("(x" in row["name"] for row in merged)


def test_write_matches_the_rendered_string(tmp_path):
    model = build(model="smoke")
    path = write_model_summary(model, tmp_path / "model_summary.txt", width=100)
    assert path.exists()
    assert path.read_text(encoding="utf-8") == model_summary(model, width=100)


def test_summary_renders_for_a_one_layer_model():
    model = build(
        model="smoke", n_encoder_layer=1, n_decoder_layer=1, n_embd=32, n_head=4, d_ff=64
    )
    text = model_summary(model)
    assert "encoder layer 0" in text
    assert "decoder layer 0" in text
    assert parameter_totals(model)["total"] == sum(
        row["params"] for row in component_rows(model)
    )


def test_component_rows_have_the_documented_keys():
    model = build(model="smoke")
    for row in component_rows(model):
        assert set(row) >= {"name", "dims", "params", "trainable", "share"}
        assert isinstance(row["name"], str) and row["name"]
        assert isinstance(row["dims"], str) and row["dims"]
        assert isinstance(row["params"], int) and row["params"] >= 0
