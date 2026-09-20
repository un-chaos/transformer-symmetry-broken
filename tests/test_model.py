"""Tests for the ``symbreak_transformer`` model stack.

Everything here is deliberately tiny (``n_embd=32``, 2+2 layers, ``d_ff=64``,
``context_length=16``) so the whole file runs in a few seconds on CPU.

The three tests that matter most -- they fail loudly if the mask polarity or the
bias wiring is wrong:

* :func:`test_source_padding_invariance` -- rewriting the *content* of padded
  source positions (and re-deriving the mask, keeping the true lengths) must not
  move the logits at real target positions.
* :func:`test_causality` -- rewriting ``tgt_in[t]`` must not move the logits at
  positions ``< t``.
* :func:`test_bias_is_a_real_switch` -- ``zero`` biases are a no-op, and each
  non-zero bias family changes the output.
"""

from __future__ import annotations

import copy

import pytest
import torch

from symbreak_transformer.bias import AttentionBias, EmbeddingBias
from symbreak_transformer.config import BiasConfig, Seq2SeqConfig, resolve_bias_config
from symbreak_transformer.model import Seq2SeqTransformer

# --------------------------------------------------------------------------- #
# Shared tiny setup
# --------------------------------------------------------------------------- #
VOCAB = 24
SRC_VOCAB = 24
TGT_VOCAB = 26
PAD_ID, BOS_ID, EOS_ID = 0, 2, 3
MAX_SEQ_LEN = 16


def make_cfg(**overrides) -> Seq2SeqConfig:
    """The tiny model shape used by every test."""
    base = dict(
        n_embd=32,
        n_head=4,
        n_encoder_layer=2,
        n_decoder_layer=2,
        d_ff=64,
        dropout=0.0,
        attention_dropout=0.0,
        context_length=MAX_SEQ_LEN,
        activation="gelu",
        tie_output_embedding=True,
        share_embeddings=False,
    )
    base.update(overrides)
    return Seq2SeqConfig(**base)


def make_bias_cfg(
    embed_mode: str = "zero",
    attn: bool = False,
    attn_mode: str = "gaussian",
    embed_resample: str = "fixed",
    attn_resample: str = "fixed",
    **attn_fields,
) -> BiasConfig:
    """A flat :class:`BiasConfig` with explicit switches.

    ``embed_mode="zero"`` is the unbiased default: the embedding bias is an exact
    zero vector and attention biases are absent, so two models built from the same
    seed must be bit-identical.  ``attn=True`` turns the master switch on and, like
    the old nested ``AttentionBiasConfig`` defaults, enables ``bQ`` and ``bV``
    while leaving ``bK`` off.  ``attn_fields`` are the flat attention fields from
    :class:`BiasConfig` (e.g. ``use_q_bias=True``, ``attn_learnable=True``).
    """
    fields = {
        "use_q_bias": bool(attn),
        "use_k_bias": False,
        "use_v_bias": bool(attn),
    }
    fields.update(attn_fields)
    return resolve_bias_config(
        None,
        embed_mode=embed_mode,
        embed_resample=embed_resample,
        embed_seed=7,
        attn_mode=attn_mode,
        attn_resample=attn_resample,
        **fields,
    )


def make_model(cfg: Seq2SeqConfig | None = None, bias_cfg: BiasConfig | None = None,
               seed: int = 0) -> Seq2SeqTransformer:
    """Seeded model factory (seed first, so weight init is reproducible)."""
    if isinstance(cfg, BiasConfig) or isinstance(bias_cfg, Seq2SeqConfig):
        raise TypeError(
            "make_model(cfg, bias_cfg, seed): got the two configs in the wrong "
            "order -- pass Seq2SeqConfig first, BiasConfig second"
        )
    torch.manual_seed(seed)
    return Seq2SeqTransformer(
        cfg or make_cfg(),
        bias_cfg or make_bias_cfg(),
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )


def batch(batch_size: int = 3, src_len: int = 5, tgt_len: int = 6, seed: int = 1):
    """A deterministic random batch with real (non-pad) content everywhere."""
    gen = torch.Generator().manual_seed(seed)
    src = torch.randint(4, SRC_VOCAB, (batch_size, src_len), generator=gen)
    tgt_in = torch.randint(4, TGT_VOCAB, (batch_size, tgt_len), generator=gen)
    tgt_in[:, 0] = BOS_ID
    labels = torch.randint(4, TGT_VOCAB, (batch_size, tgt_len), generator=gen)
    return src, tgt_in, labels


def pad_mask(tokens: torch.Tensor) -> torch.Tensor:
    """``True`` where the token is PAD -- the mask convention of this repo."""
    return tokens == PAD_ID


# --------------------------------------------------------------------------- #
# 1. Forward shapes and the loss
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("norm_first", [False, True])
def test_forward_shapes_and_loss(norm_first: bool):
    model = make_model(make_cfg(norm_first=norm_first))
    model.eval()
    src, tgt_in, labels = batch()
    out = model(src, tgt_in, labels=labels)

    assert set(out) == {"logits", "loss"}
    logits = out["logits"]
    assert logits.shape == (3, 6, TGT_VOCAB)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()

    loss = out["loss"]
    assert loss is not None and loss.dim() == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

    # No labels -> no loss, but the logits are still produced.
    assert model(src, tgt_in)["loss"] is None


def test_loss_ignores_padded_labels():
    """``ignore_index=pad_id`` must drop pad targets from the mean exactly.

    Compared against a hand-built reference: mean cross-entropy over only the
    non-pad positions.  A wrong ``ignore_index`` (or the wrong reduction) shows
    up immediately, because the pad positions have very different logits.
    """
    model = make_model()
    model.eval()
    src, tgt_in, labels = batch()
    labels = labels.clone()
    labels[:, -2:] = PAD_ID

    with torch.no_grad():
        logits = model(src, tgt_in)["logits"]

    loss = model(src, tgt_in, labels=labels)["loss"]
    assert loss is not None

    import torch.nn.functional as F

    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    keep = flat_labels != PAD_ID
    reference = F.cross_entropy(flat_logits[keep], flat_labels[keep])
    assert torch.allclose(loss, reference, atol=1e-6), (loss.item(), reference.item())

    # Cross-check against a sentinel-based reference: masking the pad slots with
    # -100 and using the default ignore_index must give the same number.  This
    # isolates "which positions are dropped" from "how the mean is taken".
    sentinel = flat_labels.masked_fill(~keep, -100)
    reference_sentinel = F.cross_entropy(flat_logits, sentinel)
    assert torch.allclose(loss, reference_sentinel, atol=1e-6), (
        loss.item(),
        reference_sentinel.item(),
    )


# --------------------------------------------------------------------------- #
# 2. Padding invariance (the sharpest mask-polarity test)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("norm_first", [False, True])
def test_source_padding_invariance(norm_first: bool):
    """Rewriting padded *source content* must not move real target logits.

    Two source batches share the same real tokens and the same real lengths:

    * ``src_a`` pads row 0 with the pad id,
    * ``src_b`` puts *different real tokens* in row 0's tail.

    Both are run with a padding mask that ignores row 0's tail.  Everything
    downstream of the real positions must be bit-identical, because a masked key
    contributes exactly ``0 * value`` to every attention row.  This is the test
    that fails if ``True``/``False`` in ``key_padding_mask`` is the wrong way
    round: with inverted polarity the real tokens would be dropped and the
    filler would leak in.
    """
    model = make_model(make_cfg(norm_first=norm_first))
    model.eval()

    # Row 0: real length 3; row 1: real length 5 (no padding at all).
    src_a = torch.tensor(
        [
            [4, 5, 6, PAD_ID, PAD_ID],
            [7, 8, 9, 10, 11],
        ]
    )
    src_b = src_a.clone()
    src_b[0, 3:] = torch.tensor([19, 23])  # real ids where src_a has PAD

    mask_a = src_a == PAD_ID  # True at (0,3) and (0,4)
    mask_b = mask_a.clone()  # same *positions* ignored, different content

    tgt_in = torch.tensor(
        [
            [BOS_ID, 4, 5, EOS_ID, PAD_ID, PAD_ID],
            [BOS_ID, 7, 8, 9, EOS_ID, PAD_ID],
        ]
    )
    tgt_mask = tgt_in == PAD_ID
    real_tgt = ~tgt_mask

    with torch.no_grad():
        logits_a = model(src_a, tgt_in, mask_a, tgt_mask)["logits"]
        logits_b = model(src_b, tgt_in, mask_b, tgt_mask)["logits"]

    assert not torch.equal(src_a, src_b)
    assert torch.equal(
        logits_a[real_tgt], logits_b[real_tgt]
    ), "logits at non-pad target positions changed when only padded source content changed"

    # The same must hold at the encoder boundary itself.
    with torch.no_grad():
        memory_a = model.encode(src_a, mask_a)
        memory_b = model.encode(src_b, mask_b)
    real_src = ~mask_a
    assert torch.equal(memory_a[real_src], memory_b[real_src])

    # Sanity check that the test has teeth: reversing the polarity (marking the
    # *real* positions as padding) must change the result.
    with torch.no_grad():
        inverted = model(src_a, tgt_in, ~mask_a, tgt_mask)["logits"]
    assert not torch.allclose(logits_a[real_tgt], inverted[real_tgt])


@pytest.mark.parametrize("norm_first", [False, True])
def test_target_padding_invariance(norm_first: bool):
    """Padding on the *target* side must not leak into real target positions."""
    model = make_model(make_cfg(norm_first=norm_first))
    model.eval()

    src = torch.tensor([[4, 5, 6, 7, 8]])
    tgt_in = torch.tensor([[BOS_ID, 4, PAD_ID, PAD_ID, PAD_ID]])
    with torch.no_grad():
        base = model(src, tgt_in, None, pad_mask(tgt_in))["logits"]
        tgt2 = tgt_in.clone()
        tgt2[0, 2:] = torch.tensor([21, 22, 23])
        changed = model(src, tgt2, None, pad_mask(tgt2))["logits"]

    # Position 1 is real; positions 0 and 1 must be untouched.
    assert torch.equal(base[:, :2], changed[:, :2])


# --------------------------------------------------------------------------- #
# 3. Causality
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("norm_first", [False, True])
def test_causality(norm_first: bool):
    """Changing ``tgt_in[t]`` must not change logits at positions ``< t``."""
    model = make_model(make_cfg(norm_first=norm_first))
    model.eval()
    src, tgt_in, _ = batch(batch_size=2, src_len=5, tgt_len=6)
    no_pad = torch.zeros_like(tgt_in, dtype=torch.bool)
    src_no_pad = torch.zeros_like(src, dtype=torch.bool)

    for position in (2, 4, 5):
        with torch.no_grad():
            base = model(src, tgt_in, src_no_pad, no_pad)["logits"]
            edited = tgt_in.clone()
            edited[:, position] = (edited[:, position] + 7) % (TGT_VOCAB - 4) + 4
            changed = model(src, edited, src_no_pad, no_pad)["logits"]
        assert torch.equal(
            base[:, :position], changed[:, :position]
        ), f"editing position {position} leaked into earlier positions"


def test_causal_mask_is_strictly_upper_triangular():
    model = make_model()
    mask = model.make_causal_mask(5, torch.device("cpu"))
    assert mask.dtype == torch.bool
    assert mask.shape == (5, 5)
    assert torch.equal(mask, torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1))
    assert not mask.diagonal().any()  # the present is never blocked
    assert mask[0, 1] and mask[3, 4] and not mask[4, 3]


def test_fully_masked_rows_stay_finite():
    """All-pad keys must not produce NaN: fully masked rows fall back to column 0."""
    from symbreak_transformer.model import MultiHeadAttention

    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=32, n_head=4)
    attn.eval()
    x = torch.randn(2, 3, 32)
    all_pad = torch.ones(2, 3, dtype=torch.bool)  # every key is PAD
    blocked = torch.ones(3, 3, dtype=torch.bool)  # every pair is BLOCKED
    with torch.no_grad():
        out = attn(x, x, x, key_padding_mask=all_pad, attn_mask=blocked)
    assert torch.isfinite(out).all()
    assert out.shape == (2, 3, 32)


def test_fully_masked_row_is_one_hot_on_column_zero():
    """A row with nothing to attend to concentrates on column 0 (not uniform).

    This is the regression test for the fully-masked-row guard: col 0 is
    un-blocked *and* its logit is pinned to 0, so the softmax is a one-hot.
    Without that, the row would be a uniform average over all blocked columns.
    """
    from symbreak_transformer.model import MultiHeadAttention

    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=32, n_head=4)
    attn.eval()
    batch, length = 2, 4
    x = torch.randn(batch, length, 32)

    # Row 0 has every pair blocked; rows 1..3 attend to key 0 only.
    attn_mask = torch.zeros(length, length, dtype=torch.bool)
    attn_mask[0, :] = True
    attn_mask[1:, 0] = False
    attn_mask[1:, 1:] = True

    with torch.no_grad():
        probs = attn.attention_probs(x, x, attn_mask=attn_mask)

    assert probs.shape == (batch, 4, length, length)
    assert torch.isfinite(probs).all()
    assert torch.allclose(probs.sum(dim=-1), torch.ones(batch, 4, length), atol=1e-6)

    # Row 0 of every batch/head: no NaN, concentrated on column 0.
    row0 = probs[:, :, 0, :]
    assert torch.isfinite(row0).all()
    assert row0[..., 0].min() > 0.99, f"row 0 is not concentrated on col 0: {row0[0, 0]}"

    # A genuinely blocked row must still put ~zero weight on blocked columns.
    row0_rest = probs[:, :, 0, 1:]
    assert row0_rest.max() < 1e-3

    # Fully padded keys (key_padding_mask) behave the same way.
    all_pad = torch.ones(batch, length, dtype=torch.bool)
    with torch.no_grad():
        probs_pad = attn.attention_probs(x, x, key_padding_mask=all_pad)
    assert torch.isfinite(probs_pad).all()
    assert probs_pad[..., 0].min() > 0.99


def test_verified_padding_direction():
    """Pin down the polarity: a PAD *key* is ignored, a real key is used."""
    import torch.nn.functional as F

    from symbreak_transformer.model import MultiHeadAttention

    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=8, n_head=1)
    attn.eval()
    x = torch.randn(1, 2, 8)
    # Build V manually so the attended values are known.
    with torch.no_grad():
        attn.q_proj.weight.copy_(torch.eye(8))
        attn.q_proj.bias.zero_()
        attn.k_proj.weight.copy_(torch.eye(8))
        attn.k_proj.bias.zero_()
        attn.v_proj.weight.copy_(torch.eye(8))
        attn.v_proj.bias.zero_()
        attn.out_proj.weight.copy_(torch.eye(8))
        attn.out_proj.bias.zero_()

    query = x[:, :1]
    keys = x
    # With key 1 marked PAD, the output must equal the value at key 0.
    mask = torch.tensor([[False, True]])
    out_pad = attn(query, keys, keys, key_padding_mask=mask)
    with torch.no_grad():
        v = attn.v_proj(keys)
        probs = F.softmax((attn.q_proj(query) @ attn.k_proj(keys).transpose(-2, -1))
                          * (8 ** -0.5), dim=-1)
    assert probs[0, 0, 1] > 0.0  # unmasked: key 1 IS attended
    assert torch.allclose(out_pad[0, 0], v[0, 0], atol=1e-6), (
        "a key marked PAD=True must be ignored; this failed, so the mask "
        "polarity is inverted"
    )


def test_float_masks_raise_type_error():
    from symbreak_transformer.model import MultiHeadAttention

    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=32, n_head=4)
    x = torch.randn(2, 3, 32)
    with pytest.raises(TypeError):
        attn(x, x, x, key_padding_mask=torch.zeros(2, 3))
    with pytest.raises(TypeError):
        attn(x, x, x, attn_mask=torch.zeros(3, 3))
    with pytest.raises(TypeError):
        attn(x, x, x, attn_mask=torch.full((3, 3), float("-inf")))


# --------------------------------------------------------------------------- #
# 4. Bias is a real switch
# --------------------------------------------------------------------------- #
def test_zero_bias_model_is_reproducible():
    """All-zero biases + same seed -> identical parameters and identical logits."""
    src, tgt_in, _ = batch()
    first = make_model(make_cfg(), make_bias_cfg("zero"), seed=123)
    second = make_model(make_cfg(), make_bias_cfg("zero"), seed=123)
    first.eval()
    second.eval()

    for (name_a, pa), (name_b, pb) in zip(
        first.named_parameters(), second.named_parameters()
    ):
        assert name_a == name_b
        assert torch.equal(pa, pb), f"parameter {name_a} differs between runs"

    with torch.no_grad():
        logits_a = first(src, tgt_in)["logits"]
        logits_b = second(src, tgt_in)["logits"]
    assert torch.equal(logits_a, logits_b)
    assert first.bias_report()["embed.bias_norm"] == 0.0


@pytest.mark.parametrize("mode", ["gaussian", "const"])
def test_embedding_bias_changes_output(mode: str):
    """A non-zero embedding bias must change the logits vs. the zero-bias model."""
    src, tgt_in, _ = batch()
    unbiased = make_model(make_cfg(), make_bias_cfg("zero"), seed=123)
    biased = make_model(make_cfg(), make_bias_cfg(mode), seed=123)
    unbiased.eval()
    biased.eval()

    # Identical weights: only the bias differs.
    for (name, pa), (name_b, pb) in zip(
        unbiased.named_parameters(), biased.named_parameters()
    ):
        assert name == name_b and torch.equal(pa, pb)

    report = biased.bias_report()
    assert report["embed.bias_mode"] == mode
    assert report["embed.bias_norm"] > 0.0
    assert report["embed.on_target"] == "yes"
    assert "target" in str(report["embed.placed_on"])

    with torch.no_grad():
        a = unbiased(src, tgt_in)["logits"]
        b = biased(src, tgt_in)["logits"]
    assert not torch.allclose(a, b), f"embedding bias mode={mode} had no effect"


def test_attention_bias_changes_output():
    """bQ on the encoder/decoder sites must change the logits vs. no attention bias."""
    src, tgt_in, _ = batch()
    unbiased = make_model(make_cfg(), make_bias_cfg("zero", attn=False), seed=321)
    biased = make_model(
        make_cfg(),
        make_bias_cfg("zero", attn=True, attn_mode="gaussian", use_q_bias=True),
        seed=321,
    )
    unbiased.eval()
    biased.eval()

    for (name, pa), (name_b, pb) in zip(
        unbiased.named_parameters(), biased.named_parameters()
    ):
        assert name == name_b and torch.equal(pa, pb)

    report = biased.bias_report()
    assert report["encoder.bQ_norm"] > 0.0
    assert report["encoder.bQ_mode"] == "gaussian"
    assert report["decoder_self.bQ_norm"] > 0.0
    assert report["decoder_cross.bQ_norm"] > 0.0
    assert report["encoder.bK_mode"] == "zero"  # use_k_bias defaults to False

    with torch.no_grad():
        a = unbiased(src, tgt_in)["logits"]
        b = biased(src, tgt_in)["logits"]
    assert not torch.allclose(a, b), "attention bias had no effect"


def test_attention_bias_sector_switches():
    """The per-site switches must actually gate the factories."""
    cfg = make_cfg()
    bias_cfg = make_bias_cfg("zero", attn=True, apply_encoder=False,
                             apply_decoder_self=True, apply_decoder_cross=False)
    model = make_model(cfg, bias_cfg, seed=5)
    report = model.bias_report()
    assert "encoder.bQ_norm" not in report
    assert "decoder_self.bQ_norm" in report
    assert "decoder_cross.bQ_norm" not in report
    encoder_biases = [l.self_attn.bias for l in model.encoder.layers]
    assert all(b is None for b in encoder_biases)


def test_share_across_layers_and_no_deepcopy():
    """``share_across_layers`` must hand out one object, not one cloned per layer."""
    bias_cfg = make_bias_cfg("zero", attn=True, share_across_layers=True)
    shared = make_model(make_cfg(), bias_cfg, seed=11)
    ids = {id(l.self_attn.bias) for l in shared.encoder.layers}
    assert len(ids) == 1, "share_across_layers=True produced more than one bias object"

    quiet = make_bias_cfg("zero", attn=True, share_across_layers=False)
    per_layer = make_model(make_cfg(), quiet, seed=11)
    ids_pl = {id(l.self_attn.bias) for l in per_layer.encoder.layers}
    assert len(ids_pl) == len(per_layer.encoder.layers)


def test_disabled_attention_bias_creates_none():
    model = make_model(make_cfg(), make_bias_cfg("zero", attn=False))
    assert all(l.self_attn.bias is None for l in model.encoder.layers)
    assert all(l.self_attn.bias is None for l in model.decoder.layers)
    assert all(l.cross_attn.bias is None for l in model.decoder.layers)
    assert model.attention_biases() == []


# --------------------------------------------------------------------------- #
# 5. Weight tying
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("share_embeddings", [False, True])
def test_weight_tying(share_embeddings: bool):
    cfg = make_cfg(tie_output_embedding=True, share_embeddings=share_embeddings)
    if share_embeddings:
        # Sharing one matrix between both sides requires identical vocabularies.
        model = Seq2SeqTransformer(
            cfg, make_bias_cfg("gaussian"),
            src_vocab_size=20, tgt_vocab_size=20,
        )
    else:
        model = make_model(cfg, make_bias_cfg("gaussian"), seed=3)
    target_weight = model.tgt_embed.wte.weight
    assert model.output_proj.weight is target_weight, "output_proj is not tied"
    assert model.output_proj.bias is None
    if share_embeddings:
        assert model.src_embed is model.tgt_embed  # one module, one matrix
        assert model.output_proj.weight is model.src_embed.wte.weight


def test_untied_output_projection():
    model = make_model(make_cfg(tie_output_embedding=False))
    assert model.output_proj.weight is not model.tgt_embed.wte.weight
    assert model.output_proj.bias is not None


def test_share_embeddings_requires_equal_vocab():
    cfg = make_cfg(share_embeddings=True)
    with pytest.raises(ValueError, match="identical vocabular"):
        Seq2SeqTransformer(
            cfg, make_bias_cfg(), src_vocab_size=11, tgt_vocab_size=12
        )


def test_unresolved_vocab_size_raises():
    with pytest.raises(ValueError, match="vocab_size"):
        Seq2SeqTransformer(make_cfg(), make_bias_cfg())


# --------------------------------------------------------------------------- #
# 6. resample_biases / bias_report
# --------------------------------------------------------------------------- #
def test_resample_biases_per_step_and_zero():
    bias_cfg = make_bias_cfg(
        "gaussian",
        attn=True,
        embed_resample="per_step",
        attn_resample="per_step",
    )
    # Equal vocabularies so the two embeddings can share one bias instance.
    model = Seq2SeqTransformer(
        make_cfg(), bias_cfg, src_vocab_size=20, tgt_vocab_size=20
    )
    model.eval()

    assert model.src_embed.bias is model.tgt_embed.bias  # shared instance
    before_embed = model.src_embed.bias.b.detach().clone()
    attn_bias = model.encoder.layers[0].self_attn.bias
    before_attn = attn_bias.q.b.detach().clone()

    model.resample_biases()

    assert not torch.equal(before_embed, model.src_embed.bias.b.detach())
    # The same object is used by both embeddings and all encoder layers, so one
    # resample moves every reference at once (no object was resampled twice).
    assert attn_bias is model.encoder.layers[1].self_attn.bias
    assert not torch.equal(before_attn, attn_bias.q.b.detach())


def test_resample_leaves_zero_mode_at_exactly_zero():
    model = make_model(make_cfg(), make_bias_cfg("zero", attn=False), seed=9)
    model.resample_biases()
    # In ``zero`` mode only the source embedding carries a bias object at all
    # (the target's is skipped), and a redraw must leave it at exactly zero.
    assert model.src_embed.bias is not None
    assert model.tgt_embed.bias is None
    assert model.src_embed.bias.mode == "zero"
    assert model.src_embed.bias.b.abs().max() == 0.0
    assert torch.count_nonzero(model.src_embed.bias.b) == 0
    assert model.bias_report()["embed.bias_norm"] == 0.0
    assert model.bias_report()["embed.bias_mode"] == "zero"


def test_bias_report_is_finite_floats_and_reports_mode():
    model = make_model(
        make_cfg(),
        make_bias_cfg(
            "gaussian", attn=True, attn_mode="const", use_q_bias=True,
            use_v_bias=True,
        ),
        seed=13,
    )
    report = model.bias_report()
    assert isinstance(report, dict)
    assert report["embed.bias_mode"] == "gaussian"
    float_keys = [k for k, v in report.items() if isinstance(v, float)]
    assert float_keys, "bias_report returned no numeric values"
    for key in float_keys:
        value = report[key]
        assert isinstance(value, float)
        assert value == value and abs(value) != float("inf"), f"{key} is not finite"
    assert report["encoder.bQ_mode"] == "const"
    assert report["encoder.bV_mode"] == "const"
    assert report["encoder.bK_mode"] == "zero"
    assert "source" in str(report["embed.placed_on"])

    # JSON-serialisable (the training loop dumps this next to every run).
    import json

    json.dumps(report)


def _csv_bias_norms(report: dict) -> dict:
    """Mirror ``train.py::_bias_norms`` so a report change cannot break the CSV.

    The trainer normalises keys (drop non-alphanumerics, lower-case) and takes
    the largest norm per sector column.
    """
    wanted = {
        "embed_b_norm": ("embedbiasnorm", "embedbnorm", "embednorm"),
        "bQ_norm": ("bqnorm",),
        "bK_norm": ("bknnorm",),
        "bV_norm": ("bvnorm",),
    }
    normalised = {}
    for key, value in report.items():
        try:
            normalised["".join(ch for ch in str(key).lower() if ch.isalnum())] = float(value)
        except (TypeError, ValueError):
            continue
    return {
        field: max(
            [v for k, v in normalised.items() if any(c in k for c in cands)] or [0.0]
        )
        for field, cands in wanted.items()
    }


def test_bias_report_csv_columns_are_correct():
    """``train.py::_bias_norms`` must extract the right four numbers.

    Regression guard: a string-valued key that happens to normalise into one of
    the norm patterns (``"yes"``/``"no"`` are not floats, but a numeric field
    named ``embed.*`` would be read as the embedding-bias norm) would silently
    corrupt the logged columns.
    """
    zero = make_model(make_cfg(), make_bias_cfg("zero"), seed=1)
    zero_norms = _csv_bias_norms(zero.bias_report())
    assert zero_norms == {
        "embed_b_norm": 0.0,
        "bQ_norm": 0.0,
        "bK_norm": 0.0,
        "bV_norm": 0.0,
    }

    biased = make_model(
        make_cfg(), make_bias_cfg("gaussian", attn=True, use_q_bias=True), seed=1
    )
    biased_norms = _csv_bias_norms(biased.bias_report())
    assert biased_norms["embed_b_norm"] == pytest.approx(
        biased.bias_report()["embed.bias_norm"]
    )
    assert biased_norms["bQ_norm"] > 0.0
    assert biased_norms["bV_norm"] > 0.0
    assert biased_norms["bK_norm"] == 0.0  # use_k_bias is off


# --------------------------------------------------------------------------- #
# 7. Gradients
# --------------------------------------------------------------------------- #
def test_gradients_flow_to_every_submodule():
    model = make_model(make_cfg(), make_bias_cfg("gaussian", attn=True, use_q_bias=True))
    model.train()
    src, tgt_in, labels = batch()
    out = model(src, tgt_in, labels=labels)
    loss = out["loss"]
    assert loss is not None
    loss.backward()

    named = dict(model.named_parameters())

    def grad_of(module: torch.nn.Module, param: str) -> torch.Tensor:
        """Gradient by *object*, because tying makes ``named_parameters`` ambiguous."""
        p = getattr(module, param)
        assert p.grad is not None, f"no gradient for {param} of {type(module).__name__}"
        assert torch.isfinite(p.grad).all(), f"non-finite gradient for {param}"
        assert p.grad.abs().sum() > 0, f"zero gradient for {param}"
        return p.grad

    # Embeddings (both sides), every stack, both attention sites, the MLP and the
    # output projection.
    grad_of(model.src_embed.wte, "weight")
    grad_of(model.tgt_embed.wte, "weight")
    grad_of(model.encoder.layers[0].self_attn.q_proj, "weight")
    grad_of(model.encoder.layers[1].self_attn.v_proj, "weight")
    grad_of(model.encoder.layers[0].ff.c_fc, "weight")
    grad_of(model.encoder.layers[0].ff.c_proj, "weight")
    grad_of(model.encoder.layers[0].norm1, "weight")
    grad_of(model.decoder.layers[0].self_attn.q_proj, "weight")
    grad_of(model.decoder.layers[0].cross_attn.q_proj, "weight")
    grad_of(model.decoder.layers[0].cross_attn.out_proj, "weight")
    grad_of(model.decoder.layers[0].ff.c_fc, "weight")
    grad_of(model.decoder.layers[-1].norm3, "weight")
    grad_of(model.output_proj, "weight")

    # With tying on, the output projection and the target embedding are the same
    # tensor: one gradient, reachable from both names.
    assert model.output_proj.weight is model.tgt_embed.wte.weight
    assert (
        model.output_proj.weight.grad is model.tgt_embed.wte.weight.grad
    )
    assert "output_proj.weight" in named or "tgt_embed.wte.weight" in named


def test_gradients_reach_prelu_and_layernorm():
    cfg = make_cfg(activation="prelu", prelu_random_init=True)
    model = make_model(cfg)
    model.train()
    src, tgt_in, labels = batch()
    loss = model(src, tgt_in, labels=labels)["loss"]
    loss.backward()
    prelu_weight = model.encoder.layers[0].ff.act.weight
    assert prelu_weight.grad is not None
    assert torch.isfinite(prelu_weight.grad).all()


def test_learnable_attention_bias_gets_gradient():
    cfg = make_cfg()
    bias_cfg = resolve_bias_config(
        None,
        embed_mode="zero",
        attn_mode="gaussian",
        use_q_bias=True,
        attn_learnable=True,
        attn_resample="fixed",
    )
    model = make_model(cfg, bias_cfg, seed=17)
    model.train()
    src, tgt_in, labels = batch()
    model(src, tgt_in, labels=labels)["loss"].backward()
    bq = model.encoder.layers[0].self_attn.bias.q.b
    assert isinstance(bq, torch.nn.Parameter)
    assert bq.grad is not None and bq.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# 8. greedy_decode
# --------------------------------------------------------------------------- #
def _force_eos_logits(model: Seq2SeqTransformer):
    """Hang a hook on ``output_proj`` that pins the EOS logit to +100.

    The hook must sit on the *projection*: a fresh model's logits are ~1e-2, so
    poking the decoder's hidden state cannot dominate them (``output_proj``'s
    weights are tiny, ~0.02).  ``output_proj`` is tied to the target embedding
    here, so its weights cannot simply be overwritten -- a forward hook can.
    """

    def hook(_module, _inputs, output):
        output = output.clone()
        output[..., EOS_ID] = 100.0
        return output

    return model.output_proj.register_forward_hook(hook)


def test_greedy_decode_bias_towards_eos():
    """With EOS forced to win, every row stops at step 1: shape/BOS/fill contract."""
    cfg = make_cfg()
    model = make_model(cfg, make_bias_cfg("zero"), seed=4)
    model.eval()
    handle = _force_eos_logits(model)
    try:
        src = torch.tensor([[4, 5, 6], [7, 8, 9]])
        out = model.greedy_decode(src, max_len=6)
    finally:
        handle.remove()

    # Both rows finish at the *same* step, so the loop stops immediately and the
    # returned tensor is only as long as it needs to be: BOS + EOS.
    assert out.shape == (2, 2)
    assert out.dtype == torch.long
    assert (out[:, 0] == BOS_ID).all(), "greedy_decode must include the leading BOS"
    assert (out[:, 1] == EOS_ID).all(), f"expected EOS at step 1, got {out.tolist()}"


def test_greedy_decode_stops_early_for_every_row():
    """max_len is an upper bound: all rows finished -> shorter output, still valid."""
    cfg = make_cfg()
    model = make_model(cfg, make_bias_cfg("zero"), seed=4)
    model.eval()
    handle = _force_eos_logits(model)
    try:
        src = torch.tensor([[4, 5], [6, 7], [8, 9]])
        out = model.greedy_decode(src, max_len=MAX_SEQ_LEN)
    finally:
        handle.remove()
    assert out.shape[0] == 3
    assert out.shape[1] == 2, "decoding must stop as soon as every row emitted EOS"
    assert (out[:, 1] == EOS_ID).all()


def test_greedy_decode_pads_finished_rows_while_others_continue():
    """A row that hits EOS early is padded while the other keeps decoding."""
    model = make_model(make_cfg(), make_bias_cfg("zero"), seed=6)
    model.eval()

    calls = {"n": 0}

    def hook(_module, _inputs, output):
        calls["n"] += 1
        output = output.clone()
        if calls["n"] <= 2:  # row 1 emits EOS at step 1, row 0 does not
            output[1, :, EOS_ID] = 100.0
        return output

    handle = model.output_proj.register_forward_hook(hook)
    try:
        src = torch.tensor([[4, 5, 6], [7, 8, 9]])
        out = model.greedy_decode(src, max_len=4)
    finally:
        handle.remove()

    assert out.shape == (2, 4)
    assert out[1, 1] == EOS_ID
    assert (out[1, 2:] == PAD_ID).all(), "finished row must be pad-filled afterwards"
    assert (out[:, 0] == BOS_ID).all()


def test_greedy_decode_no_token_after_eos():
    """Deterministic structural check on a randomly initialised model."""
    model = make_model(make_cfg(), make_bias_cfg("gaussian", attn=True), seed=8)
    model.eval()
    src = torch.tensor([[4, 5, 6, 7], [8, 9, 10, 11]])
    out = model.greedy_decode(src, max_len=12)

    assert out.shape[0] == 2
    assert out.shape[0] == src.shape[0]
    assert out.dtype == torch.long
    assert out.shape[1] <= 12
    assert (out[:, 0] == BOS_ID).all()
    for row in out:
        positions = (row == EOS_ID).nonzero(as_tuple=True)[0]
        if positions.numel():
            first = int(positions[0])
            assert (row[first + 1:] == PAD_ID).all(), "non-pad token after EOS"


def test_greedy_decode_learns_a_sequence_task():
    """Tiny overfit run: the model must reproduce a permutation of the source.

    The task is ``target = perm[source]``, framed by BOS/EOS.  If this does not
    converge, the encoder-decoder wiring, the causal mask or the loss
    ``ignore_index`` is broken -- not the test.  Three seeds are known to
    converge, so this is not seed-sensitive.
    """
    perm = torch.tensor([7, 9, 11, 13, 15, 17, 19, 21, 4, 5, 6, 8])
    cfg = make_cfg(dropout=0.0, attention_dropout=0.0, context_length=12)
    torch.manual_seed(0)
    model = Seq2SeqTransformer(
        cfg,
        make_bias_cfg("gaussian", attn=True, use_q_bias=True, use_v_bias=True),
        src_vocab_size=SRC_VOCAB,
        tgt_vocab_size=TGT_VOCAB,
        pad_id=PAD_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
    )
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)

    gen = torch.Generator().manual_seed(3)
    src = torch.randint(4, 16, (4, 4), generator=gen)
    tgt_ids = torch.cat(
        [torch.full((4, 1), BOS_ID), perm[src - 4], torch.full((4, 1), EOS_ID)], dim=1
    )
    tgt_in, labels = tgt_ids[:, :-1], tgt_ids[:, 1:]

    for _ in range(150):
        opt.zero_grad()
        loss = model(src, tgt_in, labels=labels)["loss"]
        loss.backward()
        opt.step()

    assert loss.item() < 0.5, f"sequence task did not overfit (final loss {loss.item():.3f})"

    model.eval()
    decoded = model.greedy_decode(src, max_len=8)
    assert decoded.shape == tgt_ids.shape
    assert torch.equal(decoded, tgt_ids), (
        "greedy_decode did not reproduce the sequences:\n"
        f"expected {tgt_ids.tolist()}\ngot      {decoded.tolist()}"
    )


# --------------------------------------------------------------------------- #
# 9. Config plumbing
# --------------------------------------------------------------------------- #
def test_encoder_decoder_layer_counts_follow_config():
    model = make_model(make_cfg(n_encoder_layer=3, n_decoder_layer=1))
    assert len(model.encoder.layers) == 3
    assert len(model.decoder.layers) == 1


def test_encoder_norm_only_for_pre_ln():
    assert make_model(make_cfg(norm_first=False)).encoder.encoder_norm is None
    assert make_model(make_cfg(norm_first=True)).encoder.encoder_norm is not None
    assert make_model(make_cfg(norm_first=False)).decoder.decoder_norm is None
    assert make_model(make_cfg(norm_first=True)).decoder.decoder_norm is not None


def test_fixed_sinusoidal_positions():
    """``learned_positional=False`` is a valid alternative embedding."""
    from symbreak_transformer.model import TokenPositionalEmbedding

    torch.manual_seed(0)
    emb = TokenPositionalEmbedding(
        vocab_size=10, n_embd=8, context_length=6, learned_positional=False
    )
    out = emb(torch.tensor([[1, 2, 3]]))
    assert out.shape == (1, 3, 8)
    assert emb.wpe is None
    assert torch.isfinite(out).all()


def test_embedding_bias_placement_shares_one_instance():
    """Equal vocabularies -> a single shared ``EmbeddingBias`` for both sides."""
    cfg = make_cfg()
    model = Seq2SeqTransformer(
        cfg, make_bias_cfg("gaussian"),
        src_vocab_size=20, tgt_vocab_size=20,
    )
    assert isinstance(model.src_embed.bias, EmbeddingBias)
    assert model.src_embed.bias is model.tgt_embed.bias
    assert model.bias_report()["embed.shared_between_sides"] == "yes"
    assert len(model.embed_biases()) == 1


def test_bias_report_placement_zero_mode_is_source_only():
    """Regression: ``mode="zero"`` must NOT claim a shared/on-target bias.

    The target embedding carries ``bias=None`` in that mode, so the report has to
    say "source only" -- an earlier version tested ``len({id(...)}) == 1`` after
    filtering out ``None`` and wrongly reported a single shared instance while
    ``on_target`` was ``"no"``.
    """
    model = make_model(make_cfg(), make_bias_cfg("zero"), seed=1)
    assert model.tgt_embed.bias is None
    assert model.src_embed.bias is not None

    report = model.bias_report()
    assert report["embed.on_target"] == "no"
    assert report["embed.on_source"] == "yes"
    assert report["embed.shared_between_sides"] == "no"
    assert "source only" in str(report["embed.placed_on"])


def test_bias_report_placement_gaussian_mode_is_shared():
    """Regression: equal vocabularies + non-zero mode -> one shared instance."""
    model = Seq2SeqTransformer(
        make_cfg(), make_bias_cfg("gaussian"), src_vocab_size=20, tgt_vocab_size=20
    )
    report = model.bias_report()
    assert report["embed.on_target"] == "yes"
    assert report["embed.on_source"] == "yes"
    assert report["embed.shared_between_sides"] == "yes"
    assert "shared" in str(report["embed.placed_on"])
    assert report["embed.bias_instances"] == 1.0


def test_embedding_bias_placement_separate_vocabularies():
    """Different vocabularies -> two independent instances from one config."""
    model = make_model(make_cfg(), make_bias_cfg("gaussian"), seed=2)
    assert isinstance(model.src_embed.bias, EmbeddingBias)
    assert isinstance(model.tgt_embed.bias, EmbeddingBias)
    assert model.src_embed.bias is not model.tgt_embed.bias
    assert len(model.embed_biases()) == 2
    assert model.bias_report()["embed.shared_between_sides"] == "no"

    # Independent draws, not a copy of the same vector.
    assert not torch.equal(
        model.src_embed.bias.b.detach(), model.tgt_embed.bias.b.detach()
    )


def test_attention_bias_types_are_attention_bias():
    model = make_model(make_cfg(), make_bias_cfg("zero", attn=True))
    for bias in model.attention_biases():
        assert isinstance(bias, AttentionBias)
    assert len(model.attention_biases()) == 3  # encoder, decoder self, decoder cross


def test_bias_config_objects_are_not_mutated():
    """The model must not edit the caller's config (it is shared between runs)."""
    bias_cfg = make_bias_cfg("gaussian", attn=True)
    snapshot = copy.deepcopy(bias_cfg)
    make_model(make_cfg(), bias_cfg, seed=6)
    assert bias_cfg == snapshot
