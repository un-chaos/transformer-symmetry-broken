"""Evaluation helpers: validation loss, greedy decoding and corpus BLEU.

BLEU is implemented here (no ``sacrebleu`` dependency) so the repo runs on a
bare ``torch`` + ``numpy`` install.  ``sacrebleu`` is used automatically when it
happens to be importable, for cross-checking.

Plotting lives in ``scripts/plot_curve.py``, not here: the evaluation helpers are
imported by the training loop on every run, and a plotting dependency (matplotlib
+ pandas) has no business in that path.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# ``PAD_ID`` lives with the tokenizer (``data/tokenizer.py``: 0/1/2/3 for
# pad/unk/bos/eos).  The fallback keeps this module importable on its own -- and
# the pad index is part of the frozen data contract, so the two cannot drift.
try:  # pragma: no cover - the import always succeeds in a full checkout
    from .data.tokenizer import PAD_ID
except ImportError:  # pragma: no cover
    PAD_ID = 0

__all__ = [
    "evaluate_loss",
    "corpus_bleu",
    "cross_check_with_sacrebleu",
    "greedy_translate",
    "evaluate_bleu",
]


# --------------------------------------------------------------------------- #
# Validation loss
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_loss(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    pad_id: int = PAD_ID,
    max_batches: int = 0,
) -> float:
    """Mean cross-entropy over the loader, weighted by non-pad target tokens.

    Args:
        model: the seq2seq transformer (must accept ``src, tgt_in, ...``).
        loader: a ``DataLoader`` built by
            :func:`symbreak_transformer.data.dataset.build_dataloaders`.
        device: torch device.
        pad_id: ignored label index.
        max_batches: 0 evaluates the whole loader, otherwise the first N batches.

    Returns:
        The token-weighted mean loss, or ``float("nan")`` when nothing was scored.
    """
    was_training = model.training
    model.eval()
    total = 0.0
    n_tokens = 0
    for index, batch in enumerate(loader):
        if max_batches and index >= max_batches:
            break
        tgt_out = batch["tgt_out"].to(device)
        out = model(
            batch["src"].to(device),
            batch["tgt_in"].to(device),
            src_padding_mask=batch["src_padding_mask"].to(device),
            tgt_padding_mask=batch["tgt_padding_mask"].to(device),
            labels=tgt_out,
        )
        loss = out.get("loss")
        if loss is None or not torch.isfinite(loss):
            continue
        count = int((tgt_out != pad_id).sum().item())
        if count == 0:
            continue
        total += float(loss.item()) * count
        n_tokens += count
    if was_training:
        model.train()
    if n_tokens == 0:
        return float("nan")
    return total / n_tokens


# --------------------------------------------------------------------------- #
# BLEU
# --------------------------------------------------------------------------- #
def _ngram_counts(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def corpus_bleu(
    hypotheses: Sequence[Sequence[str]],
    references: Sequence[Sequence[str]],
    max_n: int = 4,
    smoothing: str = "add1",
) -> Dict[str, object]:
    """Corpus BLEU with a brevity penalty.

    Standard clipped-precision BLEU (Papineni et al. 2002).  Very short
    hypotheses have no 4-grams, which makes the strict score collapse to zero;
    ``smoothing="add1"`` (the default) replaces a zero clipped precision by
    ``1 / (total_n + 1)`` so the number stays informative on small models.
    The unsmoothed value is always reported as ``bleu_strict`` as well.

    Args:
        hypotheses: one token sequence per sentence.
        references: one token sequence per sentence (a single reference each).
        max_n: highest n-gram order (4 for BLEU-4).
        smoothing: ``"add1"`` or ``"none"``.

    Returns:
        Dict with ``bleu`` (0-100, smoothed per ``smoothing``), ``bleu_strict``,
        ``precisions`` (percent, orders 1..max_n), ``bp``, ``ratio``,
        ``hyp_len``, ``ref_len`` and ``n_samples``.
    """
    if len(hypotheses) != len(references):
        raise ValueError(
            f"got {len(hypotheses)} hypotheses but {len(references)} references"
        )
    if smoothing not in ("add1", "none"):
        raise ValueError(f"smoothing must be 'add1' or 'none', got {smoothing!r}")

    clipped = [0] * (max_n + 1)
    total = [0] * (max_n + 1)
    hyp_len = 0
    ref_len = 0

    for hyp, ref in zip(hypotheses, references):
        hyp_len += len(hyp)
        ref_len += len(ref)
        for n in range(1, max_n + 1):
            hyp_ngrams = _ngram_counts(hyp, n)
            ref_ngrams = _ngram_counts(ref, n)
            total[n] += sum(hyp_ngrams.values())
            for gram, count in hyp_ngrams.items():
                clipped[n] += min(count, ref_ngrams.get(gram, 0))

    precisions: List[float] = []
    precisions_smoothed: List[float] = []
    for n in range(1, max_n + 1):
        if total[n] == 0:
            precisions.append(0.0)
            precisions_smoothed.append(0.0)
        else:
            raw = clipped[n] / total[n]
            precisions.append(raw)
            if raw > 0.0 or smoothing == "none":
                precisions_smoothed.append(raw)
            else:
                precisions_smoothed.append(1.0 / (total[n] + 1.0))

    if hyp_len == 0 or ref_len == 0:
        bp = 0.0
    else:
        bp = 1.0 if hyp_len > ref_len else math.exp(1.0 - ref_len / hyp_len)

    def _combine(values: Sequence[float]) -> float:
        if any(v <= 0.0 for v in values):
            return 0.0
        return bp * math.exp(sum(math.log(v) for v in values) / max_n)

    return {
        "bleu": 100.0 * _combine(precisions_smoothed),
        "bleu_strict": 100.0 * _combine(precisions),
        "precisions": [100.0 * p for p in precisions],
        "bp": bp,
        "ratio": (hyp_len / ref_len) if ref_len else 0.0,
        "hyp_len": hyp_len,
        "ref_len": ref_len,
        "n_samples": len(hypotheses),
    }


def cross_check_with_sacrebleu(
    hypotheses: Sequence[Sequence[str]], references: Sequence[Sequence[str]]
) -> Optional[float]:
    """Return sacrebleu's BLEU-4 if that package happens to be installed."""
    try:
        import sacrebleu  # type: ignore
    except Exception:
        return None
    hyp = [" ".join(h) for h in hypotheses]
    ref = [[" ".join(r) for r in references]]
    return float(sacrebleu.corpus_bleu(hyp, ref).score)


# --------------------------------------------------------------------------- #
# Greedy translation
# --------------------------------------------------------------------------- #
def _pad_batch(sequences: Sequence[Sequence[int]], pad_id: int = PAD_ID) -> torch.Tensor:
    width = max((len(s) for s in sequences), default=0)
    out = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for row, seq in enumerate(sequences):
        if seq:
            out[row, : len(seq)] = torch.tensor(list(seq), dtype=torch.long)
    return out


def _tokens(tokenizer, text: str) -> List[str]:
    """Tokenize ``text`` with the tokenizer's own tokenizer (word or char mode)."""
    return list(tokenizer.tokenize(text))


@torch.no_grad()
def greedy_translate(
    model: torch.nn.Module,
    src_texts: Sequence[str],
    src_tokenizer,
    tgt_tokenizer,
    device: torch.device,
    max_len: int = 0,
    batch_size: int = 32,
) -> List[str]:
    """Greedily translate ``src_texts`` and return the decoded strings."""
    was_training = model.training
    model.eval()
    results: List[str] = []
    for start in range(0, len(src_texts), batch_size):
        chunk = src_texts[start : start + batch_size]
        src_ids = [
            src_tokenizer.encode(t, add_bos=False, add_eos=True, max_len=max_len or 0)
            for t in chunk
        ]
        src = _pad_batch(src_ids).to(device)
        mask = src == PAD_ID
        decoded = model.greedy_decode(src, src_padding_mask=mask, max_len=max_len or 0)
        for row in decoded.tolist():
            results.append(tgt_tokenizer.decode(row))
    if was_training:
        model.train()
    return results


@torch.no_grad()
def evaluate_bleu(
    model: torch.nn.Module,
    pairs: Sequence[Tuple[str, str]],
    src_tokenizer,
    tgt_tokenizer,
    device: torch.device,
    max_len: int = 0,
    max_samples: int = 0,
    batch_size: int = 32,
    show_samples: int = 0,
) -> Dict[str, object]:
    """Greedy-decode ``pairs`` and score corpus BLEU against the references.

    Args:
        pairs: ``(source_text, target_text)`` pairs.
        max_len: decoder cap; 0 lets the model use its configured maximum.
        max_samples: 0 scores every pair.
        show_samples: also return this many ``(src, ref, hyp)`` triples.

    Returns:
        The :func:`corpus_bleu` dict plus ``sacrebleu`` (when available) and
        optionally ``samples``.
    """
    subset = list(pairs[:max_samples]) if max_samples else list(pairs)
    if not subset:
        return {"bleu": 0.0, "bleu_strict": 0.0, "n_samples": 0}

    # Length-sorted batching keeps padding waste low.
    order = sorted(range(len(subset)), key=lambda i: len(subset[i][0]))
    hypotheses: List[List[str]] = []
    references: List[List[str]] = []
    samples: List[Dict[str, str]] = []

    was_training = model.training
    model.eval()
    for start in range(0, len(order), batch_size):
        idx = order[start : start + batch_size]
        src_ids = [
            src_tokenizer.encode(
                subset[i][0], add_bos=False, add_eos=True, max_len=max_len or 0
            )
            for i in idx
        ]
        src = _pad_batch(src_ids).to(device)
        decoded = model.greedy_decode(
            src, src_padding_mask=src == PAD_ID, max_len=max_len or 0
        ).tolist()
        for row, i in zip(decoded, idx):
            hyp_text = tgt_tokenizer.decode(row)
            hypotheses.append(_tokens(tgt_tokenizer, hyp_text))
            references.append(_tokens(tgt_tokenizer, subset[i][1]))
            if len(samples) < show_samples:
                samples.append(
                    {"src": subset[i][0], "ref": subset[i][1], "hyp": hyp_text}
                )
    if was_training:
        model.train()

    result = corpus_bleu(hypotheses, references)
    sacre = cross_check_with_sacrebleu(hypotheses, references)
    if sacre is not None:
        result["sacrebleu"] = sacre
    if show_samples:
        result["samples"] = samples
    return result
