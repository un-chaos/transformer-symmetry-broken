"""End-to-end tests for the FineWeb-Edu span-corruption (denoising) pipeline.

Everything here is **offline and self-contained**: the real corpus lives in
``data/fineweb-edu/`` and is ~28 GB, so these tests never touch it and never
touch the network.  Instead they train a tiny BPE tokenizer on a few hundred
synthetic English-ish lines and write a tiny parquet corpus (two shards, more
than one row group each) under ``tmp_path``.

The layout being pinned down is the one the real 10B-token run depends on::

    ids 0..3            <pad> <unk> <bos> <eos>
    ids 4..4+V-1        the V learned BPE tokens
    ids 4+V..4+V+S-1    <extra_id_0> .. <extra_id_{S-1}> (span sentinels)

A regression in the sentinel count used to derive ``0`` (the trailing run was
scanned from the wrong end), which silently broke every corruption, so that
count is checked from both ends here.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from symbreak_transformer.config import DataConfig
from symbreak_transformer.data.bpe import (
    DEFAULT_NUM_SENTINELS,
    SENTINEL_PREFIX,
    BPETokenizer,
    build_bpe_tokenizer,
    tokenizer_path_for,
)
from symbreak_transformer.data.dataset import collate_batch
from symbreak_transformer.data.denoising import (
    FineWebDenoisingDataset,
    build_denoising_dataloaders,
    corrupt_spans,
)
from symbreak_transformer.data.tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
)

# --------------------------------------------------------------------------- #
# Tiny synthetic corpus
# --------------------------------------------------------------------------- #
_WORDS = (
    "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "river",
    "mountain", "science", "data", "model", "language", "token", "span",
    "corpus", "network", "training", "objective", "denoising", "encoder",
    "decoder", "attention", "sentence", "document", "sample", "english",
    "text", "word", "vocabulary", "batch", "gradient", "random", "seed",
)

_SENTINEL_RE = re.compile(re.escape(SENTINEL_PREFIX) + r"(\d+)>")

#: Small on purpose: the BPE trainer needs well under a second for these.
_TRAIN_VOCAB = 400
_TRAIN_SENTINELS = 20


def _sentinel_pairs(ids, itos):
    """The ``(index, id)`` pairs of the sentinels in ``ids``, in order.

    Ids alone cannot tell a sentinel from a BPE token, so the id -> token table
    is looked up; a value outside ``itos`` is simply not a sentinel.
    """
    pairs = []
    for value in ids:
        token = itos[value] if 0 <= value < len(itos) else ""
        match = _SENTINEL_RE.fullmatch(token)
        if match is not None:
            pairs.append((int(match.group(1)), value))
    return pairs


def _sentinel_ids(ids, itos):
    """Every id in ``ids`` that is a reserved sentinel, in order."""
    return [value for _, value in _sentinel_pairs(ids, itos)]


def _synthetic_lines(count: int = 300, seed: int = 17):
    """English-ish lines built from a fixed word list, deterministically."""
    rng = random.Random(seed)
    lines = []
    for index in range(count):
        length = 6 + index % 7
        words = [_WORDS[rng.randrange(len(_WORDS))] for _ in range(length)]
        words[0] = words[0].capitalize()
        lines.append(" ".join(words) + f". Doc {index}.")
    return lines


def _write_corpus(directory: Path, shards: int = 2, rows: int = 120,
                  row_group_size: int = 50) -> Path:
    """Write a tiny parquet corpus (>= 2 row groups per shard) and return it."""
    directory.mkdir(parents=True, exist_ok=True)
    lines = _synthetic_lines(shards * rows)
    for shard in range(shards):
        chunk = lines[shard * rows : (shard + 1) * rows]
        table = pa.table({"text": pa.array(chunk, type=pa.string())})
        # An explicit row_group_size gives each shard several row groups, which is
        # what the dataset's metadata-only indexing walk has to handle.
        pq.write_table(
            table, directory / f"shard_{shard:05d}.parquet",
            row_group_size=row_group_size,
        )
    return directory


def _more_sentinels(tokenizer: BPETokenizer, count: int) -> BPETokenizer:
    """The same tokenizer with a larger reserved-sentinel run.

    ``num_sentinels`` is also the ceiling on how many spans a corruption may
    make (``max_spans = num_sentinels - 1``), so a test about the *realised*
    density has to widen the budget before it can see ``noise_density`` at all.
    """
    itos = list(tokenizer.itos) + [
        f"{SENTINEL_PREFIX}{index}>"
        for index in range(tokenizer.num_sentinels, int(count))
    ]
    return BPETokenizer(tokenizer.backend, itos)


def _cfg(directory: Path, **overrides) -> DataConfig:
    kwargs = dict(
        source="fineweb",
        objective="denoising",
        tokenizer="bpe",
        fineweb_dir=str(directory),
        text_column="text",
        max_documents=200,
        val_every=10,
        noise_density=0.15,
        mean_span_length=3.0,
        bpe_vocab_size=_TRAIN_VOCAB,
        tokenizer_train_documents=200,
    )
    kwargs.update(overrides)
    return DataConfig(**kwargs)


@pytest.fixture(scope="module")
def tokenizer() -> BPETokenizer:
    """A tiny BPE tokenizer trained once for the whole module."""
    return BPETokenizer.train(
        _synthetic_lines(), vocab_size=_TRAIN_VOCAB, num_sentinels=_TRAIN_SENTINELS
    )


@pytest.fixture
def config(tmp_path, tokenizer) -> DataConfig:
    """A ``DataConfig`` pointing at a fresh tiny corpus under ``tmp_path``."""
    return _cfg(_write_corpus(tmp_path / "corpus"))


# --------------------------------------------------------------------------- #
# 1. BPE id layout
# --------------------------------------------------------------------------- #
def test_bpe_id_layout(tokenizer):
    """Specials, then BPE, then the sentinel run -- and the count is derived."""
    assert tokenizer.itos[:4] == SPECIAL_TOKENS
    assert tokenizer.vocab_size == 4 + tokenizer.bpe_vocab_size + tokenizer.num_sentinels
    assert tokenizer.bpe_vocab_size == _TRAIN_VOCAB
    assert tokenizer.num_sentinels == _TRAIN_SENTINELS
    assert tokenizer.itos[-1] == f"{SENTINEL_PREFIX}{_TRAIN_SENTINELS - 1}>"

    # The sentinels are the trailing run, in ascending order: <extra_id_0> is the
    # first reserved id (not the last), which is what `sentinel_id` indexes from.
    run_start = 4 + tokenizer.bpe_vocab_size
    assert tokenizer.sentinel_id(0) == run_start
    assert tokenizer.sentinel_id(_TRAIN_SENTINELS - 1) == tokenizer.vocab_size - 1
    assert tokenizer.itos[tokenizer.sentinel_id(_TRAIN_SENTINELS - 1)] == (
        f"{SENTINEL_PREFIX}{_TRAIN_SENTINELS - 1}>"
    )

    # Before the fix this scan went the wrong way, derived 0 sentinels and left
    # every corruption call indexing an empty run.
    assert tokenizer.num_sentinels > 0

    with pytest.raises(IndexError):
        tokenizer.sentinel_id(tokenizer.num_sentinels)
    with pytest.raises(IndexError):
        tokenizer.sentinel_id(-1)


def test_default_sentinel_budget_matches_documented_value():
    """The default budget is the one the cached corpus tokenizer was trained with."""
    assert DEFAULT_NUM_SENTINELS == 100
    assert SENTINEL_PREFIX == "<extra_id_"


# --------------------------------------------------------------------------- #
# 2. save/load round trip
# --------------------------------------------------------------------------- #
def test_save_load_round_trip(tokenizer, tmp_path):
    """A fresh file path reproduces the vocabulary and the encodings exactly."""
    samples = [
        "the quick brown fox jumps over the lazy dog",
        "Denoising corrupts spans, then reconstructs them!",
        "doc 42: a short one.",
        "",
    ]
    before = [
        tokenizer.encode(text, add_bos=True, add_eos=True) for text in samples
    ]

    path = tmp_path / "nested" / "tokenizer_bpe.json"
    written = tokenizer.save(path)
    assert Path(written).exists()

    restored = BPETokenizer.load(written)
    assert restored.vocab_size == tokenizer.vocab_size
    assert restored.num_sentinels == tokenizer.num_sentinels
    assert restored.bpe_vocab_size == tokenizer.bpe_vocab_size
    assert restored.itos == tokenizer.itos
    assert restored.sentinel_id(0) == tokenizer.sentinel_id(0)
    after = [
        restored.encode(text, add_bos=True, add_eos=True) for text in samples
    ]
    assert after == before


def test_build_bpe_tokenizer_caches_and_reloads(config, tmp_path):
    """An explicit ``texts`` corpus is used, cached, and reloaded on the 2nd call."""
    cfg = _cfg(tmp_path / "corpus2", bpe_vocab_size=_TRAIN_VOCAB)
    cache = tokenizer_path_for(cfg)
    assert cache.name == f"tokenizer_bpe_{_TRAIN_VOCAB}.json"
    assert cache.parent == Path(cfg.fineweb_dir).parent

    texts = _synthetic_lines()
    first = build_bpe_tokenizer(cfg, texts=texts)
    assert cache.exists()
    assert first.num_sentinels == DEFAULT_NUM_SENTINELS
    assert first.itos[:4] == SPECIAL_TOKENS

    # Second call must load the cache rather than retrain (the same ids prove it).
    second = build_bpe_tokenizer(cfg, texts=["nothing", "like", "the", "first"])
    assert second.vocab_size == first.vocab_size
    assert second.bpe_vocab_size == first.bpe_vocab_size
    assert second.encode("the quick brown fox") == first.encode("the quick brown fox")


# --------------------------------------------------------------------------- #
# 3. encode
# --------------------------------------------------------------------------- #
def test_encode_framing_and_truncation(tokenizer):
    text = "the quick brown fox jumps over the lazy dog"
    plain = tokenizer.encode(text, add_bos=False, add_eos=False)
    assert len(plain) >= 5

    framed = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert framed[0] == BOS_ID
    assert framed[-1] == EOS_ID
    assert framed[1:-1] == plain

    assert tokenizer.encode(text, add_bos=False, add_eos=False) == plain

    clipped = tokenizer.encode(text, add_bos=True, add_eos=True, max_len=4)
    assert clipped == [BOS_ID, plain[0], plain[1], EOS_ID]
    assert len(clipped) == 4 and clipped[-1] == EOS_ID

    # A request longer than the text changes nothing.
    assert tokenizer.encode(text, add_bos=True, add_eos=True, max_len=9999) == framed


# --------------------------------------------------------------------------- #
# 4. corrupt_spans
# --------------------------------------------------------------------------- #
def test_corrupt_spans_structure(tokenizer):
    """Source/destination framing, sentinel order and full span recovery."""
    ids = [10 + (index * 7) % 300 for index in range(120)]
    corruption = corrupt_spans(
        ids, tokenizer, noise_density=0.15, mean_span_length=3.0,
        rng=random.Random(0),
    )
    spans = corruption.spans
    src, tgt = corruption.src_ids, corruption.tgt_ids

    assert spans, "a 120-token document should always be corrupted"
    assert len(spans) <= tokenizer.num_sentinels - 1

    # --- framing: the source carries no BOS but ends with EOS; the target is
    # framed [BOS] ... [EOS].
    assert src[-1] == EOS_ID and BOS_ID not in src
    assert tgt[0] == BOS_ID and tgt[-1] == EOS_ID

    # --- sentinels in the encoder input, ascending, one per span
    src_sentinels = _sentinel_ids(src, tokenizer.itos)
    assert _sentinel_pairs(src, tokenizer.itos) == [
        (k, tokenizer.sentinel_id(k)) for k in range(len(spans))
    ]
    assert len(src_sentinels) == len(spans)

    # --- the decoder target is [BOS], then <extra_id_k> + that span's tokens for
    # every k, then the closing sentinel and EOS.
    expected = [BOS_ID]
    for index, span in enumerate(spans):
        expected.append(tokenizer.sentinel_id(index))
        expected.extend(span)
    expected.append(tokenizer.sentinel_id(len(spans)))
    expected.append(EOS_ID)
    assert tgt == expected

    # --- every removed token is present in the target, in order
    all_sentinels = {
        tokenizer.sentinel_id(k) for k in range(tokenizer.num_sentinels)
    }
    recovered = [
        value for value in tgt
        if value not in all_sentinels and value not in (BOS_ID, EOS_ID)
    ]
    for removed_span in spans:
        for token in removed_span:
            assert token in tgt
    assert recovered == [token for removed_span in spans for token in removed_span]

    # --- the corruption actually removes the spans from the source: splicing each
    # span back in at its sentinel reproduces the original document exactly.
    # (``Corruption.spans`` holds the removed *token ids*, not source positions.)
    rebuilt = []
    cursor = 0
    for value in src:
        if value in src_sentinels:
            rebuilt.extend(spans[cursor])
            cursor += 1
        elif value != EOS_ID:
            rebuilt.append(value)
    assert cursor == len(spans), "one sentinel per span, no more"
    assert rebuilt == ids

    # ...and the surviving source tokens are exactly the original ones minus the
    # span tokens (as a multiset, since a token value can repeat in the document).
    kept = [value for value in src if value not in src_sentinels and value != EOS_ID]
    assert Counter(kept) == Counter(ids) - Counter(
        token for span in spans for token in span
    )


def test_corrupt_spans_shortening_matches_sentinels(tokenizer):
    """Each span collapses to one sentinel: remove `removed`, insert `spans`.

    Note the ``+ 1``: ``corrupt_spans`` appends ``<eos>`` to the source, so the
    source is exactly ``len(ids) - removed + spans + 1`` long.
    """
    ids = [10 + (index * 7) % 300 for index in range(400)]
    corruption = corrupt_spans(
        ids, tokenizer, noise_density=0.15, mean_span_length=3.0,
        rng=random.Random(0),
    )
    removed = sum(len(span) for span in corruption.spans)
    assert removed - len(corruption.spans) == len(ids) - (len(corruption.src_ids) - 1)

    # With unit-length spans the statement is exact: every span removes one token
    # and inserts one sentinel in its place, so the source keeps every original
    # token and only gains the trailing <eos>.
    unit = corrupt_spans(
        ids, tokenizer, noise_density=0.15, mean_span_length=1.0,
        rng=random.Random(0),
    )
    assert all(len(span) == 1 for span in unit.spans)
    assert len(unit.src_ids) == len(ids) + 1
    # ...and with room for every span the budget asks for (round(400 * 0.15)),
    # the count is the full 60 rather than the reserved-sentinel ceiling.
    roomy = corrupt_spans(
        ids, _more_sentinels(tokenizer, 200), noise_density=0.15,
        mean_span_length=1.0, rng=random.Random(0),
    )
    assert len(roomy.spans) == 60
    assert len(roomy.src_ids) == len(ids) - 60 + len(roomy.spans) + 1


def test_corrupt_spans_density_band(tokenizer):
    """The realised removal fraction stays in a loose band around ``noise_density``."""
    sentinel_tokenizer = _more_sentinels(tokenizer, 200)

    def density(n_tokens: int, seed: int) -> float:
        ids = list(range(1, n_tokens + 1))
        corruption = corrupt_spans(
            ids, sentinel_tokenizer, noise_density=0.15, mean_span_length=3.0,
            rng=random.Random(seed),
        )
        return sum(len(span) for span in corruption.spans) / n_tokens

    measured = [density(1200, seed) for seed in range(8)]
    assert all(0.05 <= value <= 0.35 for value in measured), measured
    # Sanity: it really is doing damage, not cancelling out at zero.
    assert all(value > 0.0 for value in measured)


def test_corrupt_spans_is_seed_reproducible(tokenizer):
    ids = [10 + (index * 31) % 500 for index in range(300)]
    first = corrupt_spans(ids, tokenizer, rng=random.Random(0))
    same = corrupt_spans(ids, tokenizer, rng=random.Random(0))
    other = corrupt_spans(ids, tokenizer, rng=random.Random(12345))

    assert (first.src_ids, first.tgt_ids, first.spans) == (
        same.src_ids, same.tgt_ids, same.spans
    )
    assert (first.src_ids, first.tgt_ids, first.spans) != (
        other.src_ids, other.tgt_ids, other.spans
    )


@pytest.mark.parametrize("ids", [[5, 6, 7], [5, 6], [5], []])
def test_corrupt_spans_survives_tiny_documents(tokenizer, ids):
    """A three-token (or empty) document still yields a framed pair, no raise."""
    corruption = corrupt_spans(ids, tokenizer, rng=random.Random(0))
    src, tgt = corruption.src_ids, corruption.tgt_ids

    assert src[-1] == EOS_ID
    assert BOS_ID not in src
    assert tgt[0] == BOS_ID and tgt[-1] == EOS_ID
    assert len(tgt) >= 2  # collate_batch needs at least [BOS, EOS]

    expected_spans = [(0, 1)] if ids else []
    assert corruption.spans == [ids[start:end] for start, end in expected_spans]

    # Nothing was silently dropped on the way through.
    if ids:
        assert corruption.spans == [ids[:1]]


def test_corrupt_spans_truncation_keeps_eos(tokenizer):
    """``max_src_len`` / ``max_tgt_len`` cap the lengths and keep the framing."""
    ids = [10 + (index * 13) % 700 for index in range(600)]

    src_only = corrupt_spans(
        ids, tokenizer, max_src_len=16, rng=random.Random(0)
    )
    assert len(src_only.src_ids) == 16
    assert src_only.src_ids[-1] == EOS_ID

    tgt_only = corrupt_spans(
        ids, tokenizer, max_tgt_len=12, rng=random.Random(0)
    )
    assert len(tgt_only.tgt_ids) == 12
    assert tgt_only.tgt_ids[0] == BOS_ID and tgt_only.tgt_ids[-1] == EOS_ID

    both = corrupt_spans(
        ids, tokenizer, max_src_len=16, max_tgt_len=12, rng=random.Random(0)
    )
    assert len(both.src_ids) <= 16 and both.src_ids[-1] == EOS_ID
    assert len(both.tgt_ids) <= 12 and both.tgt_ids[-1] == EOS_ID

    # No truncation when the sequence already fits.
    roomy = corrupt_spans(
        [10 + (index * 13) % 700 for index in range(60)],
        tokenizer, max_src_len=512, max_tgt_len=512, rng=random.Random(0),
    )
    assert roomy.src_ids[-1] == EOS_ID and roomy.tgt_ids[-1] == EOS_ID


def test_corrupt_spans_respects_the_sentinel_budget(tokenizer):
    """Span count never exceeds the reserved sentinels minus the terminator."""
    ids = list(range(1, 2001))
    corruption = corrupt_spans(
        ids, tokenizer, noise_density=0.5, mean_span_length=1.0,
        rng=random.Random(0),
    )
    assert len(corruption.spans) <= tokenizer.num_sentinels - 1
    # The closing sentinel is the one right after the last span's sentinel.
    assert corruption.tgt_ids[-2] == tokenizer.sentinel_id(len(corruption.spans))


# --------------------------------------------------------------------------- #
# 5. FineWebDenoisingDataset
# --------------------------------------------------------------------------- #
def test_dataset_index_and_split(config, tokenizer):
    """``len()`` respects ``max_documents`` and the ``val_every`` holdout."""
    train = FineWebDenoisingDataset(config, tokenizer, split="train")
    val = FineWebDenoisingDataset(config, tokenizer, split="val")

    considered = min(len(train._entries), config.max_documents)
    assert considered == 200  # 2 shards x 3 row groups x 120 rows, capped
    assert len(val._rows) == len(range(0, considered, config.val_every))
    assert len(train._rows) + len(val._rows) == considered
    assert sorted(train._rows + val._rows) == list(range(considered))
    assert all(row % config.val_every == 0 for row in val._rows)
    assert all(row % config.val_every != 0 for row in train._rows)
    assert not set(train._rows) & set(val._rows)

    # A cap below one val_every still yields the documented splits: the cap
    # applies *before* the holdout, so 5 considered documents = 4 train + 1 val.
    capped = _cfg(Path(config.fineweb_dir), max_documents=5, val_every=10)
    assert len(FineWebDenoisingDataset(capped, tokenizer, split="train")) == 4
    assert len(FineWebDenoisingDataset(capped, tokenizer, split="val")) == 1

    with pytest.raises(ValueError):
        FineWebDenoisingDataset(config, tokenizer, split="test")


def test_dataset_index_walks_row_groups(config, tokenizer):
    """Every document of every row group is indexed, exactly once."""
    dataset = FineWebDenoisingDataset(config, tokenizer, split="train")
    expected = 0
    for shard in sorted(Path(config.fineweb_dir).glob("*.parquet")):
        metadata = pq.ParquetFile(shard).metadata
        assert metadata.num_row_groups > 1  # the fixture must exercise the walk
        expected += metadata.num_rows
    assert len(dataset._entries) == expected
    assert len({(str(s), g, r) for s, g, r in dataset._entries}) == expected


def test_dataset_getitem_is_deterministic(config, tokenizer):
    dataset = FineWebDenoisingDataset(config, tokenizer, split="train")
    assert len(dataset) > 3

    for index in (0, 1, 2):
        first = dataset[index]
        second = dataset[index]
        assert first == second
        src, tgt = first
        assert isinstance(src, list) and isinstance(tgt, list)
        assert src[-1] == EOS_ID and BOS_ID not in src
        assert tgt[0] == BOS_ID and tgt[-1] == EOS_ID
        assert all(isinstance(value, int) for value in src + tgt)

    # A different seed is allowed to corrupt differently (it usually does).
    other = FineWebDenoisingDataset(config, tokenizer, split="train", seed=999)
    assert other[0] != dataset[0] or other[1] != dataset[1]

    # Two datasets with the same seed agree, i.e. nothing global leaked in.
    twin = FineWebDenoisingDataset(config, tokenizer, split="train")
    assert [twin[i] for i in range(3)] == [dataset[i] for i in range(3)]


def test_dataset_missing_shard_message(tmp_path, tokenizer):
    cfg = _cfg(tmp_path / "empty", max_documents=10)
    with pytest.raises(FileNotFoundError) as excinfo:
        FineWebDenoisingDataset(cfg, tokenizer, split="train")
    assert "download-data" in str(excinfo.value)

    with pytest.raises(FileNotFoundError) as excinfo:
        build_bpe_tokenizer(_cfg(tmp_path / "empty2"))
    assert "download-data" in str(excinfo.value)


def test_dataset_items_satisfy_collate_batch_contract(config, tokenizer):
    dataset = FineWebDenoisingDataset(
        config, tokenizer, split="train", max_src_len=48, max_tgt_len=48
    )
    pairs = [dataset[index] for index in range(5)]
    batch = collate_batch(pairs)

    assert set(batch) == {
        "src", "tgt_in", "tgt_out", "src_padding_mask",
        "tgt_padding_mask", "src_lens", "tgt_lens",
    }
    assert batch["src"].shape[0] == len(pairs)
    assert batch["src_lens"].tolist() == [len(src) for src, _ in pairs]
    assert batch["tgt_lens"].tolist() == [len(tgt) for _, tgt in pairs]
    assert batch["src"].dtype == torch.long

    for row, (src, tgt) in enumerate(pairs):
        assert batch["src"][row, : len(src)].tolist() == src
        assert batch["tgt_in"][row, : len(tgt) - 1].tolist() == tgt[:-1]
        assert batch["tgt_out"][row, : len(tgt) - 1].tolist() == tgt[1:]

    # Nothing exceeds the requested budgets once framed.
    assert max(batch["src_lens"].tolist()) <= 48
    assert max(batch["tgt_lens"].tolist()) <= 48


# --------------------------------------------------------------------------- #
# 6. A real batch through the loaders
# --------------------------------------------------------------------------- #
def test_dataloaders_batch_contract_and_rng(config, tokenizer):
    """The full loader path: keys, mask polarity, the shift, and RNG isolation."""
    torch.manual_seed(1234)
    expected_draws = [torch.rand(1).item() for _ in range(3)]

    torch.manual_seed(1234)
    loaders = build_denoising_dataloaders(
        config, tokenizer, batch_size=4, num_workers=0, seed=7,
        max_src_len=48, max_tgt_len=48,
    )
    assert set(loaders) == {"train", "val"}
    after_draws = [torch.rand(1).item() for _ in range(3)]
    # Building the loaders must not consume the global torch stream.
    assert after_draws == expected_draws

    batch = next(iter(loaders["train"]))
    assert set(batch) == {
        "src", "tgt_in", "tgt_out", "src_padding_mask",
        "tgt_padding_mask", "src_lens", "tgt_lens",
    }
    assert batch["src"].shape[0] == 4
    assert batch["tgt_in"].shape == batch["tgt_out"].shape
    assert batch["tgt_in"].shape[1] == max(batch["tgt_lens"].tolist()) - 1
    assert batch["src_padding_mask"].dtype == torch.bool
    assert batch["tgt_padding_mask"].dtype == torch.bool

    # padding mask polarity: True == PAD, and only on the right of each row.
    for row in range(batch["src"].shape[0]):
        length = int(batch["src_lens"][row])
        assert not batch["src_padding_mask"][row, :length].any()
        assert batch["src_padding_mask"][row, length:].all()
        assert batch["src"][row, length:].tolist() == [PAD_ID] * (
            batch["src"].shape[1] - length
        )
    for row in range(batch["tgt_in"].shape[0]):
        length = int(batch["tgt_lens"][row]) - 1
        assert not batch["tgt_padding_mask"][row, :length].any()
        assert batch["tgt_padding_mask"][row, length:].all()

    # tgt_in is tgt_out shifted by one.  The shift is *per row*: the flattened
    # tensors are not one long sequence, so each row is compared against itself.
    compared = 0
    for row in range(batch["tgt_in"].shape[0]):
        width = int(batch["tgt_lens"][row]) - 1
        assert width >= 1
        assert batch["tgt_in"][row, 1:width].tolist() == (
            batch["tgt_out"][row, : width - 1].tolist()
        )
        # The teacher-forcing pair is complete: tgt_in holds BOS, tgt_out the
        # final EOS, so together they span the whole framed target.
        assert int(batch["tgt_in"][row, 0]) == BOS_ID
        assert int(batch["tgt_out"][row, width - 1]) == EOS_ID
        compared += width - 1
    assert compared > 0

    # ...and the same per-row shift holds for every batch of a full epoch.
    for extra in loaders["train"]:
        for row in range(extra["tgt_in"].shape[0]):
            width = int(extra["tgt_lens"][row]) - 1
            assert extra["tgt_in"][row, 1:width].tolist() == (
                extra["tgt_out"][row, : width - 1].tolist()
            )
            assert int(extra["tgt_out"][row, width - 1]) == EOS_ID
        assert extra["tgt_in"].shape[1] == max(extra["tgt_lens"].tolist()) - 1


def test_dataloaders_contain_a_batch_with_ragged_rows(config, tokenizer):
    """The corpus really does produce rows of differing length (mask is exercised)."""
    loaders = build_denoising_dataloaders(
        config, tokenizer, batch_size=4, num_workers=0, seed=7,
        max_src_len=48, max_tgt_len=48,
    )
    batch = next(iter(loaders["train"]))
    assert batch["src_padding_mask"].any(), "no padding to check"
    # More than one distinct real length, so the mask is not trivially uniform.
    assert len(set(batch["src_lens"].tolist())) > 1


def test_dataloaders_shuffle_is_reproducible(config, tokenizer):
    """Train shuffling depends only on ``seed``; val is unshuffled."""
    kwargs = dict(batch_size=4, num_workers=0, max_src_len=48, max_tgt_len=48)

    # A DataLoader's generator advances once per epoch (so successive epochs
    # reshuffle, which is what we want), so every comparison below builds a
    # *fresh* loader and reads exactly one epoch out of it.
    def fresh(seed):
        return build_denoising_dataloaders(config, tokenizer, seed=seed, **kwargs)

    def order(loader):
        return [int(batch["src_lens"][row]) for batch in loader
                for row in range(batch["src"].shape[0])]

    def batches(loader):
        return [tuple(tensor.flatten().tolist()) for batch in loader
                for tensor in (batch["src_lens"], batch["tgt_lens"])]

    same_a, same_b, other = fresh(7), fresh(7), fresh(1234)

    first_epoch = order(same_a["train"])
    assert first_epoch == order(same_b["train"])
    assert len(first_epoch) == len(same_a["train"].dataset)
    assert first_epoch != order(other["train"])

    # The same seed therefore gives a byte-identical epoch, batch for batch.
    assert batches(fresh(7)["train"]) == batches(fresh(7)["train"])
    assert batches(fresh(7)["train"]) != batches(fresh(1234)["train"])

    # Successive epochs of one loader are reshuffled, not replayed.
    reshuffled = fresh(7)
    assert order(reshuffled["train"]) != order(reshuffled["train"])

    # The validation split is deterministic and never shuffled.  (A different seed
    # *does* change the val content -- the corruption RNG is per (seed, index) --
    # so "unshuffled" is asserted on the sampler, which is what guarantees order.)
    val_a, val_b = fresh(7), fresh(7)
    assert order(val_a["val"]) == order(val_b["val"])
    assert batches(val_a["val"]) == batches(val_b["val"])
    assert isinstance(fresh(7)["val"].sampler, SequentialSampler)
    assert isinstance(fresh(7)["train"].sampler, RandomSampler)


def test_val_loader_covers_the_holdout(config, tokenizer):
    loaders = build_denoising_dataloaders(
        config, tokenizer, batch_size=4, num_workers=0, seed=7,
        max_src_len=48, max_tgt_len=48,
    )
    dataset = loaders["val"].dataset
    assert len(dataset) == len(range(0, 200, config.val_every))
    assert len(dataset) > 0
    produced = sum(batch["src"].shape[0] for batch in loaders["val"])
    assert produced == len(dataset)
    assert dataset.split == "val"
