"""Network-free tests for ``transformer_sym.data``.

Every test here uses either the deterministic synthetic task or files written
into pytest's ``tmp_path``.  The one HuggingFace test points ``hf_endpoint`` at a
``file://`` URL that cannot exist, which proves the raw-file cache is really used
without touching the network.  So ``python -m pytest tests/test_data.py -q``
passes with the cable unplugged.

Covered contract points (INTERFACES.md section 1):

* synthetic ``copy`` / ``reverse`` / ``sort`` determinism and token ranges,
* tokenizer special ids and deterministic vocabulary ordering
  (descending frequency, alphabetical ties, ``min_freq``, ``max_vocab``),
* the ``encode`` truncation rule that forces the last id to ``<eos>``,
* ``collate_batch`` key names, shapes, dtypes and mask polarity (True == PAD),
* a full ``build_dataloaders`` round trip asserting right-padding with ``pad_id``
  and the exact ``tgt_in`` / ``tgt_out`` one-step shift.
"""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import tempfile
import urllib.error
import uuid
from pathlib import Path

import pytest
import torch

from transformer_sym.config import DataConfig, ExperimentConfig
from transformer_sym.data import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    ParallelTextDataset,
    Tokenizer,
    build_dataloaders,
    build_tokenizers,
    collate_batch,
    download_hf_split,
    load_all_splits,
    load_local_split,
    load_parallel_data,
    synthetic_pairs,
)

# A tiny hand-built corpus with a known, unambiguous frequency profile.
CORPUS = ["b b b a a c d e"]

#: ``hf_endpoint`` that can never resolve, and is not even a network scheme, so
#: any test using it stays offline and fails fast if a cache is missing.
DEAD_ENDPOINT = "file:///nonexistent-mirror"


# --------------------------------------------------------------------------- #
# tmp_path, made robust against this machine's pytest basetemp
# --------------------------------------------------------------------------- #
def _safe_name(request: pytest.FixtureRequest) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)[:60] or "test"


@pytest.fixture
def tmp_path(tmp_path_factory, request):
    """Per-test scratch directory, usable even when pytest's basetemp is not.

    On this machine ``mkdir(mode=0o700)`` -- exactly how pytest creates
    ``%TEMP%/pytest-of-<user>`` -- yields a directory the process may no longer
    open, so the built-in ``tmp_path`` fixture raises ``PermissionError`` during
    setup before a single test body runs (and leaves an unremovable
    ``pytest-of-*`` behind).  The standard factory is therefore *tried first*,
    so behaviour and cleanup stay stock on any normal machine; only if it raises
    ``OSError`` do we substitute an ordinary directory (default permissions)
    under the system temp dir and delete it afterwards.
    """
    try:
        path = tmp_path_factory.mktemp(_safe_name(request))
    except OSError:
        path = None

    if path is not None:
        yield path
        return

    path = Path(tempfile.gettempdir()) / f"pytest-data-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)  # default mode: 0o700 is unusable here
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_synthetic_cfg(tmp_path: Path, task: str = "reverse", **overrides) -> DataConfig:
    """A small synthetic ``DataConfig`` rooted in ``tmp_path``."""
    cfg = DataConfig(
        source="synthetic",
        data_dir=str(tmp_path),
        synthetic_task=task,
        synthetic_seed=1234,
        synthetic_train_size=16,
        synthetic_val_size=5,
        synthetic_test_size=5,
        synthetic_vocab=5,
        synthetic_len=4,
        min_freq=1,
        max_vocab=100,
    )
    for key, value in overrides.items():
        assert hasattr(cfg, key), f"unknown DataConfig field {key!r}"
        setattr(cfg, key, value)
    return cfg


def write_hf_cache(
    tmp_path: Path, split: str, rows: list, repo: str = "bentrevett/multi30k"
) -> Path:
    """Write ``<tmp>/raw/<repo__name>/<split>.jsonl`` the way the loader caches."""
    raw = tmp_path / "raw" / repo.replace("/", "__")
    raw.mkdir(parents=True, exist_ok=True)
    path = raw / f"{split}.jsonl"
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# package surface
# --------------------------------------------------------------------------- #
def test_public_api_exports() -> None:
    import transformer_sym.data as data

    for name in data.__all__:
        assert hasattr(data, name), f"__all__ lists missing name {name!r}"
    assert (data.PAD_ID, data.UNK_ID, data.BOS_ID, data.EOS_ID) == (0, 1, 2, 3)
    assert data.SPECIAL_TOKENS == ["<pad>", "<unk>", "<bos>", "<eos>"]
    assert (
        Tokenizer.pad_id,
        Tokenizer.unk_id,
        Tokenizer.bos_id,
        Tokenizer.eos_id,
    ) == (PAD_ID, UNK_ID, BOS_ID, EOS_ID)


# --------------------------------------------------------------------------- #
# synthetic task
# --------------------------------------------------------------------------- #
def test_synthetic_copy_is_identity_and_deterministic(tmp_path: Path) -> None:
    cfg = make_synthetic_cfg(tmp_path, task="copy")
    first = synthetic_pairs(cfg, "train")
    second = synthetic_pairs(cfg, "train")

    assert first == second  # pure function of the config
    assert len(first) == cfg.synthetic_train_size
    for src, tgt in first:
        assert src == tgt
        tokens = src.split()
        assert len(tokens) == cfg.synthetic_len
        # decimal strings of ids in [0, synthetic_vocab)
        assert all(t.isdigit() and 0 <= int(t) < cfg.synthetic_vocab for t in tokens)

    # Generating another split first must not shift this split's stream.
    other = make_synthetic_cfg(tmp_path, task="copy")
    synthetic_pairs(other, "test")
    assert synthetic_pairs(other, "train") == first

    # Splits and seeds are independent streams.
    assert synthetic_pairs(cfg, "val") != first[: cfg.synthetic_val_size]
    cfg.synthetic_seed = 99
    assert synthetic_pairs(cfg, "train") != first


def test_synthetic_reverse_is_reversed(tmp_path: Path) -> None:
    cfg = make_synthetic_cfg(tmp_path, task="reverse")
    pairs = synthetic_pairs(cfg, "val")
    assert len(pairs) == cfg.synthetic_val_size
    for src, tgt in pairs:
        assert tgt.split() == src.split()[::-1]
    # reverse really differs from copy for (almost) every pair
    assert any(src != tgt for src, tgt in pairs)


def test_synthetic_sort_is_numerically_ascending(tmp_path: Path) -> None:
    # Single-digit ids: numeric and lexicographic order agree, so this is an
    # unambiguous "ascending" check.
    cfg = make_synthetic_cfg(tmp_path, task="sort", synthetic_vocab=10)
    for src, tgt in synthetic_pairs(cfg, "val"):
        assert tgt.split() == sorted(src.split())
        assert tgt.split() == sorted(src.split(), key=int)

    # Multi-digit ids: the tokens are ids, so "10" must sort after "9".
    cfg = make_synthetic_cfg(
        tmp_path,
        task="sort",
        synthetic_vocab=20,
        synthetic_len=8,
        synthetic_val_size=50,
    )
    ambiguous = 0
    for src, tgt in synthetic_pairs(cfg, "val"):
        tokens = src.split()
        assert [int(t) for t in tgt.split()] == sorted(int(t) for t in tokens)
        if sorted(tokens) != sorted(tokens, key=int):
            ambiguous += 1  # numeric order differs from lexicographic here
    assert ambiguous > 0, "test corpus never exercises numeric-vs-string ordering"


def test_synthetic_validation_and_zero_size(tmp_path: Path) -> None:
    cfg = make_synthetic_cfg(tmp_path)
    with pytest.raises(ValueError, match="unknown synthetic split"):
        synthetic_pairs(cfg, "dev")

    cfg.synthetic_task = "shuffle"
    with pytest.raises(ValueError, match="synthetic_task"):
        synthetic_pairs(cfg, "train")

    cfg = make_synthetic_cfg(tmp_path, synthetic_test_size=0)
    assert synthetic_pairs(cfg, "test") == []

    cfg = make_synthetic_cfg(tmp_path, synthetic_val_size=-1)
    with pytest.raises(ValueError, match=">= 0"):
        synthetic_pairs(cfg, "val")

    cfg = make_synthetic_cfg(tmp_path, synthetic_vocab=0)
    with pytest.raises(ValueError, match="synthetic_vocab"):
        synthetic_pairs(cfg, "val")


def test_synthetic_pairs_are_stable_strings(tmp_path: Path) -> None:
    """The exact bytes for a fixed seed (guards against accidental reseeding)."""
    cfg = make_synthetic_cfg(tmp_path, task="reverse", synthetic_val_size=2,
                             synthetic_seed=1234, synthetic_vocab=5, synthetic_len=4)
    pairs = synthetic_pairs(cfg, "val")
    assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)
    assert synthetic_pairs(cfg, "val") == pairs


# --------------------------------------------------------------------------- #
# tokenizer
# --------------------------------------------------------------------------- #
def test_special_ids_and_vocabulary_ordering() -> None:
    tok = Tokenizer.build(CORPUS, mode="word", min_freq=1, max_vocab=100)

    assert tok.itos[:4] == SPECIAL_TOKENS
    # descending frequency (b=3, a=2) then alphabetical (c, d, e)
    assert tok.itos[4:] == ["b", "a", "c", "d", "e"]
    assert tok.vocab_size == 9 == len(tok.itos)
    assert tok.stoi["b"] == 4 and tok.stoi["e"] == 8
    assert tok.stoi["<pad>"] == PAD_ID and tok.stoi["<eos>"] == EOS_ID

    # Deterministic regardless of the input order.
    assert Tokenizer.build(list(reversed(CORPUS)), min_freq=1, max_vocab=100).itos == tok.itos
    shuffled = CORPUS * 3
    random.Random(0).shuffle(shuffled)
    assert Tokenizer.build(shuffled, min_freq=1, max_vocab=100).itos == tok.itos
    assert Tokenizer.build(iter(CORPUS), min_freq=1, max_vocab=100).itos == tok.itos


def test_min_freq_and_max_vocab_budget() -> None:
    tok = Tokenizer.build(CORPUS, min_freq=2, max_vocab=100)
    assert tok.itos == SPECIAL_TOKENS + ["b", "a"]

    tok = Tokenizer.build(CORPUS, min_freq=3, max_vocab=100)
    assert tok.itos == SPECIAL_TOKENS + ["b"]

    # max_vocab counts the four specials.
    tok = Tokenizer.build(CORPUS, min_freq=1, max_vocab=6)
    assert tok.vocab_size == 6 and tok.itos == SPECIAL_TOKENS + ["b", "a"]

    tok = Tokenizer.build(CORPUS, min_freq=1, max_vocab=4)
    assert tok.itos == SPECIAL_TOKENS and tok.vocab_size == 4

    with pytest.raises(ValueError, match="max_vocab"):
        Tokenizer.build(CORPUS, min_freq=1, max_vocab=3)
    with pytest.raises(ValueError, match="min_freq"):
        Tokenizer.build(CORPUS, min_freq=0)
    with pytest.raises(ValueError, match="mode"):
        Tokenizer.build(CORPUS, mode="bpe")

    # A non-special itos[0] is rejected: ids 0..3 are frozen.
    with pytest.raises(ValueError, match="SPECIAL_TOKENS|exactly"):
        Tokenizer(["a", "<unk>", "<bos>", "<eos>"])


def test_tokenize_word_and_char_modes() -> None:
    tok = Tokenizer(SPECIAL_TOKENS + ["hallo", "welt", "!", ","], mode="word")
    assert tok.tokenize("Hallo, Welt!") == ["hallo", ",", "welt", "!"]
    assert tok.tokenize("MÄNNER") == ["männer"]  # re.UNICODE keeps accents in words
    tok.lowercase = False
    assert tok.tokenize("Hallo") == ["Hallo"]

    char = Tokenizer(SPECIAL_TOKENS + ["h", "e", "l", "o", "i", "!", " "], mode="char")
    assert char.tokenize("hi!") == ["h", "i", "!"]
    assert char.tokenize("HELLO") == ["h", "e", "l", "l", "o"]
    char.lowercase = False
    assert char.tokenize("HELLO") == ["H", "E", "L", "L", "O"]


def test_encode_framing_and_unknown_tokens() -> None:
    tok = Tokenizer(SPECIAL_TOKENS + ["a", "b", "c", "d", "e"], mode="word")
    a, b = tok.stoi["a"], tok.stoi["b"]

    assert tok.encode("a b") == [a, b, EOS_ID]  # add_eos defaults to True
    assert tok.encode("a b", add_eos=False) == [a, b]
    assert tok.encode("a b", add_bos=True) == [BOS_ID, a, b, EOS_ID]
    assert tok.encode("a b", add_bos=True, add_eos=False) == [BOS_ID, a, b]
    assert tok.encode("zzz", add_eos=False) == [UNK_ID]  # out-of-vocab -> <unk>
    assert tok.encode("") == [EOS_ID]


def test_encode_truncation_forces_final_eos() -> None:
    tok = Tokenizer(SPECIAL_TOKENS + ["a", "b", "c", "d", "e"], mode="word")
    a, b = tok.stoi["a"], tok.stoi["b"]

    full = tok.encode("a b c d e", add_eos=True)
    assert full == [a, b, tok.stoi["c"], tok.stoi["d"], tok.stoi["e"], EOS_ID]

    # No clipping at max_len == len, clipping at max_len < len.
    assert tok.encode("a b c d e", add_eos=True, max_len=len(full)) == full
    assert tok.encode("a b c d e", add_eos=True, max_len=3) == [a, b, EOS_ID]
    assert tok.encode("a b c d e", add_bos=True, add_eos=True, max_len=4) == [
        BOS_ID,
        a,
        b,
        EOS_ID,
    ]
    # Without add_eos the plain prefix is kept -- no forced <eos>.
    assert tok.encode("a b c d e", add_eos=False, max_len=2) == [a, b]
    assert tok.encode("a b c d e", add_bos=True, add_eos=False, max_len=2) == [BOS_ID, a]
    # max_len=0 disables clipping; a shorter list is never padded here.
    assert tok.encode("a b", add_eos=True, max_len=99) == [a, b, EOS_ID]
    assert tok.encode("a b", add_eos=True, max_len=1) == [EOS_ID]

    char = Tokenizer(SPECIAL_TOKENS + ["h", "e", "l", "o"], mode="char")
    assert char.encode("hello", add_bos=True, add_eos=True, max_len=3) == [
        BOS_ID,
        char.stoi["h"],
        EOS_ID,
    ]


def test_decode_skips_specials_and_validates_ids() -> None:
    tok = Tokenizer(SPECIAL_TOKENS + ["a", "b"], mode="word")
    assert tok.decode([BOS_ID, 4, 5, EOS_ID]) == "a b"
    assert tok.decode([BOS_ID, 4, PAD_ID, 5, EOS_ID]) == "a b"
    assert tok.decode([BOS_ID, 4, EOS_ID], skip_special=False) == "<bos> a <eos>"
    assert tok.decode([4, 4]) == "a a"
    with pytest.raises(ValueError, match="out of range"):
        tok.decode([0, 99])
    with pytest.raises(ValueError, match="out of range"):
        tok.decode([-1])

    char = Tokenizer(SPECIAL_TOKENS + ["h", "i", "!"], mode="char")
    assert char.decode([BOS_ID, 4, 5, 6, EOS_ID]) == "hi!"
    assert char.decode(torch.tensor([BOS_ID, 4, 5, 6, EOS_ID])) == "hi!"  # tensors work


def test_tokenizer_save_load_roundtrip(tmp_path: Path) -> None:
    tok = Tokenizer.build(CORPUS, min_freq=1, max_vocab=100)
    path = tok.save(tmp_path / "tok.json")
    assert isinstance(path, Path) and path.exists()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["specials"] == SPECIAL_TOKENS
    assert payload["itos"][:4] == SPECIAL_TOKENS
    assert payload["mode"] == "word"

    loaded = Tokenizer.load(path)
    assert loaded.itos == tok.itos
    assert loaded.mode == tok.mode
    assert loaded.lowercase == tok.lowercase
    assert loaded.encode("a b c") == tok.encode("a b c")

    # A directory is accepted too (tokenizer.json inside).
    folder = tmp_path / "vocab"
    folder.mkdir()
    tok.save(folder)
    assert (folder / "tokenizer.json").exists()
    assert Tokenizer.load(folder).itos == tok.itos
    assert Tokenizer.load(folder / "tokenizer.json").itos == tok.itos

    cased = Tokenizer.build(["A b"], min_freq=1, lowercase=False)
    assert Tokenizer.load(cased.save(tmp_path / "cased.json")).lowercase is False

    with pytest.raises(FileNotFoundError):
        Tokenizer.load(tmp_path / "missing.json")


def test_build_tokenizers_uses_train_split(tmp_path: Path) -> None:
    cfg = make_synthetic_cfg(tmp_path, task="reverse")
    pairs_by_split = load_all_splits(cfg)
    src_tok, tgt_tok = build_tokenizers(cfg, pairs_by_split)

    assert isinstance(src_tok, Tokenizer) and isinstance(tgt_tok, Tokenizer)
    assert src_tok.itos[:4] == SPECIAL_TOKENS and tgt_tok.itos[:4] == SPECIAL_TOKENS
    assert src_tok.vocab_size == tgt_tok.vocab_size  # same digit alphabet here
    # source and target vocabularies are separate objects
    assert src_tok is not tgt_tok

    with pytest.raises(ValueError, match="train"):
        build_tokenizers(cfg, {"val": pairs_by_split["val"]})
    with pytest.raises(ValueError, match="train"):
        build_tokenizers(cfg, {})


# --------------------------------------------------------------------------- #
# dataset + collate
# --------------------------------------------------------------------------- #
def test_dataset_framing_and_clipping() -> None:
    src_tok = Tokenizer(SPECIAL_TOKENS + ["a", "b", "c"], "word")
    tgt_tok = Tokenizer(SPECIAL_TOKENS + ["x", "y", "z"], "word")
    dataset = ParallelTextDataset(
        [("a b c", "x y z"), ("a", "x")],
        src_tok,
        tgt_tok,
        max_src_len=2,
        max_tgt_len=3,
    )
    assert len(dataset) == 2

    src_ids, tgt_ids = dataset[0]
    # source: no BOS, always ends with EOS, clipped to max_src_len with an EOS.
    assert src_ids == [src_tok.stoi["a"], EOS_ID]
    # target: [BOS, x, y, z, EOS] clipped to 3 -> [BOS, x, EOS].
    assert tgt_ids == [BOS_ID, tgt_tok.stoi["x"], EOS_ID]

    src_ids, tgt_ids = dataset[1]
    assert src_ids == [src_tok.stoi["a"], EOS_ID]
    assert tgt_ids == [BOS_ID, tgt_tok.stoi["x"], EOS_ID]

    # No clipping when the budgets are 0.
    free = ParallelTextDataset([("a b c", "x y z")], src_tok, tgt_tok)
    assert free[0][0] == [src_tok.stoi["a"], src_tok.stoi["b"], src_tok.stoi["c"], EOS_ID]
    assert free[0][1] == [
        BOS_ID,
        tgt_tok.stoi["x"],
        tgt_tok.stoi["y"],
        tgt_tok.stoi["z"],
        EOS_ID,
    ]

    src_lens, tgt_lens = dataset.lengths()
    assert src_lens == [len(dataset[i][0]) for i in range(len(dataset))]
    assert tgt_lens == [len(dataset[i][1]) for i in range(len(dataset))]
    assert src_lens == [2, 2] and tgt_lens == [3, 3]
    # lengths() hands out copies, so callers cannot corrupt the cache
    src_lens.append(99)
    assert dataset.lengths()[0] == [2, 2]


def test_collate_batch_keys_shapes_dtypes_and_mask_polarity() -> None:
    batch = [
        ([5, 6, EOS_ID], [BOS_ID, 7, 8, EOS_ID]),
        ([9, EOS_ID], [BOS_ID, 10, EOS_ID]),
    ]
    out = collate_batch(batch, pad_id=PAD_ID)

    assert set(out) == {
        "src",
        "tgt_in",
        "tgt_out",
        "src_padding_mask",
        "tgt_padding_mask",
        "src_lens",
        "tgt_lens",
    }
    assert out["src"].shape == (2, 3) and out["src"].dtype == torch.long
    assert out["tgt_in"].shape == (2, 3) and out["tgt_in"].dtype == torch.long
    assert out["tgt_out"].shape == (2, 3) and out["tgt_out"].dtype == torch.long
    assert out["src_padding_mask"].shape == (2, 3)
    assert out["src_padding_mask"].dtype == torch.bool
    assert out["tgt_padding_mask"].shape == (2, 3)
    assert out["tgt_padding_mask"].dtype == torch.bool
    assert out["src_lens"].shape == (2,) and out["src_lens"].dtype == torch.long
    assert out["tgt_lens"].shape == (2,) and out["tgt_lens"].dtype == torch.long

    assert out["src"].tolist() == [[5, 6, EOS_ID], [9, EOS_ID, PAD_ID]]
    assert out["tgt_in"].tolist() == [[BOS_ID, 7, 8], [BOS_ID, 10, PAD_ID]]
    assert out["tgt_out"].tolist() == [[7, 8, EOS_ID], [10, EOS_ID, PAD_ID]]
    assert out["src_lens"].tolist() == [3, 2]
    assert out["tgt_lens"].tolist() == [4, 3]

    # True == PAD: exactly the right-padding slots are marked, nothing else.
    assert out["src_padding_mask"].tolist() == [
        [False, False, False],
        [False, False, True],
    ]
    assert out["tgt_padding_mask"].tolist() == [
        [False, False, False],
        [False, False, True],
    ]
    for key, mask_key in (("src", "src_padding_mask"), ("tgt_in", "tgt_padding_mask")):
        pads = out[key][out[mask_key]]
        assert pads.numel() > 0 and (pads == PAD_ID).all()
        assert (out[key][~out[mask_key]] != PAD_ID).all()

    with pytest.raises(ValueError, match="empty batch"):
        collate_batch([])
    with pytest.raises(ValueError, match="at least 2"):
        collate_batch([([1, EOS_ID], [BOS_ID])])

    # A single-item batch is just an unpadded batch.
    single = collate_batch([([5, EOS_ID], [BOS_ID, 7, EOS_ID])])
    assert single["src"].tolist() == [[5, EOS_ID]]
    assert single["tgt_in"].tolist() == [[BOS_ID, 7]]
    assert single["tgt_out"].tolist() == [[7, EOS_ID]]


def test_build_dataloaders_roundtrip_on_synthetic(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.data = make_synthetic_cfg(tmp_path, task="reverse")
    cfg.model.max_seq_len = 32
    cfg.train.batch_size = 4
    cfg.train.num_workers = 0
    cfg.train.seed = 7
    cfg.validate()

    pairs_by_split = load_all_splits(cfg.data)
    assert list(pairs_by_split) == ["train", "val", "test"]
    src_tok, tgt_tok = build_tokenizers(cfg.data, pairs_by_split)
    loaders = build_dataloaders(cfg, src_tok, tgt_tok)

    assert list(loaders) == ["train", "val", "test"]
    assert len(loaders["train"].dataset) == cfg.data.synthetic_train_size == 16
    assert len(loaders["train"]) == 4  # 16 / 4, drop_last=False
    assert len(loaders["val"]) == 2  # 5 pairs -> 4 + 1 (short batch kept)
    assert loaders["train"].batch_size == 4

    batch = next(iter(loaders["val"]))
    assert set(batch) == {
        "src",
        "tgt_in",
        "tgt_out",
        "src_padding_mask",
        "tgt_padding_mask",
        "src_lens",
        "tgt_lens",
    }
    rows, src_width = batch["src"].shape
    assert rows == 4
    assert batch["tgt_in"].shape == batch["tgt_out"].shape
    assert batch["src_padding_mask"].shape == batch["src"].shape
    assert batch["tgt_padding_mask"].shape == batch["tgt_in"].shape

    # 1. padding positions carry pad_id, and only those positions are masked.
    for key, mask_key in (
        ("src", "src_padding_mask"),
        ("tgt_in", "tgt_padding_mask"),
        ("tgt_out", "tgt_padding_mask"),
    ):
        mask = batch[mask_key]
        assert (batch[key][mask] == PAD_ID).all(), f"{key} padding is not pad_id"
        assert (batch[key][~mask] != PAD_ID).all(), f"{key} has pad_id inside the mask"
    # 2. the mask is exactly right-padding at the reported length.
    for row in range(rows):
        src_len, tgt_len = int(batch["src_lens"][row]), int(batch["tgt_lens"][row])
        assert not batch["src_padding_mask"][row, :src_len].any()
        assert batch["src_padding_mask"][row, src_len:].all()
        assert not batch["tgt_padding_mask"][row, : tgt_len - 1].any()
        assert batch["tgt_padding_mask"][row, tgt_len - 1 :].all()
    assert batch["src_lens"].max() <= src_width

    # 3. tgt_in / tgt_out are the one-step shift of the same target, and the
    #    source keeps its EOS framing.  val is unshuffled, so row order matches.
    dataset = loaders["val"].dataset
    for row in range(rows):
        src_ids, tgt_ids = dataset[row]
        assert int(batch["src_lens"][row]) == len(src_ids)
        assert int(batch["tgt_lens"][row]) == len(tgt_ids)
        assert batch["src"][row, : len(src_ids)].tolist() == src_ids
        assert batch["src"][row, len(src_ids) - 1] == EOS_ID
        assert src_ids[0] != BOS_ID  # the encoder side has no BOS
        keep = len(tgt_ids) - 1
        assert batch["tgt_in"][row, :keep].tolist() == tgt_ids[:-1]
        assert batch["tgt_out"][row, :keep].tolist() == tgt_ids[1:]
        assert tgt_ids[0] == BOS_ID and tgt_ids[-1] == EOS_ID
        assert batch["tgt_in"][row, 0] == BOS_ID
        assert batch["tgt_out"][row, keep - 1] == EOS_ID

    # 4. no <unk> leaked in: every token was in the training vocabulary.
    assert (batch["src"] != UNK_ID).all()
    assert (batch["tgt_in"] != UNK_ID).all()
    assert (batch["tgt_out"] != UNK_ID).all()

    # 5. clipping budget comes from model.max_seq_len when data.max_*_len is 0.
    widths = {len(loaders[s].dataset[0][0]) for s in loaders}
    assert max(widths) <= cfg.model.max_seq_len + 1  # +1 for the EOS

    # 6. train shuffling is seeded => two builds give the same order.
    other = build_dataloaders(cfg, src_tok, tgt_tok)
    first_repeat = next(iter(other["train"]))
    assert torch.equal(first_repeat["src"], next(iter(loaders["train"]))["src"])


def test_build_dataloaders_skips_empty_splits(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.data = make_synthetic_cfg(tmp_path, task="copy", synthetic_test_size=0)
    cfg.train.batch_size = 4
    cfg.validate()
    pairs = load_all_splits(cfg.data)
    src_tok, tgt_tok = build_tokenizers(cfg.data, pairs)

    loaders = build_dataloaders(cfg, src_tok, tgt_tok)
    assert "test" not in loaders and set(loaders) == {"train", "val"}

    cfg.data.synthetic_val_size = 0
    cfg.data.synthetic_train_size = 0
    pairs = load_all_splits(cfg.data)
    with pytest.raises(ValueError, match="no non-empty split"):
        build_dataloaders(cfg, src_tok, tgt_tok, splits=("train", "val"))


def test_dataloaders_respect_data_length_overrides(tmp_path: Path) -> None:
    cfg = ExperimentConfig()
    cfg.data = make_synthetic_cfg(tmp_path, task="copy", synthetic_len=8)
    cfg.data.max_src_len = 3
    cfg.data.max_tgt_len = 4
    cfg.model.max_seq_len = 32
    cfg.train.batch_size = 2
    cfg.validate()
    pairs = load_all_splits(cfg.data)
    src_tok, tgt_tok = build_tokenizers(cfg.data, pairs)
    loaders = build_dataloaders(cfg, src_tok, tgt_tok)
    src_ids, tgt_ids = loaders["val"].dataset[0]
    assert len(src_ids) == 3 and src_ids[-1] == EOS_ID
    assert len(tgt_ids) == 4 and tgt_ids[0] == BOS_ID and tgt_ids[-1] == EOS_ID


# --------------------------------------------------------------------------- #
# local files
# --------------------------------------------------------------------------- #
def test_local_tsv_loader(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir(parents=True)
    (local / "train.tsv").write_text(
        "hallo\tworld\nzwei\tZeilen\n\n", encoding="utf-8"
    )
    cfg = DataConfig(source="local", data_dir=str(tmp_path))

    expected = [("hallo", "world"), ("zwei", "Zeilen")]
    assert load_local_split(cfg, "train") == expected
    assert load_parallel_data(cfg, "train") == expected
    assert load_all_splits(cfg, ("train",)) == {"train": expected}


def test_local_column_selection_and_jsonl_sibling(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir(parents=True)
    (local / "test.tsv").write_text("t1\ts1\tA\nt2\ts2\tB\n", encoding="utf-8")
    cfg = DataConfig(source="local", data_dir=str(tmp_path), local_src_col=2, local_tgt_col=1)
    assert load_local_split(cfg, "test") == [("A", "s1"), ("B", "s2")]

    (local / "val.jsonl").write_text(
        json.dumps({"de": "hallo", "en": "hello"}, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    plain = DataConfig(source="local", data_dir=str(tmp_path))
    assert load_local_split(plain, "val") == [("hallo", "hello")]


def test_local_loader_errors(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir(parents=True)
    (local / "empty.tsv").write_text("\n\n", encoding="utf-8")
    (local / "short.tsv").write_text("only-one-column\n", encoding="utf-8")
    cfg = DataConfig(source="local", data_dir=str(tmp_path))

    with pytest.raises(ValueError, match="no local data"):
        load_local_split(cfg, "missing")
    with pytest.raises(ValueError, match="no parallel sentence pairs"):
        load_local_split(cfg, "empty")
    with pytest.raises(ValueError, match="column"):
        load_local_split(cfg, "short")


# --------------------------------------------------------------------------- #
# HuggingFace mirror -- offline, via the raw cache and a dead endpoint
# --------------------------------------------------------------------------- #
def test_hf_split_reads_cache_and_applies_limits(tmp_path: Path) -> None:
    rows = [{"de": f"de{i}", "en": f"en{i}"} for i in range(5)]
    write_hf_cache(tmp_path, "train", rows)
    write_hf_cache(tmp_path, "val", rows)
    write_hf_cache(tmp_path, "test", rows)
    write_hf_cache(tmp_path, "empty", [])

    cfg = DataConfig(source="hf", data_dir=str(tmp_path), hf_endpoint=DEAD_ENDPOINT)
    # The endpoint cannot be reached, so these can only succeed from the cache.
    expected = [(f"de{i}", f"en{i}") for i in range(5)]
    assert download_hf_split(cfg, "val") == expected
    assert load_parallel_data(cfg, "train") == expected

    cfg.max_val_samples = 2
    cfg.max_train_samples = 3
    assert download_hf_split(cfg, "val") == expected[:2]
    assert download_hf_split(cfg, "train") == expected[:3]
    assert download_hf_split(cfg, "test") == expected  # test is unlimited

    cfg.hf_files["empty"] = "empty.jsonl"
    with pytest.raises(ValueError, match="no parallel sentence pairs"):
        download_hf_split(cfg, "empty")
    with pytest.raises(ValueError, match="hf_files"):
        download_hf_split(cfg, "dev")

    assert not (tmp_path / "raw" / "bentrevett__multi30k" / "val.jsonl.part").exists()


def test_hf_bad_json_reports_the_cache_file(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "bentrevett__multi30k"
    raw.mkdir(parents=True)
    (raw / "val.jsonl").write_text("<html>403 Forbidden</html>\n", encoding="utf-8")
    cfg = DataConfig(source="hf", data_dir=str(tmp_path), hf_endpoint=DEAD_ENDPOINT)
    with pytest.raises(ValueError, match="not valid JSON"):
        download_hf_split(cfg, "val")


def test_hf_failure_falls_back_to_synthetic(tmp_path: Path, capsys) -> None:
    cfg = DataConfig(
        source="hf",
        data_dir=str(tmp_path),
        hf_endpoint=DEAD_ENDPOINT,
        fallback_to_synthetic=True,
        synthetic_task="copy",
        synthetic_seed=5,
        synthetic_val_size=3,
        synthetic_vocab=5,
        synthetic_len=4,
        min_freq=1,
    )
    pairs = load_parallel_data(cfg, "val")
    assert pairs == synthetic_pairs(cfg, "val")
    assert len(pairs) == 3
    warning = capsys.readouterr().out
    assert "WARNING" in warning
    assert "fallback_to_synthetic" in warning
    assert "val" in warning

    cfg.fallback_to_synthetic = False
    with pytest.raises(urllib.error.URLError):
        load_parallel_data(cfg, "val")


def test_unknown_source_raises(tmp_path: Path) -> None:
    cfg = DataConfig(source="postgres", data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="unknown data.source"):
        load_parallel_data(cfg, "train")
