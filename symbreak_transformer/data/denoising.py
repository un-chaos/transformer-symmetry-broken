"""
Span corruption (the T5 "denoising" objective) for raw monolingual text.

An encoder-decoder cannot be trained on translations it does not have.  FineWeb-Edu
is **monolingual English**, so the standard way to use it is corrupt-and-reconstruct:
the encoder reads a document with a few spans replaced by sentinel tokens and the
decoder has to write the removed spans back, in order:

    original  : the quick brown fox jumps over the lazy dog
    encoder in: the quick <extra_id_0> over the <extra_id_1> dog
    decoder out: [BOS] <extra_id_0> brown fox jumps <extra_id_1> lazy [EOS]

That turns any pile of text into encoder/decoder pairs, which is why this module
is what makes the 10-billion-token corpus usable by this model.

Everything here is deterministic given ``(seed, document index)``: the same
configuration corrupts the same document the same way on every run, and the
global RNG is never touched.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from ..config import DataConfig
from .bpe import BPETokenizer
from .dataset import collate_batch
from .tokenizer import BOS_ID, EOS_ID

__all__ = [
    "Corruption",
    "corrupt_spans",
    "FineWebDenoisingDataset",
    "build_denoising_dataloaders",
]

#: Row groups kept in memory.  FineWeb shards use ~1000-row groups, so a handful
#: is a few MB while random access stays fast.
_ROW_GROUP_CACHE = 4

#: Below this many tokens a document cannot be corrupted meaningfully.
_MIN_TOKENS = 8


@dataclass
class Corruption:
    """One corrupted example, already framed for the model.

    ``src_ids`` has **no BOS and ends with EOS**; ``tgt_ids`` is
    ``[BOS] ... [EOS]``.  That is exactly what ``collate_batch`` expects, so the
    rest of the pipeline (padding, the model, the loss) needs no special case.
    """

    src_ids: List[int] = field(default_factory=list)
    tgt_ids: List[int] = field(default_factory=list)
    #: The removed token spans, in order, for inspection and tests.
    spans: List[List[int]] = field(default_factory=list)


def _plan_spans(
    n_tokens: int, noise_density: float, mean_span_length: float, rng: random.Random,
    max_spans: int,
) -> List[Tuple[int, int]]:
    """
    Choose the ``[start, end)`` spans to remove.

    ``noise_density`` fixes roughly how many tokens disappear and
    ``mean_span_length`` how long each span is, so the span count follows from the
    two.  Starts are drawn without replacement and overlapping starts are
    skipped, which keeps the spans disjoint and the sentinel order meaningful.
    """
    if n_tokens <= 0:
        return []
    target_tokens = max(1, int(round(n_tokens * noise_density)))
    n_spans = max(1, int(round(target_tokens / max(mean_span_length, 1.0))))
    # Each span needs one sentinel, and the target uses one more as a terminator.
    n_spans = max(1, min(n_spans, max_spans, n_tokens))

    starts = sorted(rng.sample(range(n_tokens), min(n_spans, n_tokens)))
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for start in starts:
        if start < cursor:
            continue  # would overlap the previous span
        if mean_span_length <= 1.0:
            length = 1
        else:
            # Exponential with the requested mean, then clamped: this is what
            # makes `mean_span_length` an average rather than a fixed width.
            length = int(round(rng.expovariate(1.0 / mean_span_length)))
            length = max(1, length)
        length = min(length, n_tokens - start)
        if length <= 0:
            continue
        spans.append((start, start + length))
        cursor = start + length
    if not spans:  # every start collided; fall back to a single short span
        start = rng.randrange(n_tokens)
        spans = [(start, min(start + 1, n_tokens))]
    return spans


def corrupt_spans(
    ids: Sequence[int],
    tokenizer: BPETokenizer,
    noise_density: float = 0.15,
    mean_span_length: float = 3.0,
    max_src_len: int = 0,
    max_tgt_len: int = 0,
    rng: Optional[random.Random] = None,
) -> Corruption:
    """
    Turn one token sequence into an encoder/decoder denoising pair.

    Args:
        ids: the document's token ids (no special tokens).
        tokenizer: supplies the sentinel ids (:meth:`BPETokenizer.sentinel_id`).
        noise_density: fraction of tokens to remove (T5 uses 0.15).
        mean_span_length: average removed-span length (T5 uses 3).
        max_src_len: truncate the encoder side to this many ids, keeping EOS.
        max_tgt_len: truncate the decoder side to this many ids, keeping BOS/EOS.
        rng: source of randomness; pass a seeded one for reproducibility.

    Returns:
        A :class:`Corruption`.  Short documents are handled by shrinking the span
        rather than by raising, so a dataset never dies on a two-word document.
    """
    rng = rng or random.Random(0)
    ids = [int(i) for i in ids]

    if len(ids) < _MIN_TOKENS:
        # Too short to corrupt properly: hide a single token so the example is
        # still a valid (if easy) denoising pair.
        spans = [(0, 1)] if ids else []
    else:
        spans = _plan_spans(
            len(ids), noise_density, mean_span_length, rng, tokenizer.num_sentinels - 1
        )

    src: List[int] = []
    tgt: List[int] = [BOS_ID]
    cursor = 0
    for index, (start, end) in enumerate(spans):
        sentinel = tokenizer.sentinel_id(index)
        src.extend(ids[cursor:start])
        src.append(sentinel)
        tgt.append(sentinel)
        tgt.extend(ids[start:end])
        cursor = end
    src.extend(ids[cursor:])
    # T5 closes the target with the next sentinel, which is also the stop symbol
    # the decoder learns to emit.
    tgt.append(tokenizer.sentinel_id(len(spans)))
    tgt.append(EOS_ID)

    if max_src_len and len(src) + 1 > max_src_len:
        src = src[: max_src_len - 1]
    src.append(EOS_ID)
    if max_tgt_len and len(tgt) > max_tgt_len:
        # Keep the framing: BOS at the front, EOS at the back.
        tgt = tgt[: max_tgt_len - 1] + [EOS_ID] if max_tgt_len > 1 else [EOS_ID]

    return Corruption(src_ids=src, tgt_ids=tgt, spans=[ids[s:e] for s, e in spans])


class FineWebDenoisingDataset(torch.utils.data.Dataset):
    """
    A map-style dataset of denoising pairs streamed out of the parquet shards.

    Map-style (rather than iterable) because the trainer computes
    ``steps_per_epoch`` from ``len(loader)``.  The document index is built from
    parquet **metadata only** -- ``num_row_groups`` and each group's row count --
    so constructing this never reads the 28 GB of text.  ``__getitem__`` reads one
    row group at a time and caches the last few, which keeps peak memory bounded
    while still allowing shuffled random access.

    The corpus has no train/validation split, so every ``cfg.val_every``-th
    document goes to ``split="val"`` and the rest to ``"train"``.
    """

    def __init__(
        self,
        cfg: DataConfig,
        tokenizer: BPETokenizer,
        split: str = "train",
        max_src_len: int = 512,
        max_tgt_len: int = 512,
        seed: int = 42,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.split = split
        self.max_src_len = int(max_src_len)
        self.max_tgt_len = int(max_tgt_len)
        self.seed = int(seed)

        self._entries: List[Tuple[Path, int, int]] = self._build_index(cfg)
        val_every = max(1, int(cfg.val_every))
        # The cap applies before the split, so `max_documents=N` means
        # "consider the first N documents, then hold some out".
        considered = len(self._entries)
        if cfg.max_documents:
            considered = min(considered, int(cfg.max_documents))
        if split == "val":
            self._rows = [i for i in range(considered) if i % val_every == 0]
        else:
            self._rows = [i for i in range(considered) if i % val_every != 0]
        self._cache: Dict[Tuple[str, int], List[str]] = {}
        self._cache_order: List[Tuple[str, int]] = []

    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_index(cfg: DataConfig) -> List[Tuple[Path, int, int]]:
        """``[(shard, row_group, row_within_group)]`` from parquet metadata only."""
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise ImportError(
                "reading the FineWeb-Edu shards needs pyarrow: pip install pyarrow"
            ) from exc

        directory = Path(str(cfg.fineweb_dir)).expanduser()
        shards = sorted(directory.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(
                f"no parquet shards under {directory}; download the corpus first "
                f"with menu item 7) in run.py (or run.bat)"
            )
        entries: List[Tuple[Path, int, int]] = []
        for shard in shards:
            metadata = pq.ParquetFile(shard).metadata
            if cfg.text_column not in (metadata.schema.names or []):
                raise KeyError(
                    f"{shard.name} has no column {cfg.text_column!r}; "
                    f"available: {list(metadata.schema.names or [])}"
                )
            for group in range(metadata.num_row_groups):
                for row in range(metadata.row_group(group).num_rows):
                    entries.append((shard, group, row))
        return entries

    def _row_group_texts(self, shard: Path, group: int) -> List[str]:
        """The texts of one row group, cached (a few MB at most)."""
        key = (str(shard), int(group))
        if key in self._cache:
            return self._cache[key]
        import pyarrow.parquet as pq

        table = pq.ParquetFile(shard).read_row_group(
            int(group), columns=[self.cfg.text_column]
        )
        texts = [text for text in table.column(0).to_pylist() if text]
        self._cache[key] = texts
        self._cache_order.append(key)
        while len(self._cache_order) > _ROW_GROUP_CACHE:
            self._cache.pop(self._cache_order.pop(0), None)
        return texts

    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self._rows)

    def document(self, index: int) -> str:
        """The raw text of the ``index``-th item of this split."""
        shard, group, row = self._entries[self._rows[index]]
        texts = self._row_group_texts(shard, group)
        if row >= len(texts):  # a group whose rows all held null text
            return ""
        return texts[row]

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        document = self.document(index)
        ids = self.tokenizer.encode(document, add_bos=False, add_eos=False)
        # Deterministic per (seed, index): a re-run corrupts identically, and the
        # global RNG is untouched so the rest of the pipeline stays reproducible.
        rng = random.Random(self.seed * 1_000_003 + int(index))
        corruption = corrupt_spans(
            ids,
            self.tokenizer,
            noise_density=self.cfg.noise_density,
            mean_span_length=self.cfg.mean_span_length,
            max_src_len=self.max_src_len,
            max_tgt_len=self.max_tgt_len,
            rng=rng,
        )
        return corruption.src_ids, corruption.tgt_ids


def build_denoising_dataloaders(
    cfg: DataConfig,
    tokenizer: BPETokenizer,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    max_src_len: int = 512,
    max_tgt_len: int = 512,
) -> Dict[str, torch.utils.data.DataLoader]:
    """
    ``{"train": DataLoader, "val": DataLoader}`` for the denoising objective.

    Uses the same ``collate_batch`` as the translation path, so padding, the mask
    polarity and the ``tgt_in``/``tgt_out`` shift are shared.  Train shuffling uses
    its own seeded generator and therefore leaves the global RNG alone.
    """
    loaders: Dict[str, torch.utils.data.DataLoader] = {}
    for split in ("train", "val"):
        dataset = FineWebDenoisingDataset(
            cfg,
            tokenizer,
            split=split,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
            seed=seed,
        )
        generator = None
        if split == "train":
            generator = torch.Generator()
            generator.manual_seed(int(seed))
        loaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=split == "train",
            generator=generator,
            num_workers=int(num_workers),
            collate_fn=collate_batch,
            drop_last=False,
        )
    return loaders
