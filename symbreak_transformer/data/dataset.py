"""``Dataset``/``DataLoader`` plumbing and the padding convention.

One convention, used everywhere downstream:

* ids are padded on the **right** with ``<pad>`` (= 0);
* a padding mask is a ``torch.bool`` tensor where **``True`` means PAD / ignore**
  (the same polarity as ``nn.MultiheadAttention``'s ``key_padding_mask``);
* the decoder is teacher-forced by shifting the target by one::

      tgt_ids = [<bos>, w1, w2, <eos>]
      tgt_in  = [<bos>, w1, w2]        # decoder input
      tgt_out = [w1, w2, <eos>]        # labels, aligned with tgt_in

The source carries **no** BOS (the encoder has no use for one) but always ends
with ``<eos>``; the target is framed ``<bos> ... <eos>``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ..config import DataConfig
from .download import Pair, load_all_splits
from .tokenizer import Tokenizer


class ParallelTextDataset(Dataset):
    """Tokenized parallel sentences, one ``(src_ids, tgt_ids)`` pair per item.

    Args:
        pairs: ``(source_sentence, target_sentence)`` strings.
        src_tokenizer: tokenizer for the source side.
        tgt_tokenizer: tokenizer for the target side.
        max_src_len: clip length for the source; ``0`` falls back to
            ``default_max_len`` (the training script passes
            ``cfg_model.context_length``), and when that is ``0`` too clipping
            is disabled.
        max_tgt_len: clip length for the **framed** target ``[BOS] ... [EOS]``,
            so the decoder gets at most that many positions; ``0`` falls back to
            ``default_max_len``, and when that is ``0`` too clipping is disabled.
            Clipping always keeps the ``<eos>`` (see :meth:`Tokenizer.encode`).
        default_max_len: fallback used by either length when it is ``0``.

    Item format:
        ``(src_ids, tgt_ids)`` with ``src_ids`` ending in ``<eos>`` and
        ``tgt_ids`` framed ``<bos> ... <eos>``.
    """

    def __init__(
        self,
        pairs: List[Pair],
        src_tokenizer: Tokenizer,
        tgt_tokenizer: Tokenizer,
        max_src_len: int = 0,
        max_tgt_len: int = 0,
        default_max_len: int = 0,
    ) -> None:
        self.pairs: List[Pair] = list(pairs)
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        fallback = int(default_max_len)
        self.max_src_len = int(max_src_len) or fallback
        self.max_tgt_len = int(max_tgt_len) or fallback
        self._lengths: Optional[Tuple[List[int], List[int]]] = None

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Tuple[List[int], List[int]]:
        """Return ``(src_ids, tgt_ids)`` for one pair."""
        src, tgt = self.pairs[index]
        src_ids = self.src_tokenizer.encode(
            src, add_bos=False, add_eos=True, max_len=self.max_src_len
        )
        tgt_ids = self.tgt_tokenizer.encode(
            tgt, add_bos=True, add_eos=True, max_len=self.max_tgt_len
        )
        return src_ids, tgt_ids

    def lengths(self) -> Tuple[List[int], List[int]]:
        """Unpadded ``(src_lengths, tgt_lengths)`` for the whole dataset.

        Computed once and cached on the instance (no global state); the returned
        lists are copies, so a caller cannot corrupt the cache.
        """
        if self._lengths is None:
            src_lengths: List[int] = []
            tgt_lengths: List[int] = []
            for src, tgt in self.pairs:
                src_lengths.append(
                    len(
                        self.src_tokenizer.encode(
                            src, add_bos=False, add_eos=True, max_len=self.max_src_len
                        )
                    )
                )
                tgt_lengths.append(
                    len(
                        self.tgt_tokenizer.encode(
                            tgt, add_bos=True, add_eos=True, max_len=self.max_tgt_len
                        )
                    )
                )
            self._lengths = (src_lengths, tgt_lengths)
        return list(self._lengths[0]), list(self._lengths[1])


def collate_batch(
    batch: List[Tuple[List[int], List[int]]], pad_id: int = 0
) -> Dict[str, torch.Tensor]:
    """Pad a batch of ``(src_ids, tgt_ids)`` into the training tensors.

    Args:
        batch: list of ``(src_ids, tgt_ids)`` as returned by
            :class:`ParallelTextDataset`.
        pad_id: right-padding id, normally ``<pad>`` = 0.

    Returns:
        Exactly these keys (and nothing else):

        ====================  ============  =============  ==========================
        key                   shape         dtype          meaning
        ====================  ============  =============  ==========================
        ``src``               ``(B, S)``    ``torch.long`` padded source ids
        ``tgt_in``            ``(B, T)``    ``torch.long`` ``tgt_ids[:-1]``
        ``tgt_out``           ``(B, T)``    ``torch.long`` ``tgt_ids[1:]`` (labels)
        ``src_padding_mask``  ``(B, S)``    ``torch.bool`` **True == PAD**
        ``tgt_padding_mask``  ``(B, T)``    ``torch.bool`` **True == PAD**
        ``src_lens``          ``(B,)``      ``torch.long`` unpadded lengths
        ``tgt_lens``          ``(B,)``      ``torch.long`` unpadded lengths
        ====================  ============  =============  ==========================

        ``T = max(tgt_lens) - 1`` because the last target id (``<eos>``) is a
        label for the second-to-last decoder position, not a decoder input.

    Raises:
        ValueError: the batch is empty, or a target has fewer than two ids (a
            target must at least be ``[BOS, EOS]`` to yield one label).
    """
    if not batch:
        raise ValueError("collate_batch received an empty batch")
    pad = int(pad_id)
    src_seqs = [list(src) for src, _ in batch]
    tgt_seqs = [list(tgt) for _, tgt in batch]
    src_lens = [len(seq) for seq in src_seqs]
    tgt_lens = [len(seq) for seq in tgt_seqs]
    for index, length in enumerate(tgt_lens):
        if length < 2:
            raise ValueError(
                f"target {index} has {length} id(s); expected at least 2 "
                f"([BOS, EOS]) so that tgt_in/tgt_out are non-empty"
            )

    size = len(batch)
    max_src = max(src_lens)
    max_tgt = max(tgt_lens) - 1  # tgt_in/tgt_out width

    src = torch.full((size, max_src), pad, dtype=torch.long)
    tgt_in = torch.full((size, max_tgt), pad, dtype=torch.long)
    tgt_out = torch.full((size, max_tgt), pad, dtype=torch.long)
    # Start fully masked (True == PAD) and unmask exactly the real positions.
    src_padding_mask = torch.ones((size, max_src), dtype=torch.bool)
    tgt_padding_mask = torch.ones((size, max_tgt), dtype=torch.bool)

    for row, (src_ids, tgt_ids) in enumerate(zip(src_seqs, tgt_seqs)):
        if src_ids:
            src[row, : len(src_ids)] = torch.tensor(src_ids, dtype=torch.long)
            src_padding_mask[row, : len(src_ids)] = False
        width = len(tgt_ids) - 1
        tgt_in[row, :width] = torch.tensor(tgt_ids[:-1], dtype=torch.long)
        tgt_out[row, :width] = torch.tensor(tgt_ids[1:], dtype=torch.long)
        tgt_padding_mask[row, :width] = False

    return {
        "src": src,
        "tgt_in": tgt_in,
        "tgt_out": tgt_out,
        "src_padding_mask": src_padding_mask,
        "tgt_padding_mask": tgt_padding_mask,
        "src_lens": torch.tensor(src_lens, dtype=torch.long),
        "tgt_lens": torch.tensor(tgt_lens, dtype=torch.long),
    }


def build_dataloaders(
    cfg: DataConfig,
    src_tokenizer: Tokenizer,
    tgt_tokenizer: Tokenizer,
    splits: Sequence[str] = ("train", "val", "test"),
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """Load the splits and wrap them in ``DataLoader``s.

    ``shuffle=True`` only for ``"train"``, and the shuffle order is reproducible
    because the loader gets its own ``torch.Generator`` seeded with ``seed`` (the
    global RNG is untouched).  ``drop_last=False``, so a trailing short batch is
    kept.  Length clipping uses ``cfg.max_src_len`` / ``cfg.max_tgt_len``; a value
    of ``0`` disables clipping for that side (the training script builds
    :class:`ParallelTextDataset` itself when it wants the model's
    ``context_length`` as the fallback).  For the target the budget applies to the
    framed ``[BOS] ... [EOS]`` list.

    Args:
        cfg: data configuration (source, split caps, length clipping).
        src_tokenizer: source tokenizer.
        tgt_tokenizer: target tokenizer.
        splits: which splits to build; a split whose pair list is empty is
            omitted from the result.
        batch_size: ``DataLoader`` batch size.
        num_workers: ``DataLoader`` worker count.
        seed: seed for the per-loader shuffle generator.

    Returns:
        ``{split: DataLoader}`` for the non-empty splits, in ``splits`` order.

    Raises:
        ValueError: every requested split was empty (nothing to train on).
    """
    pairs_by_split = load_all_splits(cfg, splits)
    max_src_len = int(cfg.max_src_len)
    max_tgt_len = int(cfg.max_tgt_len)

    loaders: Dict[str, DataLoader] = {}
    for split in splits:
        pairs = pairs_by_split.get(split) or []
        if not pairs:
            continue
        dataset = ParallelTextDataset(
            pairs,
            src_tokenizer,
            tgt_tokenizer,
            max_src_len=max_src_len,
            max_tgt_len=max_tgt_len,
        )
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        loaders[split] = DataLoader(
            dataset,
            batch_size=int(batch_size),
            shuffle=(split == "train"),
            num_workers=int(num_workers),
            collate_fn=collate_batch,
            drop_last=False,
            generator=generator,
        )
    if not loaders:
        raise ValueError(
            f"no non-empty split among {tuple(splits)} for data.source="
            f"{cfg.source!r}; nothing to train or evaluate on"
        )
    return loaders
