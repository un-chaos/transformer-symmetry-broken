"""Data package: parallel-text loading, tokenization and batching.

Public API (see the port contract, section "Public API required by the lead
agent", part B)::

    from symbreak_transformer.data import (
        Pair,                      # (source_sentence, target_sentence)
        load_parallel_data,        # dispatch on cfg.source
        load_all_splits,           # {split: pairs}
        download_hf_split,         # hf-mirror JSONL (cached under data/raw)
        synthetic_pairs,           # deterministic copy/reverse/sort task
        load_local_split,          # <data_dir>/local/<split>.tsv|.jsonl
        PAD_ID, UNK_ID, BOS_ID, EOS_ID, SPECIAL_TOKENS,
        Tokenizer, build_tokenizers,
        ParallelTextDataset, collate_batch, build_dataloaders,
    )

Everything is driven by a :class:`symbreak_transformer.config.DataConfig`: there
is no full experiment config any more, so the batch size, worker count and seed
are explicit arguments of :func:`build_dataloaders`.

The padding/mask convention (``True`` means PAD) and the ``tgt_in``/``tgt_out``
shift are documented in ``dataset.py``.
"""

from __future__ import annotations

from .dataset import ParallelTextDataset, build_dataloaders, collate_batch
from .download import (
    Pair,
    download_hf_split,
    load_all_splits,
    load_local_split,
    load_parallel_data,
    synthetic_pairs,
)
from .tokenizer import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SPECIAL_TOKENS,
    UNK_ID,
    Tokenizer,
    build_tokenizers,
)

__all__ = [
    # download.py
    "Pair",
    "load_parallel_data",
    "load_all_splits",
    "download_hf_split",
    "synthetic_pairs",
    "load_local_split",
    # tokenizer.py
    "PAD_ID",
    "UNK_ID",
    "BOS_ID",
    "EOS_ID",
    "SPECIAL_TOKENS",
    "Tokenizer",
    "build_tokenizers",
    # dataset.py
    "ParallelTextDataset",
    "collate_batch",
    "build_dataloaders",
]
