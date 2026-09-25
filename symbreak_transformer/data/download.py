"""Parallel-text loading: HuggingFace mirror, local files, and synthetic tasks.

Three sources, one entry point (:func:`load_parallel_data`), all of them
returning a list of ``(source_sentence, target_sentence)`` string pairs:

``hf``
    A JSON-Lines file from the HuggingFace *mirror* (``hf-mirror.com``).
    ``huggingface.co`` is DNS-poisoned on this machine, and the mirror answers
    ``403`` without a ``User-Agent`` header, so the request always carries
    ``cfg.user_agent``.  Downloads go through :mod:`urllib.request` (``curl.exe``
    and ``Invoke-WebRequest`` are broken here: schannel TLS failure) and are
    cached under ``<data_dir>/raw/<repo with "/" -> "__">/<filename>`` so a
    second run costs no network at all.
``local``
    ``<data_dir>/local/<split>.tsv`` (no header), optionally a ``.jsonl``
    sibling using ``cfg.src_field`` / ``cfg.tgt_field``.
``synthetic``
    A deterministic toy task (``copy`` / ``reverse`` / ``sort``) over decimal
    token ids.  This is the offline fallback: it needs no files and no network,
    so a smoke run works on a plane.

Every file/network loader either returns a **non-empty** list or raises
``ValueError`` naming the split and the path/URL it looked at, so a silent
"0 examples" training run cannot happen.
"""

from __future__ import annotations

import csv
import json
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Sequence

from ..config import SYNTHETIC_TASKS, DataConfig

#: One aligned sentence pair: ``(source_sentence, target_sentence)``.
Pair = tuple[str, str]

#: Split names the synthetic generator knows a size for.
_SYNTHETIC_SIZES = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _require_pairs(pairs: List[Pair], split: str, where: object) -> List[Pair]:
    """Return ``pairs``, or raise ``ValueError`` naming split and location."""
    if not pairs:
        raise ValueError(
            f"no parallel sentence pairs for split {split!r} at {where} "
            f"(the file/URL was read but produced 0 usable rows)"
        )
    return pairs


def _cache_path(cfg: DataConfig, filename: str) -> Path:
    """``<data_dir>/raw/<repo with "/" -> "__">/<filename>``."""
    return Path(cfg.data_dir) / "raw" / cfg.hf_repo.replace("/", "__") / filename


def _read_jsonl(path: Path, cfg: DataConfig) -> List[Pair]:
    """Parse a JSON-Lines file whose objects carry ``src_field``/``tgt_field``."""
    pairs: List[Pair] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {lineno} is not valid JSON: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(
                    f"{path}: line {lineno} must be a JSON object, got "
                    f"{type(obj).__name__}"
                )
            missing = [f for f in (cfg.src_field, cfg.tgt_field) if f not in obj]
            if missing:
                raise ValueError(
                    f"{path}: line {lineno} is missing field(s) {missing}; "
                    f"available keys are {sorted(obj)}"
                )
            src = str(obj[cfg.src_field]).strip()
            tgt = str(obj[cfg.tgt_field]).strip()
            if not src and not tgt:
                continue
            pairs.append((src, tgt))
    return pairs


def _read_tsv(path: Path, cfg: DataConfig, split: str) -> List[Pair]:
    """Read a header-less, tab-separated parallel file with column indices."""
    src_col, tgt_col = int(cfg.local_src_col), int(cfg.local_tgt_col)
    if src_col < 0 or tgt_col < 0:
        raise ValueError(
            f"local_src_col/local_tgt_col must be >= 0, got "
            f"{src_col}/{tgt_col} (split {split!r}, {path})"
        )
    need = max(src_col, tgt_col)
    pairs: List[Pair] = []
    # newline="" is the csv module's requirement for correct line splitting.
    with path.open("r", encoding="utf-8", newline="") as fh:
        for lineno, row in enumerate(csv.reader(fh, delimiter="\t"), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) <= need:
                raise ValueError(
                    f"{path}: line {lineno} has {len(row)} column(s) but "
                    f"column {need} is required (split {split!r})"
                )
            src = row[src_col].strip()
            tgt = row[tgt_col].strip()
            if not src and not tgt:
                continue
            pairs.append((src, tgt))
    return pairs


def _download_to(url: str, dest: Path, cfg: DataConfig) -> Path:
    """Download ``url`` to ``dest`` (atomically) using ``cfg.user_agent``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": cfg.user_agent})
    with urllib.request.urlopen(request, timeout=cfg.download_timeout) as response:
        payload = response.read()
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(dest)
    return dest


# --------------------------------------------------------------------------- #
# HuggingFace mirror
# --------------------------------------------------------------------------- #
def download_hf_split(cfg: DataConfig, split: str) -> List[Pair]:
    """Load one split from the HuggingFace mirror (with a local raw cache).

    Args:
        cfg: data configuration; ``hf_endpoint``, ``hf_repo``, ``hf_files``,
            ``src_field``, ``tgt_field``, ``user_agent``, ``download_timeout``
            and the ``max_{train,val}_samples`` caps are all honoured.
        split: split name; must be a key of ``cfg.hf_files``.

    Returns:
        A non-empty list of ``(source, target)`` pairs.  ``train``/``val`` are
        clipped to ``max_train_samples``/``max_val_samples`` when those are
        non-zero; every other split (``test``) is unlimited.

    Raises:
        ValueError: unknown split, unparsable payload, or an empty result.
            Network/HTTP errors propagate unchanged (``urllib`` exceptions).
    """
    if split not in cfg.hf_files:
        raise ValueError(
            f"split {split!r} has no entry in cfg.hf_files "
            f"(known: {sorted(cfg.hf_files)}); add one to download it"
        )
    relative = str(cfg.hf_files[split])
    filename = Path(relative).name
    url = f"{cfg.hf_endpoint}/datasets/{cfg.hf_repo}/resolve/main/{relative}"
    cache = _cache_path(cfg, filename)
    if not cache.exists():
        _download_to(url, cache, cfg)

    pairs = _read_jsonl(cache, cfg)
    limit = {"train": int(cfg.max_train_samples), "val": int(cfg.max_val_samples)}.get(
        split, 0
    )
    if limit > 0:
        pairs = pairs[:limit]
    return _require_pairs(pairs, split, f"{url} (cached at {cache})")


# --------------------------------------------------------------------------- #
# Synthetic task
# --------------------------------------------------------------------------- #
def synthetic_pairs(cfg: DataConfig, split: str) -> List[Pair]:
    """Build the deterministic synthetic task pairs for one split.

    Tokens are decimal strings of ids drawn uniformly from
    ``[0, cfg.synthetic_vocab)``; each sentence has exactly ``cfg.synthetic_len``
    tokens.  The RNG is seeded from ``(cfg.synthetic_seed, split)`` only, so the
    result depends on nothing else: the same config always yields the same
    pairs, and a split is a prefix of the same split generated with a larger
    size.

    Tasks (``cfg.synthetic_task``):

    * ``copy``    -- target == source
    * ``reverse`` -- target is the reversed token list
    * ``sort``    -- target is the token list sorted ascending **numerically**
      (the tokens are ids, so ``"10"`` sorts after ``"9"``)

    Args:
        cfg: data configuration (seed, per-split sizes, vocab, length, task).
        split: one of ``"train"``, ``"val"``, ``"test"``.

    Returns:
        ``cfg.synthetic_<split>_size`` pairs.  A size of ``0`` yields an empty
        list, which lets :func:`..dataset.build_dataloaders` omit that split.

    Raises:
        ValueError: unknown split, unknown task, negative size, or a
            non-positive ``synthetic_vocab`` / ``synthetic_len``.
    """
    sizes = {
        "train": cfg.synthetic_train_size,
        "val": cfg.synthetic_val_size,
        "test": cfg.synthetic_test_size,
    }
    if split not in sizes:
        raise ValueError(
            f"unknown synthetic split {split!r}; expected one of {_SYNTHETIC_SIZES}"
        )
    if cfg.synthetic_task not in SYNTHETIC_TASKS:
        raise ValueError(
            f"data.synthetic_task must be one of {SYNTHETIC_TASKS}, "
            f"got {cfg.synthetic_task!r}"
        )
    size = int(sizes[split])
    if size < 0:
        raise ValueError(f"synthetic_{split}_size must be >= 0, got {size}")
    vocab = int(cfg.synthetic_vocab)
    length = int(cfg.synthetic_len)
    if vocab < 1:
        raise ValueError(f"data.synthetic_vocab must be >= 1, got {vocab}")
    if length < 1:
        raise ValueError(f"data.synthetic_len must be >= 1, got {length}")

    # str seeds are hashed deterministically (sha512) by random.Random, so this
    # is stable across processes -- no PYTHONHASHSEED dependence.
    rng = random.Random(f"{cfg.synthetic_seed}:{split}")
    task = cfg.synthetic_task
    pairs: List[Pair] = []
    for _ in range(size):
        source = [str(rng.randrange(vocab)) for _ in range(length)]
        if task == "copy":
            target = list(source)
        elif task == "reverse":
            target = source[::-1]
        else:  # "sort"
            target = sorted(source, key=int)
        pairs.append((" ".join(source), " ".join(target)))
    return pairs


# --------------------------------------------------------------------------- #
# Local files
# --------------------------------------------------------------------------- #
def load_local_split(cfg: DataConfig, split: str) -> List[Pair]:
    """Read one split from ``<data_dir>/local/<split>.tsv`` (or ``.jsonl``).

    The TSV has no header; ``cfg.local_src_col`` / ``cfg.local_tgt_col`` select
    the columns.  If the TSV is absent but a ``<split>.jsonl`` sibling exists it
    is used instead, with the same ``src_field``/``tgt_field`` as the HF loader.

    Raises:
        ValueError: neither file exists, a row is too short, or 0 pairs were read.
    """
    base = Path(cfg.data_dir) / "local"
    tsv = base / f"{split}.tsv"
    jsonl = base / f"{split}.jsonl"
    if tsv.exists():
        pairs = _read_tsv(tsv, cfg, split)
        where: object = tsv
    elif jsonl.exists():
        pairs = _read_jsonl(jsonl, cfg)
        where = jsonl
    else:
        raise ValueError(
            f"no local data for split {split!r}: looked for {tsv} and {jsonl}"
        )
    return _require_pairs(pairs, split, where)


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def load_parallel_data(cfg: DataConfig, split: str) -> List[Pair]:
    """Load one split from whatever ``cfg.source`` selects.

    ``"hf"`` is the only source with a fallback: if the download, the HTTP
    response or the JSON parsing fails and ``cfg.fallback_to_synthetic`` is
    true, a warning is printed and :func:`synthetic_pairs` is returned instead
    (so an offline machine still runs); when the flag is false the original
    exception is re-raised untouched.

    Args:
        cfg: data configuration.
        split: split name (``"train"``, ``"val"``, ``"test"`` by default).

    Returns:
        The list of pairs for that split.

    Raises:
        ValueError: ``cfg.source`` is not one of ``hf``/``synthetic``/``local``.
        Exception: whatever the ``"hf"`` loader raised, if falling back is off.
    """
    source = str(cfg.source).strip().lower()
    if source == "hf":
        try:
            return download_hf_split(cfg, split)
        except Exception as exc:  # network, HTTP, JSON, empty file, ...
            if not cfg.fallback_to_synthetic:
                raise
            print(
                f"[data] WARNING: HuggingFace download failed for split {split!r} "
                f"({type(exc).__name__}: {exc}). "
                f"Falling back to the synthetic {cfg.synthetic_task!r} task "
                f"(data.fallback_to_synthetic=true).",
                flush=True,
            )
            return synthetic_pairs(cfg, split)
    if source == "synthetic":
        return synthetic_pairs(cfg, split)
    if source == "local":
        return load_local_split(cfg, split)
    raise ValueError(
        f"unknown data.source {cfg.source!r}; expected one of ('hf', 'synthetic', "
        f"'local')"
    )


def load_all_splits(
    cfg: DataConfig, splits: Sequence[str] = ("train", "val", "test")
) -> Dict[str, List[Pair]]:
    """Load several splits into ``{split: pairs}``, preserving ``splits`` order.

    Raises:
        ValueError: propagated from :func:`load_parallel_data` for any bad split.
    """
    return {split: load_parallel_data(cfg, split) for split in splits}
