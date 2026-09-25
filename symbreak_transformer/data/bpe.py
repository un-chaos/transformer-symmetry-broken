"""A ``tokenizers``-backed BPE tokenizer with the project's frozen special ids.

Why a BPE tokenizer at all
--------------------------
:class:`symbreak_transformer.data.tokenizer.Tokenizer` is a word/char tokenizer
whose vocabulary is bounded by ``max_vocab``.  That is fine for Multi30k (~30k
distinct tokens in 29k sentence pairs), but FineWeb-Edu is a 10-billion-token web
crawl: a word-level vocabulary would be millions of entries wide and would still
turn every rare name and typo into ``<unk>``.  Standard practice -- and what the
companion project does -- is a byte-pair-encoded subword vocabulary.

This class is a **drop-in replacement** for ``Tokenizer``: same four special ids
(``<pad> <unk> <bos> <eos>`` == 0..3), same ``itos``/``stoi``/``lowercase``
attributes, same ``tokenize``/``encode``/``decode``/``save``/``load`` methods,
same ``encode`` truncation rule (clip to ``max_len`` and force the last id to
``<eos>``).  Everything downstream (``collate_batch``, the model, ``evaluate``)
therefore works unchanged.

Id layout
---------
``tokenizers`` numbers its own vocabulary from 0.  Rather than let the backend
own the id space, the backend ids are **shifted by four** so that the shared
specials always come first:

======================  ====================================================
ids                     meaning
======================  ====================================================
``0..3``                ``<pad> <unk> <bos> <eos>`` (asserted at build time)
``4 .. 4+V-1``          the V BPE tokens the trainer learned, backend id + 4
``4+V .. 4+V+S-1``      ``<extra_id_0> ... <extra_id_{S-1}>`` span sentinels
======================  ====================================================

The sentinels are the span markers of the T5 denoising objective
(:mod:`symbreak_transformer.data.denoising`).  They are *not* part of the BPE
model -- reserving them keeps the trained vocabulary independent of how many
sentinels a later run wants -- so :meth:`tokenize` handles them itself, before
and after calling the backend.  A text may therefore contain ``<extra_id_0>`` and
it comes back as exactly one id.

Determinism
-----------
``BpeTrainer`` is deterministic for a fixed corpus and iteration order (the
merges are picked by frequency, ties by first occurrence), and the corpus itself
is read in sorted shard order by :func:`iter_fineweb_texts`.  A trained
tokenizer is cached at
``<fineweb_dir>/../tokenizer_bpe_<vocab_size>.json``, so a training run and a
later ``evaluate`` share one vocabulary even though both would otherwise have to
re-train it.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, Iterator, List, Optional

from ..config import DataConfig
from .fineweb import fineweb_local_files, iter_fineweb_texts
from .tokenizer import BOS_ID, EOS_ID, PAD_ID, SPECIAL_TOKENS, UNK_ID

__all__ = [
    "DEFAULT_NUM_SENTINELS",
    "BPETokenizer",
    "SENTINEL_PREFIX",
    "SENTINEL_RE",
    "TOKENIZER_OFFSET",
    "build_bpe_tokenizer",
    "tokenizer_path_for",
]

#: Sentinels reserved by :func:`build_bpe_tokenizer`.  The denoising objective
#: needs ``spans + 1``; with ``noise_density <= 0.5`` and
#: ``mean_span_length >= 1`` a 512-token source can need at most ~256, and T5
#: itself reserves 100 for its 512-token configuration.  100 is the same default
#: as :meth:`BPETokenizer.train`, so a hand-trained tokenizer and the cached one
#: have identical ids.
DEFAULT_NUM_SENTINELS = 100

#: Ids 0..3 are the four specials, matching ``tokenizer.SPECIAL_TOKENS`` exactly.
#: Every backend (BPE) id is shifted up by this constant.
TOKENIZER_OFFSET = len(SPECIAL_TOKENS)  # == 4

#: Sentinel names are ``<extra_id_0>``, ``<extra_id_1>``, ... (T5's convention).
SENTINEL_PREFIX = "<extra_id_"

#: Matches a sentinel anywhere in a raw string, so :meth:`BPETokenizer.tokenize`
#: can pull sentinels out before the BPE backend ever sees them.
SENTINEL_RE = re.compile(re.escape(SENTINEL_PREFIX) + r"(\d+)>")

#: Saved-file schema version (a future incompatible change bumps it).
_FORMAT_VERSION = 1

#: ``tokenizers`` must know an unknown token when the BPE model is built.  This
#: one is *replaced* by :data:`SPECIAL_TOKENS` in the public id space, so it is
#: never exposed.
_BACKEND_UNK = "<unk>"

#: A rough characters-per-document estimate, used only to decide whether the
#: cache path can be derived; never a correctness input.
_TOKENIZER_FILE = "tokenizer_bpe_{vocab}.json"


def _sentinel_token(index: int) -> str:
    """``0 -> "<extra_id_0>"``."""
    return f"{SENTINEL_PREFIX}{index}>"


def tokenizer_path_for(cfg: DataConfig) -> Path:
    """Where :func:`build_bpe_tokenizer` caches its vocabulary for ``cfg``.

    The path is derived from the corpus directory and the vocabulary size, so
    the same corpus always maps to the same file::

        <cfg.fineweb_dir>/../tokenizer_bpe_<cfg.bpe_vocab_size>.json

    (i.e. one level *above* the shard directory, so the tokenizer is never
    mistaken for a parquet shard).
    """
    vocab = int(cfg.bpe_vocab_size)
    corpus_dir = Path(str(cfg.fineweb_dir)).expanduser()
    try:
        parent = corpus_dir.resolve().parent
    except OSError:  # pragma: no cover - unresolvable path
        parent = corpus_dir.parent
    return parent / _TOKENIZER_FILE.format(vocab=vocab)


class BPETokenizer:
    """A ``tokenizers``-backed BPE tokenizer, interface-compatible with Tokenizer.

    Args:
        backend: a trained ``tokenizers.Tokenizer`` whose ids ``0..V-1`` are the
            BPE tokens (its own ``<unk>`` entry, if any, is *not* special-cased
            here; :meth:`train` strips it).
        itos: id -> token table.  ``itos[:4]`` must be ``SPECIAL_TOKENS``, the
            next ``V`` entries the BPE tokens, and any remaining entries the
            ``<extra_id_k>`` sentinels.
        mode: kept at ``"bpe"`` for interface parity with ``Tokenizer.mode``.

    Attributes:
        itos: the id -> token list (specials, BPE tokens, sentinels).
        stoi: token -> id (first occurrence wins, so a BPE merge that literally
            spells ``<eos>`` cannot shadow id 3).
        lowercase: whether :meth:`tokenize` folds case; restored by :meth:`load`.
    """

    #: Class attributes, so ``BPETokenizer.pad_id`` works without an instance.
    pad_id = PAD_ID
    unk_id = UNK_ID
    bos_id = BOS_ID
    eos_id = EOS_ID

    def __init__(self, backend, itos: List[str], mode: str = "bpe") -> None:
        self.backend = backend
        self.itos: List[str] = [str(tok) for tok in itos]
        if self.itos[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(
                f"itos[0:{len(SPECIAL_TOKENS)}] must be exactly {SPECIAL_TOKENS}, "
                f"got {self.itos[: len(SPECIAL_TOKENS)]}"
            )
        self.mode: str = mode
        self.lowercase: bool = True
        self.stoi: Dict[str, int] = {}
        for index, token in enumerate(self.itos):
            self.stoi.setdefault(token, index)

        # The reserved sentinels are the trailing `<extra_id_0> ... <extra_id_{n-1}>`
        # run in *ascending* order.  Deriving the count from `itos` (instead of
        # storing it) keeps `load` exact even for a hand-built tokenizer, but the
        # order has to match `sentinel_id`, which indexes the run from its start --
        # scanning backwards and expecting `<extra_id_0>` last would count 0.
        self.num_sentinels: int = 0
        cursor = len(self.itos) - 1
        while (
            cursor >= len(SPECIAL_TOKENS)
            and SENTINEL_RE.fullmatch(self.itos[cursor]) is not None
        ):
            cursor -= 1
        run = self.itos[cursor + 1 :]
        if run == [_sentinel_token(index) for index in range(len(run))]:
            self.num_sentinels = len(run)
        #: Everything between the specials and the sentinels is BPE.
        self.bpe_vocab_size: int = (
            len(self.itos) - len(SPECIAL_TOKENS) - self.num_sentinels
        )

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        """``len(itos)``, including the specials, the BPE ids and the sentinels."""
        return len(self.itos)

    def __len__(self) -> int:
        return len(self.itos)

    def __repr__(self) -> str:
        return (
            f"BPETokenizer(mode={self.mode!r}, vocab_size={self.vocab_size}, "
            f"bpe={self.bpe_vocab_size}, sentinels={self.num_sentinels}, "
            f"lowercase={self.lowercase})"
        )

    def sentinel_id(self, index: int) -> int:
        """The reserved id of sentinel ``index`` (``<extra_id_{index}>``).

        Raises:
            IndexError: ``index`` is outside ``[0, num_sentinels)``.  The
                denoising objective reserves one more sentinel than it can
                possibly need (T5 needs ``n+1`` sentinels for ``n`` spans), so
                this firing means a caller asked for more spans than it
                reserved.
        """
        position = int(index)
        if not 0 <= position < self.num_sentinels:
            raise IndexError(
                f"sentinel {position} is outside the {self.num_sentinels} reserved "
                f"sentinels (ids {self.vocab_size - self.num_sentinels}.."
                f"{self.vocab_size - 1})"
            )
        return len(self.itos) - self.num_sentinels + position

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 32000,
        lowercase: bool = True,
        num_sentinels: int = 100,
        progress: Optional[Callable[[int], None]] = None,
    ) -> "BPETokenizer":
        """Train a BPE vocabulary on an iterable of raw documents.

        Args:
            texts: documents (an iterator is fine; nothing is held in memory).
            vocab_size: size of the **BPE** vocabulary, i.e. the number of
                learned subword tokens.  The final ``vocab_size`` property is
                ``vocab_size + 4 + num_sentinels`` (the trainer stops early when
                the corpus does not contain that many distinct merges).
            lowercase: fold case with a ``Lowercase`` normalizer before merging.
            num_sentinels: how many ``<extra_id_k>`` ids to reserve.
            progress: optional ``callable(documents_seen)``, called every 1000
                documents -- a hook for the CLI's progress line, never required.

        Returns:
            A :class:`BPETokenizer` whose ``itos[:4] == SPECIAL_TOKENS``.

        Raises:
            ImportError: the ``tokenizers`` package is missing.
            ValueError: ``vocab_size`` or ``num_sentinels`` is not positive.
        """
        try:
            from tokenizers import Tokenizer, decoders, models, normalizers
            from tokenizers import pre_tokenizers, trainers
        except ImportError as exc:  # pragma: no cover - dependency hint
            raise ImportError(
                "training a BPE tokenizer needs the 'tokenizers' package: "
                "pip install tokenizers"
            ) from exc

        if int(vocab_size) < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if int(num_sentinels) < 0:
            raise ValueError(f"num_sentinels must be >= 0, got {num_sentinels}")

        backend = Tokenizer(models.BPE(unk_token=_BACKEND_UNK, fuse_unk=True))
        if lowercase:
            backend.normalizer = normalizers.Lowercase()
        # Whitespace *and* punctuation, so "world." and "world" "." agree and a
        # sentence-ending period never lands inside a word piece.  This mirrors
        # `tokenizer.Tokenizer.tokenize` (``\w+|[^\w\s]``) closely enough that
        # the same text splits the same way.
        backend.pre_tokenizer = pre_tokenizers.Sequence(
            [pre_tokenizers.Whitespace(), pre_tokenizers.Punctuation()]
        )
        backend.decoder = decoders.BPEDecoder()

        trainer = trainers.BpeTrainer(
            vocab_size=int(vocab_size),
            min_frequency=1,
            # The model needs *an* unknown token; it becomes the public id 1.
            special_tokens=[_BACKEND_UNK],
            show_progress=False,
        )

        def counted(source: Iterable[str]) -> Iterator[str]:
            seen = 0
            for document in source:
                text = "" if document is None else str(document)
                if not text.strip():
                    # Empty documents contribute nothing and only slow the trainer.
                    continue
                seen += 1
                if progress is not None and seen % 1000 == 0:
                    progress(seen)
                yield text

        backend.train_from_iterator(counted(texts), trainer=trainer)

        # Rebuild the public id space: specials first, then every BPE token in
        # backend id order (shifted by +4), then the reserved sentinels.
        vocab = backend.get_vocab()
        bpe_tokens = [token for token, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
        itos = list(SPECIAL_TOKENS) + bpe_tokens
        itos += [_sentinel_token(index) for index in range(int(num_sentinels))]
        tokenizer = cls(backend, itos, mode="bpe")
        tokenizer.lowercase = bool(lowercase)
        if tokenizer.itos[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            # Belt and braces: the id layout is load-bearing for every consumer.
            raise AssertionError(
                f"trained itos[0:{len(SPECIAL_TOKENS)}] != {SPECIAL_TOKENS}: "
                f"{tokenizer.itos[: len(SPECIAL_TOKENS)]}"
            )
        return tokenizer

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> List[str]:
        """Split ``text`` into tokens, sentinels included.

        Sentinels are pulled out first (one token each, whatever the surrounding
        whitespace), then each remaining segment goes through the BPE backend.
        The returned tokens are the strings in :attr:`itos`, so they can be fed
        back through :attr:`stoi` -- exactly like ``Tokenizer.tokenize``.
        """
        text = "" if text is None else str(text)
        pieces: List[str] = []
        cursor = 0
        for match in SENTINEL_RE.finditer(text):
            pieces.extend(self._tokenize_plain(text[cursor : match.start()]))
            pieces.append(self._sentinels.get(match.group(0), SPECIAL_TOKENS[UNK_ID]))
            cursor = match.end()
        pieces.extend(self._tokenize_plain(text[cursor:]))
        return pieces

    def _tokenize_plain(self, text: str) -> List[str]:
        """BPE-tokenize a segment that contains no sentinel."""
        if not text.strip():
            return []
        return list(self.backend.encode(text, add_special_tokens=False).tokens)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
        max_len: int = 0,
    ) -> List[int]:
        """Encode ``text`` to ids, optionally framing it with BOS/EOS.

        Identical contract to ``Tokenizer.encode``: ``[BOS] + tokens + [EOS]``,
        and when ``max_len > 0`` clips to that many ids and forces the last id
        to ``<eos>`` if ``add_eos`` (so a truncated sequence keeps its framing).
        Unknown tokens become ``<unk>`` (id 1).
        """
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(
            self.stoi.get(token, self.unk_id) for token in self.tokenize(text)
        )
        if add_eos:
            ids.append(self.eos_id)

        limit = int(max_len)
        if limit > 0 and len(ids) > limit:
            ids = ids[:limit]
            if add_eos:
                ids[-1] = self.eos_id
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Turn ids back into text, joining tokens with single spaces.

        Sentinels are **kept** (unlike ``<pad> <unk> <bos> <eos>``) because the
        denoising objective's whole point is to make them visible in a decoded
        target: ``skip_special`` only drops ids 0..3.

        Raises:
            ValueError: an id is outside ``[0, vocab_size)``.
        """
        pieces: List[str] = []
        specials = set(range(len(SPECIAL_TOKENS)))
        for raw in ids:
            index = int(raw)
            if not 0 <= index < len(self.itos):
                raise ValueError(
                    f"id {index} is out of range for a vocabulary of size "
                    f"{len(self.itos)}"
                )
            if skip_special and index in specials:
                continue
            pieces.append(self.itos[index])
        return " ".join(pieces)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    @property
    def _sentinels(self) -> Dict[str, int]:
        """``{"<extra_id_0>": id, ...}`` for the reserved sentinels."""
        start = len(self.itos) - self.num_sentinels
        return {self.itos[start + i]: start + i for i in range(self.num_sentinels)}

    def save(self, path) -> Path:
        """Write the tokenizer as **one** JSON file and return the path written.

        The payload carries both halves of the state: ``itos`` (the id layout,
        sentinels included) and the ``tokenizers`` backend serialised with
        ``to_str()`` (the merges, the normalizer, the pre-tokenizer).  A load
        therefore reproduces the tokenizer exactly, which is what lets a training
        run and a later ``evaluate`` agree on ids.

        ``path`` may be a ``.json`` file or a directory (``tokenizer.json`` is
        then written inside it).  Returns the file actually written.
        """
        target = Path(path)
        if target.is_dir():
            target = target / "tokenizer.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": _FORMAT_VERSION,
            "mode": self.mode,
            "lowercase": bool(self.lowercase),
            "specials": list(SPECIAL_TOKENS),
            "num_sentinels": int(self.num_sentinels),
            "itos": list(self.itos),
            "backend": self.backend.to_str(),
        }
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        return target

    @classmethod
    def load(cls, path) -> "BPETokenizer":
        """Load a tokenizer previously written by :meth:`save`.

        ``path`` may be the JSON file or the directory containing it.

        Raises:
            FileNotFoundError: the file does not exist.
            ValueError: the payload is not a BPE-tokenizer object.
        """
        from tokenizers import Tokenizer

        target = Path(path)
        if target.is_dir():
            target = target / "tokenizer.json"
        if not target.exists():
            raise FileNotFoundError(f"tokenizer file not found: {target}")
        with target.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or "itos" not in payload:
            raise ValueError(
                f"{target}: expected a tokenizer JSON object with an 'itos' list, "
                f"got {type(payload).__name__}"
            )
        if "backend" not in payload:
            raise ValueError(
                f"{target}: this file has no 'backend' entry; it was written by the "
                f"word/char Tokenizer, not BPETokenizer"
            )
        backend = Tokenizer.from_str(str(payload["backend"]))
        tokenizer = cls(backend, list(payload["itos"]), mode=str(payload.get("mode", "bpe")))
        tokenizer.lowercase = bool(payload.get("lowercase", True))
        return tokenizer


def build_bpe_tokenizer(
    cfg: DataConfig,
    texts: Optional[Iterable[str]] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> BPETokenizer:
    """Return the BPE tokenizer for ``cfg``, training and caching it if needed.

    The cache path is :func:`tokenizer_path_for` --
    ``<cfg.fineweb_dir>/../tokenizer_bpe_<cfg.bpe_vocab_size>.json`` -- so the
    expensive training happens once per (corpus, vocabulary size) and every later
    run (training and the evaluation that follows it) loads the same vocabulary.

    Training text comes from ``texts`` when given, otherwise from
    :func:`symbreak_transformer.data.fineweb.iter_fineweb_texts`, limited to
    ``cfg.tokenizer_train_documents`` documents (``0`` means "all of them").
    The number of reserved sentinels is fixed at :data:`DEFAULT_NUM_SENTINELS`.

    Args:
        cfg: data configuration (``fineweb_dir``, ``bpe_vocab_size``,
            ``tokenizer_train_documents``, ``text_column``, ``lowercase``).
        texts: explicit training documents; skips the corpus entirely when given.
        progress: optional ``callable(documents_seen)`` for the CLI's progress.

    Returns:
        A loaded or freshly trained :class:`BPETokenizer`.

    Raises:
        FileNotFoundError: no shards under ``cfg.fineweb_dir`` and no ``texts``.
    """
    target = tokenizer_path_for(cfg)
    if target.exists():
        try:
            cached = BPETokenizer.load(target)
        except (ValueError, KeyError) as exc:
            print(f"[bpe] ignoring unreadable cache {target}: {exc}", flush=True)
        else:
            if cached.bpe_vocab_size == int(cfg.bpe_vocab_size):
                print(
                    f"[bpe] loaded {target} (vocab_size={cached.vocab_size}, "
                    f"bpe={cached.bpe_vocab_size}, sentinels={cached.num_sentinels})",
                    flush=True,
                )
                return cached
            print(
                f"[bpe] cache {target} holds a {cached.bpe_vocab_size}-token BPE "
                f"vocabulary, want {int(cfg.bpe_vocab_size)}; retraining",
                flush=True,
            )

    if texts is None:
        limit = int(cfg.tokenizer_train_documents) or None
        shards = fineweb_local_files(cfg.fineweb_dir)
        print(
            f"[bpe] training on up to {limit if limit else 'all'} document(s) from "
            f"{len(shards)} shard(s) in {cfg.fineweb_dir}",
            flush=True,
        )
        source: Iterable[str] = iter_fineweb_texts(
            cfg.fineweb_dir, column=cfg.text_column, limit=limit
        )
    else:
        source = texts

    tokenizer = BPETokenizer.train(
        source,
        vocab_size=int(cfg.bpe_vocab_size),
        lowercase=bool(cfg.lowercase),
        num_sentinels=DEFAULT_NUM_SENTINELS,
        progress=progress,
    )
    tokenizer.save(target)
    print(
        f"[bpe] trained vocab_size={tokenizer.vocab_size} "
        f"(bpe={tokenizer.bpe_vocab_size}, sentinels={tokenizer.num_sentinels}) -> "
        f"{target}",
        flush=True,
    )
    return tokenizer
