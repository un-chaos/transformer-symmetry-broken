"""A tiny, dependency-free word/character tokenizer with frozen special ids.

The four special ids are **fixed** because the rest of the project (model,
optimizer, evaluation) hard-codes them::

    0 = <pad>   1 = <unk>   2 = <bos>   3 = <eos>

Training a tokenizer is where a hidden non-determinism usually creeps into a
research repo, so the vocabulary order here is fully specified: ``itos[0:4]``
are the specials, and everything after them is sorted by **descending
frequency, ties broken alphabetically**.  The same corpus therefore always
produces byte-identical vocabularies, whatever the iteration order of the input.

``tokenizers``/``transformers`` are deliberately not used: a 30k-word-vocabulary
translation experiment does not need subword machinery, and this keeps the
tokenizer readable and auditable.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..config import DataConfig

#: Fixed special ids, shared by every ``Tokenizer`` instance.
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

#: ``itos[0:4]``; the order *is* the id assignment.
SPECIAL_TOKENS: List[str] = ["<pad>", "<unk>", "<bos>", "<eos>"]

#: Accepted ``Tokenizer.mode`` values.
TOKENIZER_MODES: Tuple[str, ...] = ("word", "char")

#: Saved-file schema version (a future incompatible change bumps it).
_FORMAT_VERSION = 1


class Tokenizer:
    """Word- or character-level tokenizer with a deterministic vocabulary.

    Args:
        itos: id -> token table.  The first four entries **must** be
            :data:`SPECIAL_TOKENS` in that exact order (``ValueError``
            otherwise), so ids 0..3 always mean pad/unk/bos/eos.
        mode: ``"word"`` (regex word pieces, optionally lower-cased) or
            ``"char"`` (one token per character).

    Attributes:
        itos: the id -> token list.
        stoi: the token -> id map (first occurrence wins).
        lowercase: whether :meth:`tokenize` lower-cases its input.  Set by
            :meth:`build` / :meth:`load`; defaults to ``True`` for a directly
            constructed tokenizer.
    """

    #: Class attributes, so ``Tokenizer.pad_id`` works without an instance.
    pad_id = PAD_ID
    unk_id = UNK_ID
    bos_id = BOS_ID
    eos_id = EOS_ID

    def __init__(self, itos: List[str], mode: str = "word") -> None:
        self.itos: List[str] = [str(tok) for tok in itos]
        if self.itos[: len(SPECIAL_TOKENS)] != SPECIAL_TOKENS:
            raise ValueError(
                f"itos[0:{len(SPECIAL_TOKENS)}] must be exactly {SPECIAL_TOKENS}, "
                f"got {self.itos[: len(SPECIAL_TOKENS)]}"
            )
        if mode not in TOKENIZER_MODES:
            raise ValueError(
                f"tokenizer mode must be one of {TOKENIZER_MODES}, got {mode!r}"
            )
        self.mode: str = mode
        self.lowercase: bool = True
        self.stoi: Dict[str, int] = {}
        for index, token in enumerate(self.itos):
            # setdefault: for duplicated strings the *first* (lowest) id wins,
            # which keeps the specials authoritative.
            self.stoi.setdefault(token, index)

    # ------------------------------------------------------------------ #
    # Basic properties
    # ------------------------------------------------------------------ #
    @property
    def vocab_size(self) -> int:
        """Number of entries in ``itos`` (including the four specials)."""
        return len(self.itos)

    def __len__(self) -> int:
        return len(self.itos)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(mode={self.mode!r}, vocab_size={self.vocab_size}, "
            f"lowercase={self.lowercase})"
        )

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        sentences: Iterable[str],
        mode: str = "word",
        min_freq: int = 2,
        max_vocab: int = 30000,
        lowercase: bool = True,
    ) -> "Tokenizer":
        """Build a vocabulary from a corpus of raw sentences.

        Tokens are counted with exactly the same :meth:`tokenize` the encoder
        uses, so the vocabulary can never disagree with encoding.  Tokens whose
        count is below ``min_freq`` are dropped; the remaining ones are sorted by
        ``(-count, token)`` (descending frequency, alphabetical tie-break) and
        truncated so that ``max_vocab`` holds **including** the four specials.

        Args:
            sentences: any iterable of raw strings (a generator is fine).
            mode: ``"word"`` or ``"char"``.
            min_freq: minimum occurrence count for a token to be kept (>= 1).
            max_vocab: hard cap on ``vocab_size``, specials included (>= 4).
            lowercase: whether to fold case before counting/encoding.

        Returns:
            A :class:`Tokenizer` whose ``itos`` starts with the specials.

        Raises:
            ValueError: bad ``mode``, ``min_freq < 1``, or ``max_vocab`` too
                small to hold the special tokens.
        """
        if mode not in TOKENIZER_MODES:
            raise ValueError(
                f"tokenizer mode must be one of {TOKENIZER_MODES}, got {mode!r}"
            )
        if int(min_freq) < 1:
            raise ValueError(f"min_freq must be >= 1, got {min_freq}")
        if int(max_vocab) < len(SPECIAL_TOKENS):
            raise ValueError(
                f"max_vocab must be >= {len(SPECIAL_TOKENS)} to hold the special "
                f"tokens, got {max_vocab}"
            )

        probe = cls(list(SPECIAL_TOKENS), mode=mode)
        probe.lowercase = bool(lowercase)
        counts: Counter = Counter()
        for sentence in sentences:
            counts.update(probe.tokenize(sentence))

        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        budget = int(max_vocab) - len(SPECIAL_TOKENS)
        kept = [token for token, count in ranked if count >= int(min_freq)][:budget]

        tokenizer = cls(list(SPECIAL_TOKENS) + kept, mode=mode)
        tokenizer.lowercase = bool(lowercase)
        return tokenizer

    # ------------------------------------------------------------------ #
    # Encoding / decoding
    # ------------------------------------------------------------------ #
    def tokenize(self, text: str) -> List[str]:
        """Split ``text`` into tokens (lower-casing first when enabled).

        ``word`` mode uses ``\\w+|[^\\w\\s]`` with ``re.UNICODE``, so accented
        letters stay inside words and punctuation becomes its own token; ``char``
        mode returns one token per character.
        """
        text = "" if text is None else str(text)
        if self.mode == "char":
            return list(text.lower() if self.lowercase else text)
        if self.lowercase:
            text = text.lower()
        return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = True,
        max_len: int = 0,
    ) -> List[int]:
        """Encode ``text`` to ids, optionally framing it with BOS/EOS.

        Args:
            text: raw sentence.
            add_bos: prepend ``<bos>``.
            add_eos: append ``<eos>``.
            max_len: ``> 0`` clips the framed id list to that many ids.  When
                clipping happens and ``add_eos`` is set, the **last id is forced
                to ``<eos>``**, so a truncated sequence always keeps its
                framing instead of ending mid-sentence.

        Returns:
            The list of ids (``<unk>`` for tokens outside the vocabulary).
        """
        ids: List[int] = []
        if add_bos:
            ids.append(BOS_ID)
        ids.extend(self.stoi.get(token, UNK_ID) for token in self.tokenize(text))
        if add_eos:
            ids.append(EOS_ID)

        limit = int(max_len)
        if limit > 0 and len(ids) > limit:
            ids = ids[:limit]
            if add_eos:
                ids[-1] = EOS_ID
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> str:
        """Turn ids back into text.

        Args:
            ids: iterable of ids (tensors are fine, they are cast to ``int``).
            skip_special: drop the four special ids (0..3) from the output.

        Returns:
            Tokens joined with ``" "`` (``""`` in ``char`` mode).

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
        return ("" if self.mode == "char" else " ").join(pieces)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path) -> Path:
        """Write the tokenizer as JSON, and return the file actually written.

        ``path`` may be a ``.json`` file, or an existing directory (in which case
        ``tokenizer.json`` is written inside it).  The directory is created when
        missing.
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
            "itos": list(self.itos),
        }
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return target

    @classmethod
    def load(cls, path) -> "Tokenizer":
        """Load a tokenizer previously written by :meth:`save`.

        ``path`` may be the JSON file or the directory that contains
        ``tokenizer.json``.

        Raises:
            FileNotFoundError: the file does not exist.
            ValueError: the payload is not a tokenizer object.
        """
        target = Path(path)
        if target.is_dir():
            target = target / "tokenizer.json"
        if not target.exists():
            raise FileNotFoundError(f"tokenizer file not found: {target}")
        with target.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict) or "itos" not in payload:
            raise ValueError(
                f"{target}: expected a tokenizer JSON object with an 'itos' list, "
                f"got {type(payload).__name__}"
            )
        tokenizer = cls(list(payload["itos"]), mode=str(payload.get("mode", "word")))
        tokenizer.lowercase = bool(payload.get("lowercase", True))
        return tokenizer


def build_tokenizers(
    cfg: DataConfig, pairs_by_split: Dict[str, List[Tuple[str, str]]]
) -> Tuple[Tokenizer, Tokenizer]:
    """Build the ``(source, target)`` tokenizer pair from the training split.

    Source and target get **separate** vocabularies (two languages, two token
    distributions).  Nothing is written to disk here: saving under
    ``<data_dir>/tokenizer`` is an explicit call from the training entry point.

    Args:
        cfg: data configuration (``tokenizer`` mode, ``lowercase``, ``min_freq``,
            ``max_vocab``).
        pairs_by_split: ``{split: [(src, tgt), ...]}``; only ``"train"`` is used.

    Returns:
        ``(src_tokenizer, tgt_tokenizer)``.

    Raises:
        ValueError: there is no non-empty ``"train"`` entry to learn from.
    """
    train = list(pairs_by_split.get("train") or [])
    if not train:
        raise ValueError(
            f"cannot build a vocabulary: pairs_by_split has no 'train' pairs "
            f"(keys={sorted(pairs_by_split)})"
        )
    src_tokenizer = Tokenizer.build(
        (src for src, _ in train),
        mode=cfg.tokenizer,
        min_freq=cfg.min_freq,
        max_vocab=cfg.max_vocab,
        lowercase=cfg.lowercase,
    )
    tgt_tokenizer = Tokenizer.build(
        (tgt for _, tgt in train),
        mode=cfg.tokenizer,
        min_freq=cfg.min_freq,
        max_vocab=cfg.max_vocab,
        lowercase=cfg.lowercase,
    )
    return src_tokenizer, tgt_tokenizer
