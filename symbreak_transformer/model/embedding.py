"""Token + positional embedding, with the optional symmetry-breaking bias ``b``.

The embedding is the *first* place a global rotation of the ``n_embd`` space can
be broken, so the embedding bias
:class:`~symbreak_transformer.bias.EmbeddingBias` lives here.  With ``b = 0`` the
embedding is equivariant under ``x -> R x`` for any orthogonal ``R``; with
``b != 0`` it is not, because ``R (x + b) = Rx + Rb`` and ``Rb != b`` unless ``R``
fixes ``b``.

Order of operations (fixed by the interface contract, do not reorder):

1. ``x = wte(tokens)``
2. ``x = x * sqrt(n_embd)``  (only when ``scale``)
3. ``x = x + wpe(arange(T))``  (learned table by default, sinusoids otherwise)
4. ``x = bias(x)``  -- the user-visible ``x + b`` (skipped when ``bias is None``)
5. dropout
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from ..bias import EmbeddingBias

__all__ = ["TokenPositionalEmbedding", "sinusoidal_positions"]


def sinusoidal_positions(context_length: int, n_embd: int) -> torch.Tensor:
    """Classic fixed sinusoidal position table, shape ``(context_length, n_embd)``.

    Row ``p`` is ``[sin(p / 10000^(2i/d)), cos(p / 10000^(2i/d)), ...]`` in the
    usual interleaved layout: even indices take the sine, odd indices the cosine.
    Entries stay in ``[-1, 1]``; the caller adds the table to an already
    ``sqrt(n_embd)``-scaled token embedding.
    """
    if context_length < 1:
        raise ValueError(f"context_length must be >= 1, got {context_length}")
    if n_embd < 1:
        raise ValueError(f"n_embd must be >= 1, got {n_embd}")
    position = torch.arange(context_length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, n_embd, 2, dtype=torch.float32)
        * (-math.log(10000.0) / n_embd)
    )
    table = torch.zeros(context_length, n_embd, dtype=torch.float32)
    table[:, 0::2] = torch.sin(position * div_term)
    if n_embd > 1:
        table[:, 1::2] = torch.cos(position * div_term[: table[:, 1::2].shape[1]])
    return table


class TokenPositionalEmbedding(nn.Module):
    """Embedding of ``(B, T)`` token ids into ``(B, T, n_embd)``.

    Args:
        vocab_size: number of token ids the table covers.
        n_embd: embedding width.
        context_length: length of the learned positional table / sinusoid table.
        dropout: dropout probability applied to the final embedding.
        scale: multiply the token embedding by ``sqrt(n_embd)`` (Vaswani et al.).
        bias: optional :class:`~symbreak_transformer.bias.EmbeddingBias`; when
            given, ``x = x + b`` is applied *after* the positional term and
            *before* dropout.  This is the symmetry-breaking bias ``b``.
        learned_positional: ``True`` (default) uses an ``nn.Embedding`` position
            table; ``False`` uses the fixed sinusoidal table instead.

    Attributes:
        wte: the token embedding table.
        wpe: the learned position table (``None`` when ``learned_positional``).
    """

    def __init__(
        self,
        vocab_size: int,
        n_embd: int,
        context_length: int,
        dropout: float = 0.0,
        scale: bool = True,
        bias: Optional[EmbeddingBias] = None,
        learned_positional: bool = True,
    ) -> None:
        super().__init__()
        if vocab_size < 1:
            raise ValueError(f"vocab_size must be >= 1, got {vocab_size}")
        if n_embd < 1:
            raise ValueError(f"n_embd must be >= 1, got {n_embd}")
        if context_length < 1:
            raise ValueError(f"context_length must be >= 1, got {context_length}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.vocab_size = int(vocab_size)
        self.n_embd = int(n_embd)
        self.context_length = int(context_length)
        self.scale = bool(scale)
        self.learned_positional = bool(learned_positional)
        self.embed_scale = math.sqrt(self.n_embd) if self.scale else 1.0

        self.wte = nn.Embedding(self.vocab_size, self.n_embd)
        if self.learned_positional:
            self.wpe: Optional[nn.Embedding] = nn.Embedding(
                self.context_length, self.n_embd
            )
            self.register_buffer("_pos_table", None, persistent=False)
        else:
            self.wpe = None
            self.register_buffer(
                "_pos_table",
                sinusoidal_positions(self.context_length, self.n_embd),
                persistent=False,
            )

        self.bias = bias
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------ #
    @property
    def bias_norm(self) -> float:
        """L2 norm of the embedding bias ``b``; exactly ``0.0`` when there is none."""
        if self.bias is None:
            return 0.0
        return float(self.bias.b.detach().float().norm())

    # ------------------------------------------------------------------ #
    def _positions(self, length: int, device: torch.device) -> torch.Tensor:
        """Return the positional term ``(T, n_embd)`` for a sequence of ``length``."""
        if length > self.context_length:
            raise ValueError(
                f"sequence length {length} exceeds context_length "
                f"{self.context_length}; raise model.context_length or clip the data"
            )
        if self.learned_positional:
            idx = torch.arange(length, device=device)
            return self.wpe(idx)  # type: ignore[misc]
        return self._pos_table[:length].to(device=device, dtype=torch.float32)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed ``tokens``.

        Args:
            tokens: ``(B, T)`` long tensor of token ids.

        Returns:
            ``(B, T, n_embd)`` float tensor.
        """
        if tokens.dim() != 2:
            raise ValueError(
                f"TokenPositionalEmbedding expects (B, T) token ids, "
                f"got shape {tuple(tokens.shape)}"
            )
        if tokens.dtype not in (torch.int32, torch.int64):
            raise TypeError(
                f"tokens must be an integer tensor, got dtype {tokens.dtype}"
            )
        length = int(tokens.shape[1])
        x = self.wte(tokens)
        if self.scale:
            x = x * self.embed_scale
        x = x + self._positions(length, tokens.device).unsqueeze(0)
        if self.bias is not None:
            # ``x + b``: the symmetry-breaking offset.  A ``zero``-mode bias adds
            # an exact zero, so applying it unconditionally is a no-op then.
            x = self.bias(x)
        return self.dropout(x)

    def extra_repr(self) -> str:
        return (
            f"vocab_size={self.vocab_size}, n_embd={self.n_embd}, "
            f"context_length={self.context_length}, scale={self.scale}, "
            f"learned_positional={self.learned_positional}, "
            f"bias={'none' if self.bias is None else self.bias.mode}"
        )
