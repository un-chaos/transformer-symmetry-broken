"""Hand-written multi-head attention with optional ``bQ``/``bK``/``bV`` biases.

The point of this module is that the bias injection stays *visible*: no
``nn.MultiheadAttention``, no ``scaled_dot_product_attention``, just explicit
projections, an explicit score matrix, an explicit mask and a softmax.

Mask polarity (fixed by the interface contract -- getting this backwards is the
classic silent bug):

* ``key_padding_mask`` -- bool ``(B, S)``, **``True`` means PAD / ignore**.
* ``attn_mask`` -- bool broadcastable to ``(B, Tq, S)``, **``True`` means
  BLOCKED**.

Float (additive) masks are deliberately unsupported and raise ``TypeError``.
Blocked positions are filled with ``torch.finfo(scores.dtype).min`` rather than
``-inf``, so a fully blocked row cannot produce ``NaN``; such a row additionally
has its first column un-blocked and pinned to logit ``0``, so its softmax is a
clean finite one-hot on column 0 rather than a uniform average over blocked
positions.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bias import AttentionBias

__all__ = ["MultiHeadAttention"]


def _validate_bool_mask(mask: torch.Tensor, name: str) -> None:
    """Reject float masks: additive masking is not part of this contract."""
    if mask.dtype != torch.bool:
        raise TypeError(
            f"{name} must be a bool tensor (True = "
            f"{'PAD/ignore' if name == 'key_padding_mask' else 'BLOCKED'}), "
            f"got dtype {mask.dtype}. Float/additive masks are not supported."
        )


class MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention.

    Args:
        d_model: model width; must be divisible by ``n_head``.
        n_head: number of attention heads.
        dropout: dropout on the attention probabilities.
        bias: optional :class:`~symbreak_transformer.bias.AttentionBias`; when it
            is active, ``bQ``/``bK``/``bV`` are added to the per-head q/k/v
            tensors right before the scores are computed (reference behaviour).

    Attributes:
        head_dim: ``d_model // n_head``.
        q_proj, k_proj, v_proj, out_proj: separate ``nn.Linear`` layers (each
            with its own bias), so no fused ``in_proj_weight`` hides the wiring.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.0,
        bias: Optional[AttentionBias] = None,
    ) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_head ({n_head})"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = int(d_model)
        self.n_head = int(n_head)
        self.head_dim = self.d_model // self.n_head
        self.dropout_p = float(dropout)

        self.q_proj = nn.Linear(self.d_model, self.d_model)
        self.k_proj = nn.Linear(self.d_model, self.d_model)
        self.v_proj = nn.Linear(self.d_model, self.d_model)
        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.bias = bias
        self.attn_dropout = nn.Dropout(self.dropout_p)

    # ------------------------------------------------------------------ #
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, T, d_model)`` -> ``(B, n_head, T, head_dim)``."""
        batch, length, _ = x.shape
        return x.view(batch, length, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, n_head, T, head_dim)`` -> ``(B, T, d_model)``."""
        batch, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, length, self.d_model)

    @staticmethod
    def _combine_masks(
        scores: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor],
        attn_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Return a bool ``blocked`` mask broadcastable to ``scores``, or ``None``.

        Combines the two mask conventions (``True`` = PAD and ``True`` = BLOCKED,
        both meaning "do not attend") and never edits the caller's tensors.
        """
        blocked: Optional[torch.Tensor] = None
        if attn_mask is not None:
            _validate_bool_mask(attn_mask, "attn_mask")
            blocked = attn_mask
        if key_padding_mask is not None:
            _validate_bool_mask(key_padding_mask, "key_padding_mask")
            pad = key_padding_mask[:, None, None, :]  # (B, 1, 1, S)
            blocked = pad if blocked is None else (blocked | pad)
        if blocked is None:
            return None
        # ``expand`` (not ``repeat``) keeps this allocation-free; ``masked_fill``
        # accepts a broadcastable mask.
        return blocked.expand_as(scores)

    @staticmethod
    def _attend(scores: torch.Tensor, blocked: torch.Tensor) -> torch.Tensor:
        """Softmax over the last axis after applying the ``blocked`` mask.

        ``blocked`` is a bool tensor already broadcast to ``scores``' shape, with
        ``True`` meaning "do not attend".
        """
        min_val = torch.finfo(scores.dtype).min
        # Rows with every position blocked (an all-pad query row, or an all-True
        # attn_mask row) would otherwise softmax to a *uniform* average over
        # blocked positions.  Give them a valid, finite, one-hot-ish distribution
        # concentrated on column 0 instead.
        fully = blocked.all(dim=-1, keepdim=True)  # (..., Tq, 1)
        if bool(fully.any()):
            first = torch.zeros_like(blocked)
            first[..., 0] = True  # column-0 selector
            # Un-block column 0 for those rows only, so its *real* logit is what
            # the softmax sees...
            masked = blocked & ~(fully & first)
            scores = scores.masked_fill(masked, min_val)
            # ...and pin that logit to 0.0, which turns "softmax over a row of
            # min_val" into a clean one-hot on column 0.  A ``torch.where`` rather
            # than an in-place write keeps autograd happy.
            scores = torch.where(fully & first, torch.zeros_like(scores), scores)
        else:
            scores = scores.masked_fill(blocked, min_val)

        if scores.dtype in (torch.float16, torch.bfloat16):
            return F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
        return F.softmax(scores, dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Attend from ``query`` to ``key``/``value``.

        Args:
            query: ``(B, Tq, d_model)``.
            key: ``(B, S, d_model)``.
            value: ``(B, S, d_model)``.
            key_padding_mask: bool ``(B, S)``; ``True`` marks a PAD key that must
                be ignored.
            attn_mask: bool tensor broadcastable to ``(B, Tq, S)``; ``True``
                marks a blocked query/key pair (e.g. the causal mask).

        Returns:
            ``(B, Tq, d_model)``.
        """
        if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
            raise ValueError(
                "MultiHeadAttention expects (B, T, d_model) tensors, got "
                f"query {tuple(query.shape)}, key {tuple(key.shape)}, "
                f"value {tuple(value.shape)}"
            )
        if key.shape[:-1] != value.shape[:-1]:
            raise ValueError(
                f"key and value must share (B, S), got {tuple(key.shape)} and "
                f"{tuple(value.shape)}"
            )

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        if self.bias is not None and self.bias.active:
            # bQ / bK / bV enter here, on the (B, n_head, T, head_dim) tensors.
            q, k, v = self.bias.apply(q, k, v)

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        blocked = self._combine_masks(scores, key_padding_mask, attn_mask)
        if blocked is None:
            if scores.dtype in (torch.float16, torch.bfloat16):
                attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(scores.dtype)
            else:
                attn = F.softmax(scores, dim=-1)
        else:
            attn = self._attend(scores, blocked)

        attn = self.attn_dropout(attn)
        out = attn @ v
        return self.out_proj(self._merge_heads(out))

    @torch.no_grad()
    def attention_probs(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the post-softmax attention weights ``(B, n_head, Tq, S)``.

        Exposed for diagnostics and tests: it is the only way to inspect the
        block mask's effect (and the fully-masked-row fallback) directly, since
        ``forward`` multiplies the weights by ``V`` before returning.  Dropout is
        not applied.
        """
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        if self.bias is not None and self.bias.active:
            q = self.bias.q(q)
            k = self.bias.k(k)
        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        blocked = self._combine_masks(scores, key_padding_mask, attn_mask)
        if blocked is None:
            return F.softmax(scores, dim=-1)
        return self._attend(scores, blocked)

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, n_head={self.n_head}, "
            f"head_dim={self.head_dim}, dropout={self.dropout_p}, "
            f"bias={'none' if self.bias is None else ('active' if self.bias.active else 'inactive')}"
        )
