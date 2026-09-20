"""Encoder stack: a loop of freshly built :class:`EncoderLayer` modules.

Normalisation placement:

``norm_first=False`` (post-LN, the Vaswani et al. original)
    ``x = ln(x + sublayer(x))`` -- the normalisation is applied to the *output*
    of the residual add.  No final ``encoder_norm`` is attached in this variant,
    because the last layer already ends with a LayerNorm.

``norm_first=True`` (pre-LN, the modern/trainable variant)
    ``x = x + sublayer(ln(x))``, followed by a final ``encoder_norm``.  The final
    norm is attached **only** in this variant (it is essential here: without it
    the residual stream would leave the stack unnormalised, and it would be
    redundant in the post-LN case).
"""

from __future__ import annotations

from typing import Callable, List, Optional

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .bias import AttentionBias
from .config import ModelConfig
from .feedforward import FeedForward

__all__ = ["BiasFactory", "EncoderLayer", "Encoder"]

#: Called once per layer; returns the ``AttentionBias`` that layer should use
#: (or ``None``).  Returning one shared instance reproduces
#: ``AttentionBiasConfig.share_across_layers``.
BiasFactory = Callable[[], Optional[AttentionBias]]


class EncoderLayer(nn.Module):
    """One post-LN or pre-LN encoder layer: self-attention + feed-forward.

    Args:
        d_model: model width.
        n_heads: attention heads.
        d_ff: feed-forward hidden width.
        dropout: dropout used inside the residual branches.
        attention_dropout: dropout inside the attention softmax.
        activation: feed-forward activation (``gelu``/``relu``/``prelu``).
        norm_first: pre-LN when ``True``, post-LN when ``False``.
        attn_bias: optional :class:`~transformer_sym.bias.AttentionBias` applied
            to the self-attention q/k/v.
        prelu_random_init: use :class:`RandomPReLU1d` instead of ``nn.PReLU``.
        prelu_slope_mean: mean of the random PReLU slopes.
        prelu_slope_std: std of the random PReLU slopes.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = False,
        attn_bias: Optional[AttentionBias] = None,
        prelu_random_init: bool = False,
        prelu_slope_mean: float = 0.2,
        prelu_slope_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm_first = bool(norm_first)
        self.self_attn = MultiHeadAttention(
            d_model, n_heads, dropout=attention_dropout, bias=attn_bias
        )
        self.ff = FeedForward(
            d_model,
            d_ff,
            dropout=dropout,
            activation=activation,
            prelu_random_init=prelu_random_init,
            prelu_slope_mean=prelu_slope_mean,
            prelu_slope_std=prelu_slope_std,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args: ``src`` ``(B, S, d_model)``, bool mask ``(B, S)`` (True = PAD)."""
        if self.norm_first:
            src = src + self.dropout1(
                self.self_attn(
                    self.norm1(src),
                    self.norm1(src),
                    self.norm1(src),
                    key_padding_mask=src_key_padding_mask,
                )
            )
            src = src + self.dropout2(self.ff(self.norm2(src)))
        else:
            src = self.norm1(
                src
                + self.dropout1(
                    self.self_attn(
                        src, src, src, key_padding_mask=src_key_padding_mask
                    )
                )
            )
            src = self.norm2(src + self.dropout2(self.ff(src)))
        return src


class Encoder(nn.Module):
    """``num_layers`` :class:`EncoderLayer` modules applied in sequence.

    Args:
        cfg: the model configuration; ``cfg.activation``, ``cfg.norm_first``,
            ``cfg.dropout``, ``cfg.attention_dropout`` and the PReLU knobs are
            all read from it.
        bias_factory: called **once per layer** to obtain that layer's
            attention bias.  With ``share_across_layers=True`` the factory
            returns the same :class:`~transformer_sym.bias.AttentionBias`
            instance every time, which is exactly why this class must never
            ``copy.deepcopy`` a layer: a deep copy would clone the bias buffers
            and silently break the sharing (the copies would then be resampled
            independently, and ``share_across_layers`` would be a lie).
        num_layers: defaults to ``cfg.n_encoder_layers``.

    Attributes:
        layers: the ``nn.ModuleList`` of layers.
        encoder_norm: final ``nn.LayerNorm`` -- present **only** for the pre-LN
            (``norm_first=True``) variant, ``None`` otherwise.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        bias_factory: BiasFactory,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        n_layers = int(num_layers if num_layers is not None else cfg.n_encoder_layers)
        if n_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {n_layers}")

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=cfg.d_model,
                    n_heads=cfg.n_heads,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    attention_dropout=cfg.attention_dropout,
                    activation=cfg.activation,
                    norm_first=cfg.norm_first,
                    attn_bias=bias_factory(),
                    prelu_random_init=cfg.prelu_random_init,
                    prelu_slope_mean=cfg.prelu_slope_mean,
                    prelu_slope_std=cfg.prelu_slope_std,
                )
                for _ in range(n_layers)
            ]
        )
        # Pre-LN only: the residual stream leaves the stack unnormalised.
        self.encoder_norm: Optional[nn.LayerNorm] = (
            nn.LayerNorm(cfg.d_model) if cfg.norm_first else None
        )

    def forward(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args: ``src`` ``(B, S, d_model)``, bool mask ``(B, S)`` (True = PAD)."""
        for layer in self.layers:
            src = layer(src, src_key_padding_mask=src_key_padding_mask)
        if self.encoder_norm is not None:
            src = self.encoder_norm(src)
        return src

    def attn_biases(self) -> List[AttentionBias]:
        """The distinct attention-bias objects used by this stack (by identity)."""
        seen: dict = {}
        for layer in self.layers:
            b = layer.self_attn.bias
            if b is not None:
                seen[id(b)] = b
        return list(seen.values())
