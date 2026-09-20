"""Decoder stack: self-attention + cross-attention + feed-forward, per layer.

The decoder owns two attention sites that can each carry their own
``bQ``/``bK``/``bV`` biases:

* self-attention (masked causally, ``self_attn_bias``),
* cross-attention over the encoder memory (``cross_attn_bias``).

They are supplied by two independent factories so
``BiasConfig.apply_decoder_self`` / ``apply_decoder_cross`` can switch them on
separately, and so ``share_across_layers`` can share one object within a site
while the two sites stay distinct.

Same development rule as :mod:`symbreak_transformer.model.encoder`: layers are
**built in a loop** and never ``copy.deepcopy``-ed, otherwise a shared
``AttentionBias`` would be cloned per layer and ``share_across_layers`` would stop
working.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..bias import AttentionBias
from ..config import Seq2SeqConfig
from .attention import MultiHeadAttention
from .encoder import BiasFactory
from .feedforward import FeedForward

__all__ = ["DecoderLayer", "Decoder"]


class DecoderLayer(nn.Module):
    """One post-LN or pre-LN decoder layer.

    Args:
        d_model: model width.
        n_head: attention heads.
        d_ff: feed-forward hidden width.
        dropout: dropout used inside the residual branches.
        attention_dropout: dropout inside both attention softmaxes.
        activation: feed-forward activation (``gelu``/``relu``/``prelu``).
        norm_first: pre-LN when ``True``, post-LN when ``False``.
        self_attn_bias: optional bias for the masked self-attention.
        cross_attn_bias: optional bias for the encoder cross-attention.
        prelu_random_init: use :class:`RandomPReLU1d` instead of ``nn.PReLU``.
        prelu_slope_mean: mean of the random PReLU slopes.
        prelu_slope_std: std of the random PReLU slopes.
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation: str = "gelu",
        norm_first: bool = False,
        self_attn_bias: Optional[AttentionBias] = None,
        cross_attn_bias: Optional[AttentionBias] = None,
        prelu_random_init: bool = False,
        prelu_slope_mean: float = 0.2,
        prelu_slope_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.norm_first = bool(norm_first)
        self.self_attn = MultiHeadAttention(
            d_model, n_head, dropout=attention_dropout, bias=self_attn_bias
        )
        self.cross_attn = MultiHeadAttention(
            d_model, n_head, dropout=attention_dropout, bias=cross_attn_bias
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
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            tgt: ``(B, T, d_model)`` decoder inputs.
            memory: ``(B, S, d_model)`` encoder outputs.
            tgt_mask: bool causal mask, ``True`` = BLOCKED.
            tgt_key_padding_mask: bool ``(B, T)``, ``True`` = PAD.
            memory_key_padding_mask: bool ``(B, S)``, ``True`` = PAD.
        """
        if self.norm_first:
            x = self.norm1(tgt)
            tgt = tgt + self.dropout1(
                self.self_attn(
                    x,
                    x,
                    x,
                    key_padding_mask=tgt_key_padding_mask,
                    attn_mask=tgt_mask,
                )
            )
            x = self.norm2(tgt)
            tgt = tgt + self.dropout2(
                self.cross_attn(
                    x,
                    memory,
                    memory,
                    key_padding_mask=memory_key_padding_mask,
                )
            )
            tgt = tgt + self.dropout3(self.ff(self.norm3(tgt)))
        else:
            tgt = self.norm1(
                tgt
                + self.dropout1(
                    self.self_attn(
                        tgt,
                        tgt,
                        tgt,
                        key_padding_mask=tgt_key_padding_mask,
                        attn_mask=tgt_mask,
                    )
                )
            )
            tgt = self.norm2(
                tgt
                + self.dropout2(
                    self.cross_attn(
                        tgt,
                        memory,
                        memory,
                        key_padding_mask=memory_key_padding_mask,
                    )
                )
            )
            tgt = self.norm3(tgt + self.dropout3(self.ff(tgt)))
        return tgt


class Decoder(nn.Module):
    """``num_layers`` :class:`DecoderLayer` modules applied in sequence.

    Args:
        cfg: the model configuration (shape, dropout, activation, PReLU knobs).
        self_attn_bias_factory: called once per layer for the self-attention bias.
        cross_attn_bias_factory: called once per layer for the cross-attention bias.
        num_layers: defaults to ``cfg.n_decoder_layer``.

    Attributes:
        layers: the ``nn.ModuleList`` of layers.
        decoder_norm: final ``nn.LayerNorm``, present **only** for the pre-LN
            (``norm_first=True``) variant.
    """

    def __init__(
        self,
        cfg: Seq2SeqConfig,
        self_attn_bias_factory: BiasFactory,
        cross_attn_bias_factory: BiasFactory,
        num_layers: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        n_layers = int(num_layers if num_layers is not None else cfg.n_decoder_layer)
        if n_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {n_layers}")

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=cfg.n_embd,
                    n_head=cfg.n_head,
                    d_ff=cfg.d_ff,
                    dropout=cfg.dropout,
                    attention_dropout=cfg.attention_dropout,
                    activation=cfg.activation,
                    norm_first=cfg.norm_first,
                    self_attn_bias=self_attn_bias_factory(),
                    cross_attn_bias=cross_attn_bias_factory(),
                    prelu_random_init=cfg.prelu_random_init,
                    prelu_slope_mean=cfg.prelu_slope_mean,
                    prelu_slope_std=cfg.prelu_slope_std,
                )
                for _ in range(n_layers)
            ]
        )
        self.decoder_norm: Optional[nn.LayerNorm] = (
            nn.LayerNorm(cfg.n_embd) if cfg.norm_first else None
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the stack; see :meth:`DecoderLayer.forward` for the argument shapes."""
        for layer in self.layers:
            tgt = layer(
                tgt,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        if self.decoder_norm is not None:
            tgt = self.decoder_norm(tgt)
        return tgt

    def attn_biases(self) -> List[AttentionBias]:
        """The distinct attention-bias objects used by this stack (by identity)."""
        seen: dict = {}
        for layer in self.layers:
            for b in (layer.self_attn.bias, layer.cross_attn.bias):
                if b is not None:
                    seen[id(b)] = b
        return list(seen.values())
