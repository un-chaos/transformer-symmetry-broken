"""Model package: the encoder-decoder Transformer and its building blocks.

Public API::

    from symbreak_transformer.model import (
        TokenPositionalEmbedding,   # token + positional embedding (+b)
        sinusoidal_positions,       # fixed sinusoidal position table
        MultiHeadAttention,         # explicit q/k/v/o attention with bQ/bK/bV
        FeedForward,                # position-wise MLP
        RandomPReLU1d,              # PReLU with per-feature random slopes
        EncoderLayer,               # self-attention + MLP
        Encoder,                    # n_encoder_layer EncoderLayer modules
        DecoderLayer,               # self-attention + cross-attention + MLP
        Decoder,                    # n_decoder_layer DecoderLayer modules
        BiasFactory,                # Callable[[], AttentionBias | None]
        Seq2SeqTransformer,         # the full model
    )

``Seq2SeqTransformer`` owns the embeddings (and therefore the embedding bias
``b``); the attention biases are handed to the stacks by the factories in
``seq2seq.py``.  See ``seq2seq.py`` for the placement rules and
``bias.py`` for the two bias families.
"""

from __future__ import annotations

from .attention import MultiHeadAttention
from .decoder import Decoder, DecoderLayer
from .embedding import TokenPositionalEmbedding, sinusoidal_positions
from .encoder import BiasFactory, Encoder, EncoderLayer
from .feedforward import FeedForward, RandomPReLU1d
from .seq2seq import Seq2SeqTransformer

__all__ = [
    # embedding.py
    "TokenPositionalEmbedding",
    "sinusoidal_positions",
    # attention.py
    "MultiHeadAttention",
    # feedforward.py
    "FeedForward",
    "RandomPReLU1d",
    # encoder.py
    "BiasFactory",
    "EncoderLayer",
    "Encoder",
    # decoder.py
    "DecoderLayer",
    "Decoder",
    # seq2seq.py
    "Seq2SeqTransformer",
]
