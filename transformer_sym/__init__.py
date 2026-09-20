"""Symmetry-broken encoder-decoder Transformer with an energy-conserving optimizer.

The package is deliberately split one-concept-per-file so each piece can be read
and modified in isolation:

======================  =====================================================
File                    Responsibility
======================  =====================================================
``config.py``           Typed configuration objects + YAML load/save.
``bias.py``             The embedding bias ``b`` and per-head ``bQ/bK/bV``.
``embedding.py``        Token + positional embedding (applies ``b``).
``attention.py``        Multi-head attention (self & cross) with optional biases.
``feedforward.py``      Position-wise MLP.
``encoder.py``          Encoder layer + encoder stack.
``decoder.py``          Decoder layer + decoder stack.
``model.py``            ``Seq2SeqTransformer`` tying encoder and decoder together.
``optimizer.py``        EGD (energy-conserving descent) and an AdamW baseline.
``train.py``            Training loop (closure-based for EGD).
``evaluate.py``         Validation loss, greedy decoding, corpus BLEU.
``utils.py``            Seeding, CSV logging, timing, small helpers.
``data/``               Dataset download (HuggingFace mirror), tokenizer, batching.
======================  =====================================================

Physics/motivation
------------------
Attention only ever sees the embedding ``x`` through the linear maps
``W_q, W_k, W_v``.  With ``b = 0`` the whole Q-K sector is invariant under a
global rotation of the embedding space, ``x -> R x`` with ``R`` orthogonal:
``R`` can be absorbed into the weights, so the model has an ``O(d_model)``
symmetry.  Adding a *fixed* bias, ``x -> x + b`` with ``b != 0``, is not
covariant under that rotation (``R(x + b) = Rx + Rb != Rx + b``), so a non-zero
``b`` explicitly breaks the rotational symmetry of the embedding.  ``b`` is a
non-learned buffer by default, controlled entirely from the config.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
