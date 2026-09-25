"""
Symmetry-breaking encoder-decoder Transformer with an energy-conserving optimizer.

A small, readable seq2seq Transformer built to study one question: what happens
to learning when a fixed bias ``b`` is added to the embedding, breaking the
``O(n_embd)`` rotation symmetry that attention would otherwise respect?

Layout::

    config.py        dataclass + PRESETS (model sizes) and BiasPresets
    optimizer.py     EGD (energy-conserving descent) + AdamW / SGD baselines
    utils.py         device, seeding, logging, stall watchdog, optimizer reporting
    evaluate.py      validation loss, greedy decoding, corpus BLEU
    bias.py          the embedding bias b and the per-head bQ / bK / bV
    model/           embedding, attention, feedforward, encoder, decoder, seq2seq
    data/            dataset download, tokenizer, batching

Entry points live in ``scripts/`` and are driven by the single user-facing menu in
``run.py`` (double-click ``run.bat``); nothing here is meant to be launched by
hand::

    run.bat            # or: python run.py   ->  numbered menu, no configuration
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
