"""``Seq2SeqTransformer`` -- the encoder-decoder model, wired from the config.

Layout::

    src tokens --> TokenPositionalEmbedding(+b) --> Encoder ----------------+
                                                                          |
    tgt tokens --> TokenPositionalEmbedding(+b) --> Decoder <-------------+
                                                      |
                                                      +--> output_proj --> logits

Embedding-bias placement (the decision this module implements, and the one
:meth:`Seq2SeqTransformer.bias_report` states):

* ``b`` is applied to the **source** embedding, and
* to the **target** embedding as well **whenever ``cfg.bias.embed.mode != "zero"``**
  (a ``zero``-mode bias is an exact zero offset, so attaching it to the target
  would only add dead state to the checkpoint).
* When ``src_vocab_size == tgt_vocab_size`` the two embeddings share **one and
  the same** :class:`~transformer_sym.bias.EmbeddingBias` instance -- the same
  ``b`` is added on both sides, matching the "one bias for the model" intent.
  When the vocabularies differ, each side gets its own instance built from the
  same config (seeds offset by one so the draws are independent but reproducible).

Attention-bias placement: three independent factories
(encoder self-attention, decoder self-attention, decoder cross-attention) are
built from :class:`~transformer_sym.config.AttentionBiasConfig`; see
:meth:`Seq2SeqTransformer._attention_bias_factory`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bias import AttentionBias, EmbeddingBias
from .config import BiasConfig, ModelConfig
from .decoder import Decoder
from .embedding import TokenPositionalEmbedding
from .encoder import BiasFactory, Encoder

__all__ = ["Seq2SeqTransformer"]

#: Modules used both in ``torch.compile`` and in :meth:`Seq2SeqTransformer._probe_pads`.
_BIAS_PROBE_SITE = "_bias_probe"


class Seq2SeqTransformer(nn.Module):
    """A small encoder-decoder Transformer with two families of symmetry breaks.

    Args:
        cfg: the model shape (:class:`~transformer_sym.config.ModelConfig`).
        bias_cfg: the bias switches (:class:`~transformer_sym.config.BiasConfig`).
        src_vocab_size: defaults to ``cfg.vocab_size_src``; a config value of
            ``0`` means "not resolved yet" and raises, because the tokenizer owns
            the real vocabulary size.
        tgt_vocab_size: same, for the target side.
        pad_id: index ignored by the loss and treated as PAD by the masks.
        bos_id: decoder start token used by :meth:`greedy_decode`.
        eos_id: token that stops a row in :meth:`greedy_decode`.

    Attributes:
        src_embed, tgt_embed: :class:`~transformer_sym.embedding.TokenPositionalEmbedding`
            modules carrying the ``b`` offset.
        encoder, decoder: the two stacks.
        output_proj: ``d_model -> tgt_vocab``; ``bias=False`` so its weight can be
            tied to the target embedding matrix when ``cfg.tie_output_embedding``.
    """

    def __init__(
        self,
        cfg: ModelConfig,
        bias_cfg: BiasConfig,
        src_vocab_size: Optional[int] = None,
        tgt_vocab_size: Optional[int] = None,
        pad_id: int = 0,
        bos_id: int = 2,
        eos_id: int = 3,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.bias_cfg = bias_cfg

        src_vocab = self._resolve_vocab_size(
            src_vocab_size if src_vocab_size is not None else cfg.vocab_size_src,
            "src_vocab_size",
            "cfg.vocab_size_src",
        )
        tgt_vocab = self._resolve_vocab_size(
            tgt_vocab_size if tgt_vocab_size is not None else cfg.vocab_size_tgt,
            "tgt_vocab_size",
            "cfg.vocab_size_tgt",
        )

        if cfg.share_embeddings and src_vocab != tgt_vocab:
            raise ValueError(
                "cfg.share_embeddings=True requires identical vocabularies, got "
                f"src_vocab_size={src_vocab} and tgt_vocab_size={tgt_vocab}"
            )

        self.src_vocab_size = src_vocab
        self.tgt_vocab_size = tgt_vocab
        self.pad_id = int(pad_id)
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)

        # ---------------- embeddings + the embedding bias b ---------------- #
        embed_cfg = bias_cfg.embed
        bias_on_target = embed_cfg.mode != "zero"
        if embed_cfg.mode != "zero" and src_vocab == tgt_vocab:
            # One shared b for both sides: the same offset in the same space.
            src_bias: Optional[EmbeddingBias] = EmbeddingBias(cfg.d_model, embed_cfg)
            tgt_bias: Optional[EmbeddingBias] = src_bias
        elif embed_cfg.mode != "zero":
            # Separate vocabularies: two instances from one config, seeds offset so
            # the draws differ yet stay reproducible.
            import copy as _copy  # local: only used to derive a sibling config

            src_bias = EmbeddingBias(cfg.d_model, embed_cfg)
            sibling = _copy.copy(embed_cfg)
            sibling.seed = int(embed_cfg.seed) + 1
            tgt_bias = EmbeddingBias(cfg.d_model, sibling)
        else:
            src_bias = EmbeddingBias(cfg.d_model, embed_cfg)
            tgt_bias = None

        if cfg.share_embeddings:
            shared = TokenPositionalEmbedding(
                src_vocab,
                cfg.d_model,
                cfg.max_seq_len,
                dropout=cfg.dropout,
                scale=cfg.scale_embedding,
                bias=src_bias,
            )
            self.src_embed = shared
            self.tgt_embed = shared
        else:
            self.src_embed = TokenPositionalEmbedding(
                src_vocab,
                cfg.d_model,
                cfg.max_seq_len,
                dropout=cfg.dropout,
                scale=cfg.scale_embedding,
                bias=src_bias,
            )
            self.tgt_embed = TokenPositionalEmbedding(
                tgt_vocab,
                cfg.d_model,
                cfg.max_seq_len,
                dropout=cfg.dropout,
                scale=cfg.scale_embedding,
                bias=tgt_bias,
            )

        # ---------------- attention-bias factories ---------------- #
        self.encoder = Encoder(cfg, self._attention_bias_factory("encoder"))
        self.decoder = Decoder(
            cfg,
            self_attn_bias_factory=self._attention_bias_factory("decoder_self"),
            cross_attn_bias_factory=self._attention_bias_factory("decoder_cross"),
        )

        # ---------------- output projection / tying ---------------- #
        if cfg.tie_output_embedding:
            self.output_proj = nn.Linear(cfg.d_model, tgt_vocab, bias=False)
            self.output_proj.weight = self.tgt_embed.wte.weight  # type: ignore[union-attr]
        else:
            self.output_proj = nn.Linear(cfg.d_model, tgt_vocab)

        self._scale_residual_outputs()
        self._init_all_weights()
        if cfg.tie_output_embedding:
            # Re-apply: ``_init_weights`` replaced the embedding matrix, so the
            # earlier tie now points at the stale tensor.
            self.output_proj.weight = self.tgt_embed.wte.weight  # type: ignore[union-attr]

        n_params = sum(p.numel() for p in self.parameters())
        n_emb = self.tgt_embed.wte.weight.numel() + self.src_embed.wte.weight.numel()
        self._n_non_embedding_params = n_params - n_emb
        self._probe_pads: List[torch.Tensor] = []
        self.register_forward_pre_hook(self._make_pad_probe())

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _resolve_vocab_size(value: int, arg_name: str, cfg_name: str) -> int:
        size = int(value)
        if size < 1:
            raise ValueError(
                f"{arg_name} is {size}, but the vocabulary size must be >= 1. "
                f"Pass the tokenizer's vocab_size explicitly (the config value "
                f"{cfg_name} defaults to 0, which means 'not resolved yet')."
            )
        return size

    def _attention_bias_factory(self, site: str) -> BiasFactory:
        """Build the factory that decides the bias object for one attention site.

        ``site`` is one of ``"encoder"``, ``"decoder_self"``, ``"decoder_cross"``;
        it maps to ``AttentionBiasConfig.apply_encoder`` /
        ``apply_decoder_self`` / ``apply_decoder_cross``.

        * ``bias_cfg.enabled`` false, or the site's switch off -> ``lambda: None``.
        * ``share_across_layers`` true -> one instance created here and returned
          by every call to the factory.
        * ``share_across_layers`` false -> a fresh ``AttentionBias`` per call, so
          every layer has independent buffers.
        """
        bias_cfg = self.bias_cfg
        attn = bias_cfg.attention
        switch = {
            "encoder": attn.apply_encoder,
            "decoder_self": attn.apply_decoder_self,
            "decoder_cross": attn.apply_decoder_cross,
        }[site]

        if not attn.enabled or not switch:
            return lambda: None

        head_dim = self.cfg.d_model // self.cfg.n_heads
        if attn.share_across_layers:
            shared = AttentionBias(self.cfg.n_heads, head_dim, attn)
            self.add_module(f"attn_bias_{site}", shared)
            return lambda: shared
        return lambda: AttentionBias(self.cfg.n_heads, head_dim, attn)

    def _make_pad_probe(self):
        """A forward pre-hook that records the padded positions of (src, tgt_in).

        Used by :meth:`bias_report` to decide whether a shared attention bias is
        actually reachable from a padded position: if the model is *asked* to
        process padding, then "padding invariance" below is no longer a property
        of the whole attention map.
        """

        def hook(_module, inputs):
            if len(inputs) < 2:
                return
            src, tgt_in = inputs[0], inputs[1]
            if not torch.is_tensor(src) or not torch.is_tensor(tgt_in):
                return
            if self.pad_id < 0:
                return
            probe = torch.cat(
                [(src.detach() == self.pad_id).reshape(-1),
                 (tgt_in.detach() == self.pad_id).reshape(-1)]
            )
            self._probe_pads.append(probe)
            if len(self._probe_pads) > 16:
                del self._probe_pads[0]

        return hook

    # ------------------------------------------------------------------ #
    # Weight init
    # ------------------------------------------------------------------ #
    def _init_all_weights(self) -> None:
        """Apply :meth:`_init_weights` to every submodule.

        Deliberately *not* ``self.apply(...)``: :class:`~transformer_sym.bias.AttentionBias`
        defines an ``apply(q, k, v)`` method with a different meaning, which would
        shadow ``nn.Module.apply`` on that submodule and blow up the traversal.
        """
        for module in self.modules():
            self._init_weights(module)

    def _init_weights(self, module: nn.Module) -> None:
        """nanoGPT-style initialisation, applied with ``self.apply``.

        * ``nn.Linear`` -> ``N(0, init_std^2)``, additionally multiplied by
          ``(2 * (n_encoder_layers + n_decoder_layers)) ** -0.5`` when the module
          is a residual-output projection (``out_proj``, ``ff.c_proj``,
          ``output_proj``) and ``cfg.scaled_residual_init`` is set.
        * ``nn.Embedding`` -> ``N(0, init_std^2)``.  The fixed sinusoidal table is
          a non-persistent buffer, not an ``nn.Embedding``, so it is untouched.
        * ``nn.LayerNorm`` -> weight 1, bias 0.
        * Biases -> zero (the bias *config* governs the symmetry-breaking offsets,
          which are buffers/parameters of ``bias.py`` and are never re-initialised
          here).

        Note: ``AttentionBias`` subclasses ``nn.Module`` but defines its own
        ``apply(q, k, v)``, so the traversal in :meth:`_init_all_weights` avoids
        ``nn.Module.apply``.
        """
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
            if self.cfg.scaled_residual_init and getattr(
                module, "_residual_out", False
            ):
                scale = (2 * (self.cfg.n_encoder_layers + self.cfg.n_decoder_layers)) ** -0.5
                module.weight.data.mul_(scale)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

    def _scale_residual_outputs(self) -> None:
        """Mark residual-output ``nn.Linear`` modules for the scaled init."""
        for layer in self.encoder.layers:
            layer.self_attn.out_proj._residual_out = True
            layer.ff.c_proj._residual_out = True
        for layer in self.decoder.layers:
            layer.self_attn.out_proj._residual_out = True
            layer.cross_attn.out_proj._residual_out = True
            layer.ff.c_proj._residual_out = True
        self.output_proj._residual_out = True

    # ------------------------------------------------------------------ #
    # Core forward
    # ------------------------------------------------------------------ #
    def encode(
        self,
        src: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Embed + encode ``src``; returns the memory ``(B, S, d_model)``.

        Args:
            src: ``(B, S)`` long source ids.
            src_padding_mask: bool ``(B, S)``, **``True`` means PAD**.
        """
        return self.encoder(self.src_embed(src), src_key_padding_mask=src_padding_mask)

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        memory_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Decode ``tgt_in`` against ``memory`` and project to vocabulary logits.

        The causal mask is built internally, so callers cannot forget it.

        Args:
            tgt_in: ``(B, T)`` long decoder inputs (starts with BOS).
            memory: ``(B, S, d_model)`` encoder output.
            tgt_padding_mask: bool ``(B, T)``, **``True`` means PAD**.
            memory_padding_mask: bool ``(B, S)``, **``True`` means PAD**.

        Returns:
            ``(B, T, tgt_vocab_size)`` logits.
        """
        tgt_mask = self.make_causal_mask(int(tgt_in.shape[1]), tgt_in.device)
        hidden = self.decoder(
            self.tgt_embed(tgt_in),
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_padding_mask,
        )
        return self.output_proj(hidden)

    def forward(
        self,
        src: torch.Tensor,
        tgt_in: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        tgt_padding_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Encode, decode, and optionally compute the cross-entropy loss.

        Args:
            src: ``(B, S)`` source ids.
            tgt_in: ``(B, T)`` decoder input ids (starts with BOS).
            src_padding_mask: bool ``(B, S)``, ``True`` = PAD.
            tgt_padding_mask: bool ``(B, T)``, ``True`` = PAD.
            labels: ``(B, T)`` next-token targets, or ``None``.
            label_smoothing: forwarded to :func:`torch.nn.functional.cross_entropy`.

        Returns:
            ``{"logits": (B, T, tgt_vocab_size), "loss": scalar tensor | None}``.
            The loss ignores ``pad_id`` positions; it is ``None`` when ``labels``
            is ``None``.
        """
        memory = self.encode(src, src_padding_mask)
        logits = self.decode(tgt_in, memory, tgt_padding_mask, src_padding_mask)
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=self.pad_id,
                label_smoothing=label_smoothing,
            )
        return {"logits": logits, "loss": loss}

    # ------------------------------------------------------------------ #
    # Masks
    # ------------------------------------------------------------------ #
    def make_causal_mask(self, size: int, device=None) -> torch.Tensor:
        """Upper-triangular bool mask ``(size, size)`` with ``True`` = BLOCKED.

        ``torch.triu(ones, diagonal=1)``: the diagonal stays visible, everything
        strictly above it (the future) is blocked.
        """
        return torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1
        )

    def make_padding_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convenience helper: ``tokens == pad_id`` as a bool mask (``True`` = PAD)."""
        return tokens == self.pad_id

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_padding_mask: Optional[torch.Tensor] = None,
        max_len: int = 0,
        bos_id: Optional[int] = None,
        eos_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedy (argmax) decoding with early stopping.

        Args:
            src: ``(B, S)`` source ids.
            src_padding_mask: bool ``(B, S)``, ``True`` = PAD.
            max_len: maximum *total* length including the leading BOS; ``0`` means
                ``cfg.max_seq_len``.
            bos_id: start token; defaults to the model's ``bos_id``.
            eos_id: stop token; defaults to the model's ``eos_id``.

        Returns:
            ``(B, L)`` long tensor **including** the leading BOS.  Everything
            after a row's first ``eos_id`` is filled with ``pad_id``.  Rows that
            never emit EOS are simply returned at full length (no padding to add).
        """
        self.eval()
        bos = self.bos_id if bos_id is None else int(bos_id)
        eos = self.eos_id if eos_id is None else int(eos_id)
        limit = int(max_len) if max_len and max_len > 0 else int(self.cfg.max_seq_len)

        memory = self.encode(src, src_padding_mask)
        batch = int(src.shape[0])
        device = src.device

        generated = torch.full((batch, 1), bos, dtype=torch.long, device=device)
        finished = torch.zeros(batch, dtype=torch.bool, device=device)
        while generated.shape[1] < limit and not bool(finished.all()):
            logits = self.decode(generated, memory, None, src_padding_mask)
            next_token = logits[:, -1, :].argmax(dim=-1)
            next_token = torch.where(
                finished, torch.full_like(next_token, self.pad_id), next_token
            )
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
            finished = finished | (next_token == eos)
        return self._pad_after_eos(generated, eos)

    def _pad_after_eos(self, tokens: torch.Tensor, eos_id: int) -> torch.Tensor:
        """Replace every token *after* a row's first EOS with ``pad_id``.

        The EOS itself is kept.  Implemented with a cumulative count of EOS
        tokens seen *before* each position, so no Python loop over time is
        needed::

            count_before_t = cumsum(is_eos)[t] - is_eos[t]
            after_eos      = count_before_t > 0
        """
        is_eos = (tokens == eos_id).to(torch.int64)
        count_before = torch.cumsum(is_eos, dim=1) - is_eos
        return tokens.masked_fill(count_before > 0, self.pad_id)

    # ------------------------------------------------------------------ #
    # Bias housekeeping
    # ------------------------------------------------------------------ #
    def embed_biases(self) -> List[EmbeddingBias]:
        """The distinct embedding-bias instances attached to this model."""
        seen: dict = {}
        for module in (self.src_embed, self.tgt_embed):
            b = module.bias
            if b is not None:
                seen[id(b)] = b
        return list(seen.values())

    def attention_biases(self) -> List[AttentionBias]:
        """The distinct attention-bias instances attached to this model."""
        seen: dict = {}
        for stack in (self.encoder, self.decoder):
            for b in stack.attn_biases():
                if b is not None:
                    seen[id(b)] = b
        return list(seen.values())

    def resample_biases(self) -> None:
        """Redraw every ``per_step`` bias exactly once.

        Shared objects are resampled once, not once per reference: the module
        lists are de-duplicated by ``id()`` before calling ``resample()``, which
        is the whole point of ``share_across_layers`` / a shared embedding bias.
        ``mode="zero"`` biases stay at exactly zero, because their ``resample``
        redraws from the same zero distribution.
        """
        for bias in self.embed_biases():
            bias.resample()
        for bias in self.attention_biases():
            bias.resample()

    def bias_report(self) -> Dict[str, Union[float, str]]:
        """Flat report of the active biases and their norms.

        Keys (an ``embed`` block, then one block per distinct attention bias):

        * ``embed.bias_norm`` / ``embed.bias_mode`` -- norm and mode of ``b``
          (``0.0`` / ``"zero"`` when no embedding bias is attached).
        * ``embed.placed_on`` -- human-readable statement of the placement
          decision, i.e. whether ``b`` sits on the source embedding, on both
          embeddings, and whether the two share one instance.  With
          ``mode="zero"`` the target bias is not attached at all, so this reads
          ``"source only (target bias skipped because mode='zero')"`` and
          ``embed.on_target == "no"``.
        * ``padding.attention_bias_invariant`` -- ``"yes"`` unless the last
          forward pass was handed padded positions that an attention bias can
          reach (see :meth:`_padding_invariance_holds`).  Deliberately *not*
          named ``embed.*``: ``train.py::_bias_norms`` runs ``float()`` over
          every report value and would otherwise read this string as a norm.
        * ``<site>.bQ_norm`` / ``bK_norm`` / ``bV_norm`` and
          ``<site>.bQ_mode`` / ``bK_mode`` / ``bV_mode`` per attention site
          (``encoder`` / ``decoder_self`` / ``decoder_cross``, plus ``_2``,
          ``_3``, ... when layers do not share a bias object).

        All norms are finite Python floats.
        """
        report: Dict[str, Union[float, str]] = {}

        embed_biases = self.embed_biases()
        # "Shared" means: BOTH sides carry a bias AND it is literally the same
        # object.  Testing only ``len({id(...)}) == 1`` is wrong, because the
        # ``if m.bias is not None`` filter drops the target's ``None`` and leaves
        # a single id -- which would claim a shared bias while the bias sits on
        # the source only (the ``mode="zero"`` case).
        src_bias_obj = self.src_embed.bias
        tgt_bias_obj = self.tgt_embed.bias
        shared_embed = (
            src_bias_obj is not None
            and tgt_bias_obj is not None
            and src_bias_obj is tgt_bias_obj
        )
        if embed_biases:
            report["embed.bias_norm"] = float(
                embed_biases[0].b.detach().float().norm()
            )
            report["embed.bias_mode"] = str(embed_biases[0].mode)
        else:
            report["embed.bias_norm"] = 0.0
            report["embed.bias_mode"] = "zero"
        report["embed.bias_instances"] = float(len(embed_biases))
        report["embed.on_source"] = "yes" if src_bias_obj is not None else "no"
        report["embed.on_target"] = "yes" if tgt_bias_obj is not None else "no"
        report["embed.shared_between_sides"] = "yes" if shared_embed else "no"
        report["embed.placed_on"] = (
            "source + target (one shared EmbeddingBias instance)"
            if shared_embed
            else (
                "source + target (two EmbeddingBias instances from one config)"
                if tgt_bias_obj is not None
                else "source only (target bias skipped because mode='zero')"
            )
        )
        report["padding.attention_bias_invariant"] = (
            "yes" if self._padding_invariance_holds() else "no"
        )
        report["model.non_embedding_params"] = float(self._n_non_embedding_params)

        counts: Dict[str, int] = {}
        for bias in self.attention_biases():
            site = self._site_name(bias)
            index = counts.get(site, 0)
            counts[site] = index + 1
            key = site if index == 0 else f"{site}_{index + 1}"
            stats = bias.stats()
            for sector in ("Q", "K", "V"):
                report[f"{key}.b{sector}_norm"] = float(stats[f"b{sector}_norm"])
                report[f"{key}.b{sector}_mode"] = str(stats[f"b{sector}_mode"])
            report[f"{key}.enabled"] = float(bias.enabled)
        if not counts:
            report["attention.enabled"] = 0.0
            report["attention.mode"] = str(self.bias_cfg.attention.mode)
        return report

    def _site_name(self, bias: AttentionBias) -> str:
        """Recover the attention site a bias object belongs to (last match wins).

        Called in construction order (encoder layers, then decoder layers, self
        before cross), so callers append ``_2``, ``_3``, ... for repeated sites.
        """
        site = "attention"
        for layer in self.encoder.layers:
            if layer.self_attn.bias is bias:
                site = "encoder"
        for layer in self.decoder.layers:
            if layer.self_attn.bias is bias:
                site = "decoder_self"
            if layer.cross_attn.bias is bias:
                site = "decoder_cross"
        return site

    def _padding_invariance_holds(self) -> bool:
        """Whether the attention biases respect padding in the last batch seen."""
        if not self._probe_pads or not self.attention_biases():
            return True
        return not bool(torch.stack(self._probe_pads).any())

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return (
            f"src_vocab={self.src_vocab_size}, tgt_vocab={self.tgt_vocab_size}, "
            f"d_model={self.cfg.d_model}, enc={len(self.encoder.layers)}, "
            f"dec={len(self.decoder.layers)}, heads={self.cfg.n_heads}, "
            f"pad_id={self.pad_id}, tie_output={self.cfg.tie_output_embedding}"
        )
