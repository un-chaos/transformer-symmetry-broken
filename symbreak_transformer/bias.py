"""Symmetry-breaking bias modules.

Two independent families, both driven from :mod:`symbreak_transformer.config`:

``EmbeddingBias``
    The bias ``b`` the user asked for.  Applied once, to the embedding:
    ``x = Embed(tokens) * scale + b``, with ``b`` of shape ``(n_embd,)``
    broadcast over the sequence.  ``b != 0`` breaks the ``O(n_embd)`` rotation
    symmetry of the embedding, because a rotation ``R`` cannot be pushed through
    the *fixed* offset: ``R(x + b) = Rx + Rb != Rx + b``.

``AttentionBias`` / ``BiasSector``
    Reference-style per-head biases ``bQ``/``bK``/``bV`` of shape
    ``(n_head, head_dim)`` added to q/k/v.  ``bQ`` enters through the softmax so
    its effect is exponentially amplified; ``bV`` only passes through a linear
    map (power-law effect).

Both are **buffers by default** -- they are part of the experiment, not of the
learned parameters -- and only become ``nn.Parameter`` when ``learnable=True``
is passed.

The bias switches live in one flat :class:`~symbreak_transformer.config.BiasConfig`
(there is no nested embedding/attention sub-config any more), so
``EmbeddingBias`` takes the embedding-side fields as explicit keyword arguments
and ``AttentionBias`` reads the attention-side fields straight off the flat
config.
"""

from __future__ import annotations

from typing import Dict, Sequence, Union

import torch
import torch.nn as nn

from .config import BIAS_MODES, RESAMPLE_MODES, BiasConfig

__all__ = [
    "BIAS_MODES",
    "RESAMPLE_MODES",
    "resolve_vector",
    "EmbeddingBias",
    "BiasSector",
    "AttentionBias",
]


def resolve_vector(
    spec: Union[float, int, Sequence[float]],
    size: int,
    name: str = "value",
) -> torch.Tensor:
    """Turn a scalar-or-list config value into a float32 vector of length ``size``.

    A scalar is broadcast; a sequence must already have exactly ``size`` entries
    (no silent truncation, so per-dimension schedules such as
    ``torch.linspace(0.05, 0.15, head_dim)`` stay honest).
    """
    if isinstance(spec, (list, tuple)):
        if len(spec) != size:
            raise ValueError(f"{name}: expected {size} entries, got {len(spec)}")
        return torch.tensor([float(v) for v in spec], dtype=torch.float32)
    return torch.full((size,), float(spec), dtype=torch.float32)


def _validate_mode(mode: str, name: str) -> str:
    if mode not in BIAS_MODES:
        raise ValueError(f"{name}: mode must be one of {BIAS_MODES}, got {mode!r}")
    return mode


class _BiasBase(nn.Module):
    """Shared machinery: a dedicated RNG so bias draws never disturb global seeding."""

    _buffer_name: str = "b"

    def _make_generator(self, device: torch.device) -> torch.Generator:
        gen = getattr(self, "_gen", None)
        if gen is None or gen.device != device:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(self.seed))
            self._gen = gen
        return gen

    def _randn(self, shape, device: torch.device) -> torch.Tensor:
        return torch.randn(shape, device=device, generator=self._make_generator(device))

    @torch.no_grad()
    def resample(self) -> None:
        """Redraw the bias if the resample policy says so.  No-op for ``fixed``."""
        raise NotImplementedError


class EmbeddingBias(_BiasBase):
    """The additive bias ``b`` on the embedding, shape ``(n_embd,)``.

    Args:
        n_embd: embedding width.
        mode: ``"zero"``, ``"gaussian"`` or ``"const"``.
        mean: mean of the Gaussian draw (scalar or a length-``n_embd`` sequence).
        std: std of the Gaussian draw (scalar or a length-``n_embd`` sequence).
        const_value: the constant filling ``b`` when ``mode="const"``.
        resample: ``"fixed"`` (draw once at init) or ``"per_step"`` (redraw on
            every optimizer step).
        learnable: make ``b`` an ``nn.Parameter`` instead of a fixed buffer.
        seed: seed of this bias's dedicated RNG stream.
    """

    def __init__(
        self,
        n_embd: int,
        mode: str = "zero",
        mean: Union[float, Sequence[float]] = 0.0,
        std: Union[float, Sequence[float]] = 0.02,
        const_value: float = 1.0,
        resample: str = "fixed",
        learnable: bool = False,
        seed: int = 1234,
    ) -> None:
        super().__init__()
        self.n_embd = int(n_embd)
        self.mode = _validate_mode(mode, "EmbeddingBias")
        self.mean = mean
        self.std = std
        self.const_value = float(const_value)
        # NOTE: not ``self.resample`` -- that name is taken by the method below.
        self.resample_mode = resample
        self.learnable = bool(learnable)
        self.seed = int(seed)

        if self.learnable and self.resample_mode != "fixed":
            raise ValueError("EmbeddingBias: learnable=True requires resample='fixed'")

        init = self._draw(torch.device("cpu"))
        if self.learnable:
            self.b = nn.Parameter(init)
        else:
            self.register_buffer("b", init)

    # ------------------------------------------------------------------ #
    @property
    def active(self) -> bool:
        """``True`` when the bias can influence the forward pass at all."""
        return self.mode != "zero" or self.learnable

    @property
    def is_zero(self) -> bool:
        """``True`` when ``b`` is exactly zero for the current mode (no learning)."""
        return self.mode == "zero" and not self.learnable

    def _draw(self, device: torch.device) -> torch.Tensor:
        if self.mode == "zero":
            return torch.zeros(self.n_embd, device=device)
        if self.mode == "const":
            return torch.full(
                (self.n_embd,), float(self.const_value), device=device
            )
        mean = resolve_vector(self.mean, self.n_embd, "bias.embed.mean").to(device)
        std = resolve_vector(self.std, self.n_embd, "bias.embed.std").to(device)
        return mean + std * self._randn((self.n_embd,), device)

    @torch.no_grad()
    def resample(self) -> None:
        if self.resample_mode == "fixed" or self.learnable:
            return
        self.b.copy_(self._draw(self.b.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add ``b`` to ``x``; ``b`` broadcasts over batch and time dimensions."""
        return x + self.b

    def extra_repr(self) -> str:
        return (
            f"n_embd={self.n_embd}, mode={self.mode}, "
            f"resample={self.resample_mode}, learnable={self.learnable}"
        )


class BiasSector(_BiasBase):
    """One per-head additive bias (e.g. ``bQ``) of shape ``(n_head, head_dim)``.

    ``share_across_heads=True`` reproduces the reference behaviour: a single
    ``head_dim`` vector is drawn and expanded to every head.
    """

    def __init__(
        self,
        n_head: int,
        head_dim: int,
        mode: str = "zero",
        mean: Union[float, Sequence[float]] = 0.0,
        std: Union[float, Sequence[float]] = 0.05,
        const_value: float = 1.0,
        share_across_heads: bool = True,
        resample: str = "fixed",
        learnable: bool = False,
        seed: int = 1234,
        name: str = "bias",
    ):
        super().__init__()
        self.n_head = int(n_head)
        self.head_dim = int(head_dim)
        self.mode = _validate_mode(mode, name)
        self.mean = mean
        self.std = std
        self.const_value = float(const_value)
        self.share_across_heads = bool(share_across_heads)
        self.resample_mode = resample
        self.learnable = bool(learnable)
        self.seed = int(seed)
        self.name = name

        init = self._draw(torch.device("cpu"))
        if learnable:
            self.b = nn.Parameter(init)
        else:
            self.register_buffer("b", init)

    # ------------------------------------------------------------------ #
    @property
    def active(self) -> bool:
        return self.mode != "zero" or self.learnable

    def _draw(self, device: torch.device) -> torch.Tensor:
        if self.mode == "zero":
            return torch.zeros(self.n_head, self.head_dim, device=device)
        if self.mode == "const":
            return torch.full(
                (self.n_head, self.head_dim), self.const_value, device=device
            )

        mean = resolve_vector(self.mean, self.head_dim, f"{self.name}.mean").to(device)
        std = resolve_vector(self.std, self.head_dim, f"{self.name}.std").to(device)
        if self.share_across_heads:
            vec = mean + std * self._randn((self.head_dim,), device)
            return vec.unsqueeze(0).expand(self.n_head, self.head_dim).contiguous()
        return mean.view(1, -1) + std.view(1, -1) * self._randn(
            (self.n_head, self.head_dim), device
        )

    @torch.no_grad()
    def resample(self) -> None:
        if self.resample_mode == "fixed" or self.learnable:
            return
        self.b.copy_(self._draw(self.b.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add this bias to a ``(batch, n_head, time, head_dim)`` tensor."""
        if x.dim() != 4:
            raise ValueError(
                f"BiasSector expects a 4-D (B, H, T, D) tensor, got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.n_head or x.shape[-1] != self.head_dim:
            raise ValueError(
                f"BiasSector expects (B, {self.n_head}, T, {self.head_dim}), "
                f"got {tuple(x.shape)}"
            )
        return x + self.b.view(1, self.n_head, 1, self.head_dim)

    def extra_repr(self) -> str:
        return (
            f"n_head={self.n_head}, head_dim={self.head_dim}, mode={self.mode}, "
            f"share_across_heads={self.share_across_heads}, "
            f"resample={self.resample_mode}"
        )


class AttentionBias(nn.Module):
    """Container holding the ``bQ``/``bK``/``bV`` sectors for one attention site.

    A sector that is disabled is still allocated but pinned to ``zero`` mode, so
    ``apply`` stays branch-free and the state dict has a stable shape.

    Args:
        n_head: number of attention heads.
        head_dim: width of one head.
        cfg: the flat :class:`~symbreak_transformer.config.BiasConfig`; the
            ``use_q_bias`` / ``use_k_bias`` / ``use_v_bias`` switches, the
            ``mean_Q``/``std_Q`` (etc.) distributions and the ``attn_*`` knobs are
            all read from it.
    """

    def __init__(self, n_head: int, head_dim: int, cfg: BiasConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = int(n_head)
        self.head_dim = int(head_dim)
        self.enabled = bool(cfg.attention_enabled)

        # Each sector gets its own RNG stream.  Without the offset, bQ, bK and bV
        # would all be seeded identically and (with equal mean/std) would be
        # three copies of the *same* vector, silently coupling the Q-K and V-O
        # symmetry breaking.  The offsets keep the draws independent and still
        # fully reproducible from ``cfg.attn_seed``.
        seed_offset = {"q": 0, "k": 1, "v": 2}
        enabled_flag = {
            "q": cfg.use_q_bias,
            "k": cfg.use_k_bias,
            "v": cfg.use_v_bias,
        }

        def sector(letter: str, enabled: bool) -> BiasSector:
            on = self.enabled and enabled
            mode = cfg.attn_mode if on else "zero"
            return BiasSector(
                n_head=n_head,
                head_dim=head_dim,
                mode=mode,
                mean=getattr(cfg, f"mean_{letter.upper()}"),
                std=getattr(cfg, f"std_{letter.upper()}"),
                const_value=cfg.const_value,
                share_across_heads=cfg.share_across_heads,
                resample=cfg.attn_resample,
                learnable=cfg.attn_learnable and on,
                seed=int(cfg.attn_seed) + seed_offset[letter],
                name=f"b{letter.upper()}",
            )

        self.q = sector("q", enabled_flag["q"])
        self.k = sector("k", enabled_flag["k"])
        self.v = sector("v", enabled_flag["v"])

    # ------------------------------------------------------------------ #
    @property
    def active(self) -> bool:
        """``True`` when at least one sector can change the forward pass."""
        return self.enabled and any(s.active for s in (self.q, self.k, self.v))

    def sectors(self) -> Dict[str, BiasSector]:
        return {"q": self.q, "k": self.k, "v": self.v}

    @torch.no_grad()
    def resample(self) -> None:
        for s in (self.q, self.k, self.v):
            s.resample()

    def apply(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Return ``(q + bQ, k + bK, v + bV)`` (each bias may be zero)."""
        return self.q(q), self.k(k), self.v(v)

    def stats(self) -> Dict[str, float]:
        """Norms/means of each sector -- handy for symmetry diagnostics."""
        out: Dict[str, float] = {}
        for letter, s in self.sectors().items():
            b = s.b.detach()
            out[f"b{letter.upper()}_norm"] = float(b.norm())
            out[f"b{letter.upper()}_absmean"] = float(b.abs().mean())
            out[f"b{letter.upper()}_mode"] = s.mode
        return out

    def extra_repr(self) -> str:
        return (
            f"n_head={self.n_head}, head_dim={self.head_dim}, "
            f"enabled={self.enabled}, mode={self.cfg.attn_mode}"
        )
