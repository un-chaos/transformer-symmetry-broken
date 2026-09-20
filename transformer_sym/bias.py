"""Symmetry-breaking bias modules.

Two independent families, both driven from :mod:`transformer_sym.config`:

``EmbeddingBias``
    The bias ``b`` the user asked for.  Applied once, to the embedding:
    ``x = Embed(tokens) * scale + b``, with ``b`` of shape ``(d_model,)``
    broadcast over the sequence.  ``b != 0`` breaks the ``O(d_model)`` rotation
    symmetry of the embedding, because a rotation ``R`` cannot be pushed through
    the *fixed* offset: ``R(x + b) = Rx + Rb != Rx + b``.

``AttentionBias`` / ``BiasSector``
    Reference-style per-head biases ``bQ``/``bK``/``bV`` of shape
    ``(n_heads, head_dim)`` added to q/k/v.  ``bQ`` enters through the softmax so
    its effect is exponentially amplified; ``bV`` only passes through a linear
    map (power-law effect).

Both are **buffers by default** -- they are part of the experiment, not of the
learned parameters -- and only become ``nn.Parameter`` when ``learnable: true``
is set in the config.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn

from .config import AttentionBiasConfig, BIAS_MODES, EmbeddingBiasConfig

__all__ = ["resolve_vector", "EmbeddingBias", "BiasSector", "AttentionBias"]


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
    """The additive bias ``b`` on the embedding, shape ``(d_model,)``.

    Args:
        d_model: embedding width.
        cfg: :class:`~transformer_sym.config.EmbeddingBiasConfig`.
    """

    def __init__(self, d_model: int, cfg: EmbeddingBiasConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(d_model)
        self.mode = _validate_mode(cfg.mode, "EmbeddingBias")
        self.seed = cfg.seed

        if cfg.learnable and cfg.resample != "fixed":
            raise ValueError("EmbeddingBias: learnable=True requires resample='fixed'")

        init = self._draw(torch.device("cpu"))
        if cfg.learnable:
            self.b = nn.Parameter(init)
        else:
            self.register_buffer("b", init)

    # ------------------------------------------------------------------ #
    @property
    def active(self) -> bool:
        """``True`` when the bias can influence the forward pass at all."""
        return self.mode != "zero" or self.cfg.learnable

    @property
    def is_zero(self) -> bool:
        """``True`` when ``b`` is exactly zero for the current mode (no learning)."""
        return self.mode == "zero" and not self.cfg.learnable

    def _draw(self, device: torch.device) -> torch.Tensor:
        if self.mode == "zero":
            return torch.zeros(self.d_model, device=device)
        if self.mode == "const":
            return torch.full(
                (self.d_model,), float(self.cfg.const_value), device=device
            )
        mean = resolve_vector(self.cfg.mean, self.d_model, "bias.embed.mean").to(device)
        std = resolve_vector(self.cfg.std, self.d_model, "bias.embed.std").to(device)
        return mean + std * self._randn((self.d_model,), device)

    @torch.no_grad()
    def resample(self) -> None:
        if self.cfg.resample == "fixed" or self.cfg.learnable:
            return
        self.b.copy_(self._draw(self.b.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add ``b`` to ``x``; ``b`` broadcasts over batch and time dimensions."""
        return x + self.b

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, mode={self.mode}, "
            f"resample={self.cfg.resample}, learnable={self.cfg.learnable}"
        )


class BiasSector(_BiasBase):
    """One per-head additive bias (e.g. ``bQ``) of shape ``(n_heads, head_dim)``.

    ``share_across_heads=True`` reproduces the reference behaviour: a single
    ``head_dim`` vector is drawn and expanded to every head.
    """

    def __init__(
        self,
        n_heads: int,
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
        self.n_heads = int(n_heads)
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
            return torch.zeros(self.n_heads, self.head_dim, device=device)
        if self.mode == "const":
            return torch.full(
                (self.n_heads, self.head_dim), self.const_value, device=device
            )

        mean = resolve_vector(self.mean, self.head_dim, f"{self.name}.mean").to(device)
        std = resolve_vector(self.std, self.head_dim, f"{self.name}.std").to(device)
        if self.share_across_heads:
            vec = mean + std * self._randn((self.head_dim,), device)
            return vec.unsqueeze(0).expand(self.n_heads, self.head_dim).contiguous()
        return mean.view(1, -1) + std.view(1, -1) * self._randn(
            (self.n_heads, self.head_dim), device
        )

    @torch.no_grad()
    def resample(self) -> None:
        if self.resample_mode == "fixed" or self.learnable:
            return
        self.b.copy_(self._draw(self.b.device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add this bias to a ``(batch, n_heads, time, head_dim)`` tensor."""
        if x.dim() != 4:
            raise ValueError(
                f"BiasSector expects a 4-D (B, H, T, D) tensor, got shape {tuple(x.shape)}"
            )
        if x.shape[1] != self.n_heads or x.shape[-1] != self.head_dim:
            raise ValueError(
                f"BiasSector expects (B, {self.n_heads}, T, {self.head_dim}), "
                f"got {tuple(x.shape)}"
            )
        return x + self.b.view(1, self.n_heads, 1, self.head_dim)

    def extra_repr(self) -> str:
        return (
            f"n_heads={self.n_heads}, head_dim={self.head_dim}, mode={self.mode}, "
            f"share_across_heads={self.share_across_heads}, "
            f"resample={self.resample_mode}"
        )


class AttentionBias(nn.Module):
    """Container holding the ``bQ``/``bK``/``bV`` sectors for one attention site.

    A sector that is disabled is still allocated but pinned to ``zero`` mode, so
    ``apply`` stays branch-free and the state dict has a stable shape.
    """

    def __init__(self, n_heads: int, head_dim: int, cfg: AttentionBiasConfig):
        super().__init__()
        self.cfg = cfg
        self.n_heads = int(n_heads)
        self.head_dim = int(head_dim)
        self.enabled = bool(cfg.enabled)

        # Each sector gets its own RNG stream.  Without the offset, bQ, bK and bV
        # would all be seeded identically and (with equal mean/std) would be
        # three copies of the *same* vector, silently coupling the Q-K and V-O
        # symmetry breaking.  The offsets keep the draws independent and still
        # fully reproducible from ``cfg.seed``.
        seed_offset = {"q": 0, "k": 1, "v": 2}

        def sector(letter: str, enabled: bool) -> BiasSector:
            on = self.enabled and enabled
            mode = cfg.mode if on else "zero"
            return BiasSector(
                n_heads=n_heads,
                head_dim=head_dim,
                mode=mode,
                mean=getattr(cfg, f"{letter}_mean"),
                std=getattr(cfg, f"{letter}_std"),
                const_value=cfg.const_value,
                share_across_heads=cfg.share_across_heads,
                resample=cfg.resample,
                learnable=cfg.learnable and on,
                seed=int(cfg.seed) + seed_offset[letter],
                name=f"b{letter.upper()}",
            )

        self.q = sector("q", cfg.q_enabled)
        self.k = sector("k", cfg.k_enabled)
        self.v = sector("v", cfg.v_enabled)

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
            f"n_heads={self.n_heads}, head_dim={self.head_dim}, "
            f"enabled={self.enabled}, mode={self.cfg.mode}"
        )
