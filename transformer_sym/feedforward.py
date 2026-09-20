"""Position-wise feed-forward network (the MLP block).

Two activations are available, matching the reference project:

``gelu`` / ``relu``
    Symmetric activations -- the baseline.

``prelu``
    A PReLU.  With ``prelu_random_init=True`` the negative slopes are drawn
    per feature from ``N(prelu_slope_mean, prelu_slope_std^2)``
    (:class:`RandomPReLU1d`), which is itself a symmetry break: a plain
    ``nn.PReLU`` starts every slope at the same value, whereas the reference's
    ``AsymmetricMLPPreLU`` gives every hidden feature its own initial slope.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ACTIVATIONS

__all__ = ["RandomPReLU1d", "FeedForward"]


class RandomPReLU1d(nn.Module):
    """PReLU with a learnable per-feature negative slope, randomly initialised.

    Mirrors ``AsymmetricMLPPreLU``/``RandomPReLU1d`` from the reference project:
    the slopes are a single ``(features,)`` parameter and ``F.prelu`` is applied
    to the transposed ``(B, features, T)`` view so that every *feature* (not
    every time step) owns one slope.

    Args:
        features: number of channels the slopes are attached to.
        init_slope: mean of the initial slope distribution (reference: ``0.2``).
        slope_std: standard deviation of the initial slope distribution
            (reference: ``1.0``).
    """

    def __init__(
        self, features: int, init_slope: float = 0.2, slope_std: float = 1.0
    ) -> None:
        super().__init__()
        if features < 1:
            raise ValueError(f"features must be >= 1, got {features}")
        self.features = int(features)
        self.init_slope = float(init_slope)
        self.slope_std = float(slope_std)
        self.weight = nn.Parameter(
            torch.randn(self.features) * self.slope_std + self.init_slope
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the per-feature PReLU to ``(B, T, C)`` (or ``(B, C, T)`` if C==features)."""
        if x.dim() != 3:
            raise ValueError(
                f"RandomPReLU1d expects a 3-D tensor, got shape {tuple(x.shape)}"
            )
        if x.shape[-1] == self.features:
            # (B, T, C) -> (B, C, T) so ``weight`` lines up with the features.
            return F.prelu(x.transpose(1, 2), self.weight).transpose(1, 2)
        if x.shape[1] == self.features:
            return F.prelu(x, self.weight)
        raise ValueError(
            f"RandomPReLU1d(features={self.features}) cannot be applied to shape "
            f"{tuple(x.shape)}"
        )

    def extra_repr(self) -> str:
        return (
            f"features={self.features}, init_slope={self.init_slope}, "
            f"slope_std={self.slope_std}"
        )


class FeedForward(nn.Module):
    """``c_proj(act(c_fc(x)))`` applied independently at every position.

    Args:
        d_model: model width.
        d_ff: hidden width.
        dropout: dropout applied to the hidden activations.
        activation: one of ``"gelu"``, ``"relu"``, ``"prelu"``.
        prelu_random_init: only for ``activation="prelu"`` -- use
            :class:`RandomPReLU1d` instead of ``nn.PReLU(d_ff)``.
        prelu_slope_mean: mean of the random initial slopes.
        prelu_slope_std: std of the random initial slopes.

    Attributes:
        c_fc: ``d_model -> d_ff``.
        c_proj: ``d_ff -> d_model`` (marked as a residual output projection, so
            :meth:`Seq2SeqTransformer._init_weights` scales its init).
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
        prelu_random_init: bool = False,
        prelu_slope_mean: float = 0.2,
        prelu_slope_std: float = 1.0,
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(
                f"activation must be one of {ACTIVATIONS}, got {activation!r}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.activation = activation
        self.prelu_random_init = bool(prelu_random_init)

        self.c_fc = nn.Linear(self.d_model, self.d_ff)
        if activation == "gelu":
            self.act: nn.Module = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif prelu_random_init:
            self.act = RandomPReLU1d(
                self.d_ff, init_slope=prelu_slope_mean, slope_std=prelu_slope_std
            )
        else:
            self.act = nn.PReLU(self.d_ff)
        self.c_proj = nn.Linear(self.d_ff, self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: ``x`` ``(B, T, d_model)``.  Returns: ``(B, T, d_model)``."""
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_ff={self.d_ff}, "
            f"activation={self.activation}, prelu_random_init={self.prelu_random_init}"
        )
