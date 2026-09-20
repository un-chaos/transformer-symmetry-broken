"""EGD -- energy-conserving descent, plus the AdamW baseline and a builder.

``EGD`` is a faithful port of ``ECD_q1_scaled`` from the reference project
(``Symmetry-breaking-attention-bias-main/ecd_symbreak/optimizer.py``), with the
two deliberate fixes required by ``INTERFACES.md`` section 2:

1. **Dimension double-counting.**  The reference accumulates
   ``self.dim += q.numel()`` inside ``if self.iteration == 0:`` while
   ``self.iteration`` is only incremented *after* a successful update.  A first
   step that is skipped (because ``loss - F0 <= eps2``) therefore re-enters the
   initialisation branch on the next step and counts every parameter twice --
   and also rescales ``lr``/``nu`` twice.  Here a separate ``self._initialized``
   flag guarantees that initialisation happens exactly once.
2. **A missing closure.**  The reference silently does nothing when
   ``closure is None`` (it needs the loss *and* ``param.grad``); here that is a
   clear ``ValueError``.

Everything else follows the reference exactly.  The idea: treat the parameters
``q`` as positions and the negative gradients as momenta ``p`` in a Hamiltonian
system whose "energy" denominator is ``F(q) - F0``.  The Liouville measure of the
resulting dynamics concentrates samples towards low loss,
``p(Theta) ~ (F(Theta) - F0) ** (-eta * d / 2)``, so ``F0`` must be kept strictly
below the smallest reachable loss, otherwise every update is skipped.  The
momenta are renormalised every step (energy conservation) and optionally kicked
by Gaussian noise of amplitude ``nu``.

Notes:
    * ``step`` is decorated with ``@torch.no_grad()`` and calls the closure
      inside ``with torch.enable_grad():``, so the closure is responsible for
      ``zero_grad`` / ``backward`` / returning the loss.
    * Parameters whose ``.grad is None`` are skipped everywhere instead of
      crashing (the reference indexes ``q.grad.data`` before checking); they
      start from zero momenta and a gradient that only appears later is handled
      without a ``KeyError``.
    * The ``nu`` noise is drawn from a dedicated ``torch.Generator``, so a run is
      reproducible from ``seed`` and the global RNG stream is left untouched.
"""

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional

import torch
from torch.optim.optimizer import Optimizer

if TYPE_CHECKING:  # pragma: no cover - imported for type hints only
    from .config import TrainConfig

__all__ = ["EGD", "optimizer_requires_closure", "build_optimizer"]

#: Scalar bookkeeping carried in ``state_dict()["egd"]``.  These are exactly the
#: quantities the reference keeps on the optimizer instance and *not* in the
#: per-parameter ``state``, so without them a resumed run would silently restart
#: the dynamics (dimension, momentum normalisation, energy offset, RNG stream).
_EGD_STATE_KEYS: tuple[str, ...] = (
    "eta",
    "auto_F0_margin",
    "eps1",
    "eps2",
    "weight_decay",
    "consEn",
    "seed",
    "lr",                       # already rescaled by 1/sqrt(eta)
    "nu",                       # already rescaled by 1/sqrt(dim)
    "F0",
    "F0_resolved",
    "Finit",
    "dim",
    "p2",
    "q2",
    "normalization_coefficient",
    "iteration",
    "skipped_updates",
    "last_loss",
    "_initialized",
)


class EGD(Optimizer):
    """Energy-conserving descent (reference name: ``ECD_q1_scaled``).

    Args:
        params: iterable of ``torch.nn.Parameter`` (or param groups) to optimise.
        lr: learning rate; internally rescaled to ``lr / sqrt(eta)``.
        eta: concentration parameter controlling how sharply the dynamics
            concentrates around low loss.
        F0: loss offset; must stay *below* the smallest reachable loss or every
            update is skipped.  ``F0=None`` resolves it at the first step to
            ``Finit - auto_F0_margin``.
        auto_F0_margin: how far below the initial loss an automatic ``F0`` sits.
        nu: amplitude of the Gaussian "bounce" added to the momenta each step
            (internally rescaled by ``1 / sqrt(dim)``); ``0`` disables the noise.
        eps1: numerical epsilon for the energy-conservation normalisation.
        eps2: numerical epsilon for the ``loss - F0 > eps2`` update test.
        weight_decay: L2 coefficient; enters as ``g = grad + weight_decay * q``
            and adds ``0.5 * weight_decay * |q|^2`` to the energy denominator.
        consEn: enable the energy-conservation rescaling of the momenta.
        seed: seed of the dedicated noise generator.  ``None`` draws a random
            one (the global RNG is never used either way).

    Attributes:
        dim: total number of optimised parameters, filled in once.
        Finit: loss value seen at initialisation.
        F0: resolved loss offset (a float once ``step`` has run).
        F0_resolved: same value, kept for introspection.
        iteration: number of *successful* updates.
        skipped_updates: number of steps where ``loss - F0 <= eps2``.
        restored_egd_state: whether the last :meth:`load_state_dict` found (and
            applied) the ``"egd"`` bookkeeping entry.

    Resuming:
        ``state_dict()`` carries one extra top-level entry, ``"egd"``, holding all
        of the above plus the dedicated noise generator's state, while leaving the
        standard ``state`` (including every parameter's ``momenta``) and
        ``param_groups`` structure untouched.  ``load_state_dict()`` restores it,
        so ``save -> fresh optimizer -> load`` continues bit-for-bit where the run
        stopped: ``lr``/``nu`` come back already rescaled, ``_initialized`` stays
        ``True``, and the RNG resumes the same ``nu`` noise sequence.  A
        checkpoint without ``"egd"`` (an older EGD run, or an AdamW dict) still
        loads; a ``RuntimeWarning`` is raised and the EGD scalars are
        re-initialised at the next step instead.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor],
        lr: float = 1.0,
        eta: float = 100.0,
        F0: Optional[float] = 1.0,
        auto_F0_margin: float = 1.0,
        nu: float = 0.0,
        eps1: float = 1e-10,
        eps2: float = 1e-40,
        weight_decay: float = 0.0,
        consEn: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        if lr <= 0:
            raise ValueError(f"lr must be > 0, got {lr!r}")
        if eta <= 0:
            raise ValueError(f"eta must be > 0, got {eta!r}")
        if nu < 0:
            raise ValueError(f"nu must be >= 0, got {nu!r}")
        if eps1 <= 0:
            raise ValueError(f"eps1 must be > 0, got {eps1!r}")
        if eps2 < 0:
            raise ValueError(f"eps2 must be >= 0, got {eps2!r}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {weight_decay!r}")

        defaults: Dict[str, Any] = dict(
            lr=lr, eta=eta, F0=F0, auto_F0_margin=auto_F0_margin, nu=nu,
            eps1=eps1, eps2=eps2, weight_decay=weight_decay, consEn=consEn,
            seed=seed,
        )
        super().__init__(params, defaults)

        self.lr = float(lr)
        self.eta = float(eta)
        self.F0: Optional[float] = None if F0 is None else float(F0)
        self.F0_resolved: Optional[float] = None
        self.auto_F0_margin = float(auto_F0_margin)
        self.nu = float(nu)
        self.eps1 = float(eps1)
        self.eps2 = float(eps2)
        self.weight_decay = float(weight_decay)
        self.consEn = bool(consEn)
        self.seed = seed

        self.iteration = 0
        self.skipped_updates = 0
        self.Finit = 0.0
        self.dim = 0
        self.p2 = 1.0          # |p|^2, carried between steps
        self.q2 = 0.0          # |q|^2 used by the L2 term
        self.normalization_coefficient = 1.0
        self.last_loss: Optional[float] = None

        self._initialized = False
        self._restored_egd_state = False
        self.generator: Optional[torch.Generator] = None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _first_param(self) -> torch.Tensor:
        """Return the first parameter of the first group (for device/dtype)."""
        for group in self.param_groups:
            for q in group["params"]:
                return q
        raise ValueError("EGD was given no parameters")

    def _initialize(self, loss: float) -> None:
        """One-time initialisation: ``Finit``, ``F0``, ``dim``, momenta, rescalings."""
        device = self._first_param().device
        self.generator = torch.Generator(device=device.type)
        if self.seed is None:
            # Draw a random seed from the generator itself; the global RNG is
            # never touched (the reference does the same).
            self.generator.manual_seed(self.generator.seed())
        else:
            self.generator.manual_seed(int(self.seed))

        self.Finit = loss
        if not self.consEn:
            self.normalization_coefficient = 1.0
        if self.F0 is None:
            self.F0 = loss - self.auto_F0_margin
        self.F0_resolved = float(self.F0)

        # Total number of parameters.  Counted here, once, under the
        # ``self._initialized`` guard -- never again (fix #1).
        self.dim = sum(
            q.numel() for group in self.param_groups for q in group["params"]
        )

        # Momenta start along minus the gradient.  A parameter that produced no
        # gradient starts at zero momenta: every update loop skips it, and the
        # entry exists so that a gradient appearing later cannot key-error.
        all_params = []
        p2init = 0.0
        for group in self.param_groups:
            for q in group["params"]:
                if q.grad is None:
                    p = torch.zeros_like(q.detach())
                else:
                    p = -q.grad.detach().clone()
                    p2init += float(p.pow(2).sum())
                self.state[q]["momenta"] = p
                all_params.append(q)

        # ... and are normalised so that |p(0)| = 1 over all parameters.
        if p2init > 0.0:
            scale = 1.0 / math.sqrt(p2init)
            for q in all_params:
                self.state[q]["momenta"].mul_(scale)
            self.p2 = 1.0
        else:
            self.p2 = 0.0

        # Rescale the hyperparameters (exactly once).
        self.lr = self.lr / math.sqrt(self.eta)
        self.nu = self.nu / math.sqrt(self.dim)
        self._sync_param_groups()

        self._initialized = True

    def _momenta(self, q: torch.Tensor) -> torch.Tensor:
        """Momentum buffer of ``q``; created lazily if it is missing.

        Initialisation fills this in for every parameter, so the lazy path is
        only reached by a parameter group added after the first step.
        """
        state = self.state[q]
        if "momenta" not in state:
            state["momenta"] = torch.zeros_like(q.detach())
        return state["momenta"]

    def _sync_param_groups(self) -> None:
        """Mirror the *effective* hyper-parameters into ``param_groups``.

        ``Optimizer.__init__`` seeds every group with the values passed to the
        constructor, but ``lr`` and ``nu`` are rescaled once at initialisation and
        ``F0`` may be resolved from ``None``.  Writing the effective values back
        keeps ``param_groups`` (which is what ``state_dict()`` stores) consistent
        with what the dynamics actually uses, instead of leaving a stale raw
        ``lr`` next to the rescaled ``self.lr``.
        """
        for group in self.param_groups:
            group["lr"] = self.lr
            group["nu"] = self.nu
            group["F0"] = self.F0

    def _update(self, denom: float) -> None:
        """One successful EGD update: momentum kick, drift, momentum rotation."""
        # --- energy-conservation rescaling of the momenta (reference consEn) ---
        if self.consEn:
            p2true = 1.0
            if abs(p2true - self.p2) < self.eps1:
                self.normalization_coefficient = 1.0
            elif p2true < 0.0:  # kept for parity with the reference (dead branch)
                self.normalization_coefficient = 1.0
            elif self.p2 == 0.0:
                self.normalization_coefficient = 1.0
            else:
                self.normalization_coefficient = math.sqrt(p2true / self.p2)

        # --- momentum kick: gradient orthogonalised against p ----------------
        # ``prefactor = -0.5 * lr * eta * dim / (dim - 1)``; a single scalar
        # parameter would divide by zero, so that degenerate case is guarded.
        if self.dim > 1:
            prefactor = -0.5 * self.lr * self.eta * self.dim / (self.dim - 1)
        else:
            prefactor = -0.5 * self.lr * self.eta * self.dim
        scale = prefactor / denom

        p2 = 0.0
        for group in self.param_groups:
            for q in group["params"]:
                if q.grad is None:
                    continue
                p = self._momenta(q)
                g = q.grad.detach() + self.weight_decay * q.detach()
                p.mul_(self.normalization_coefficient)
                dotp = torch.dot(p.reshape(-1), g.reshape(-1))
                p.add_(scale * (g - p * dotp))
                p2 += float(p.pow(2).sum())

        # --- drift (position update) + random rotation of the momenta --------
        pnorm = math.sqrt(p2)
        p2new = 0.0
        for group in self.param_groups:
            for q in group["params"]:
                if q.grad is None:
                    continue
                state = self.state[q]
                p = self._momenta(q)
                q.add_(self.lr * p)
                z = torch.randn(
                    p.shape, device=p.device, dtype=p.dtype,
                    generator=self.generator,
                )
                p = (p / pnorm if pnorm > 0.0 else torch.zeros_like(p)) + self.nu * z
                state["momenta"] = p
                p2new += float(p.pow(2).sum())

        # --- renormalise the new direction -----------------------------------
        if pnorm > 0.0 and p2new > 0.0:
            scale_back = pnorm / math.sqrt(p2new)
            for group in self.param_groups:
                for q in group["params"]:
                    state = self.state[q]
                    if "momenta" in state:
                        state["momenta"] = state["momenta"] * scale_back
            self.p2 = pnorm * pnorm
        else:
            self.p2 = 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def step(
        self, closure: Optional[Callable[[], torch.Tensor]] = None
    ) -> Optional[torch.Tensor]:
        """Perform one EGD step and return the loss the closure produced.

        Args:
            closure: required callable that recomputes the loss and its
                gradients (``zero_grad`` -> ``backward`` -> ``return loss``).
                EGD needs both the loss value (for ``F(q) - F0``) and
                ``param.grad`` (to build the momenta).

        Returns:
            The loss tensor returned by ``closure`` (unchanged, also when the
            update was skipped).

        Raises:
            ValueError: if ``closure`` is ``None``, returns ``None``, or returns
                something that is not a single-element tensor.
        """
        if closure is None:
            raise ValueError(
                "EGD.step() requires a closure that recomputes the loss and its "
                "gradients: the energy-conserving dynamics needs both the loss "
                "value (for the F(q) - F0 denominator) and param.grad (to update "
                "the momenta). Pass something like\n"
                "    def closure():\n"
                "        optimizer.zero_grad()\n"
                "        loss = model(...)\n"
                "        loss.backward()\n"
                "        return loss\n"
                "as optimizer.step(closure)."
            )

        with torch.enable_grad():
            loss = closure()

        if loss is None:
            raise ValueError(
                "the closure passed to EGD.step() returned None; it must return "
                "the recomputed scalar loss tensor (after calling backward())."
            )
        if not torch.is_tensor(loss):
            loss = torch.as_tensor(loss)
        if loss.numel() != 1:
            raise ValueError(
                f"EGD needs a single scalar loss, got a tensor with shape "
                f"{tuple(loss.shape)}"
            )

        loss_value = float(loss.detach())
        self.last_loss = loss_value

        if not self._initialized:
            self._initialize(loss_value)

        # |q|^2 for the L2 term (computed once per step, like the reference).
        q2 = 0.0
        if self.weight_decay != 0.0:
            for group in self.param_groups:
                for q in group["params"]:
                    q2 += float(q.detach().pow(2).sum())
        self.q2 = q2

        denom = loss_value + 0.5 * self.weight_decay * q2 - self.F0
        if denom > self.eps2:
            self._update(denom)
            self.iteration += 1
        else:
            # ``loss <= F0``: the energy denominator is not positive, so the
            # dynamics is frozen for this step.
            self.skipped_updates += 1

        return loss

    def stats(self) -> Dict[str, float]:
        """Small float-only snapshot of the optimizer state.

        Returns:
            ``{"iteration", "F0", "loss", "momentum_norm", "skipped_updates"}``;
            entries that are not known yet (before the first step) are ``nan``.
        """
        nan = float("nan")
        return {
            "iteration": float(self.iteration),
            "F0": nan if self.F0 is None else float(self.F0),
            "loss": nan if self.last_loss is None else float(self.last_loss),
            "momentum_norm": math.sqrt(self.p2) if self.p2 > 0.0 else 0.0,
            "skipped_updates": float(self.skipped_updates),
        }

    # ------------------------------------------------------------------ #
    # Save / restore (resume support)
    # ------------------------------------------------------------------ #
    def state_dict(self) -> Dict[str, Any]:
        """Return the optimizer state, plus an ``"egd"`` bookkeeping entry.

        The standard ``torch.optim`` structure (``state``, including every
        parameter's ``momenta``, and ``param_groups``) is returned untouched.
        ``"egd"`` adds the scalars the reference keeps on the instance and the
        state of the dedicated noise generator, so that a run can be resumed
        exactly where it stopped.  Every value is a plain float/bool/``None`` or a
        ``torch.ByteTensor``, so the dict stays ``torch.save``/``torch.load``
        compatible (also with the default ``weights_only=True``).

        Returns:
            ``{"state": ..., "param_groups": ..., "egd": {...}}``.
        """
        state = super().state_dict()
        extra: Dict[str, Any] = {key: getattr(self, key) for key in _EGD_STATE_KEYS}
        extra["generator_state"] = (
            None if self.generator is None else self.generator.get_state()
        )
        state["egd"] = extra
        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Restore everything :meth:`state_dict` saved.

        A checkpoint carrying the ``"egd"`` entry restores the full dynamics:
        ``iteration``/``skipped_updates``/``F0``/``F0_resolved``/``dim``/``Finit``/
        ``p2`` are copied over, ``lr`` and ``nu`` are taken as the *already
        rescaled* values of the saved run, ``_initialized`` becomes ``True`` (so
        the initialisation block -- and therefore the rescaling -- cannot run a
        second time), and the noise generator resumes the saved RNG stream.

        A checkpoint without ``"egd"`` (an older EGD run or an AdamW-style dict)
        is still accepted: the base class restores the parameters and momenta, a
        ``RuntimeWarning`` is emitted, and the EGD scalars keep their
        freshly-initialised values, so the next :meth:`step` initialises the
        dynamics again.

        Args:
            state_dict: a dict previously returned by :meth:`state_dict` (the
                ``"egd"`` entry is optional).

        Returns:
            Whatever ``torch.optim.Optimizer.load_state_dict`` returns (``None``).
        """
        payload = dict(state_dict)
        extra = payload.pop("egd", None)

        result = super().load_state_dict(payload)
        self._restored_egd_state = extra is not None

        if extra is None:
            warnings.warn(
                "EGD.load_state_dict(): the checkpoint has no 'egd' entry, so only "
                "the parameters and their momenta were restored. EGD's scalar "
                "bookkeeping (iteration, F0, dim, the rescaled lr/nu and the noise "
                "RNG state) is re-initialised at the next step, so the resumed run "
                "will not match the uninterrupted one.",
                RuntimeWarning,
                stacklevel=2,
            )
            return result

        for key in _EGD_STATE_KEYS:
            if key in extra:
                setattr(self, key, extra[key])

        generator_state = extra.get("generator_state")
        if generator_state is not None:
            if self.generator is None:
                self.generator = torch.Generator(
                    device=self._first_param().device.type
                )
            if isinstance(generator_state, torch.Tensor):
                generator_state = generator_state.to(self.generator.device)
            self.generator.set_state(generator_state)

        # ``lr``/``nu``/``F0`` are already the effective values: mirror them into
        # the param groups that the base class just restored.
        self._sync_param_groups()
        return result

    @property
    def restored_egd_state(self) -> bool:
        """Whether the last :meth:`load_state_dict` applied the ``"egd"`` entry.

        ``False`` means the checkpoint carried no EGD bookkeeping (an older run or
        an AdamW-style dict), in which case the dynamics is re-initialised at the
        next :meth:`step`.
        """
        return self._restored_egd_state


def optimizer_requires_closure(name: str) -> bool:
    """Whether the optimizer called ``name`` must be stepped with a closure.

    Args:
        name: optimizer name, e.g. ``"egd"`` or ``"adamw"``.

    Returns:
        ``True`` for ``"egd"`` (its dynamics needs the loss *and* the
        gradients), ``False`` for everything else.
    """
    return str(name).strip().lower() == "egd"


def build_optimizer(model: torch.nn.Module, cfg: "TrainConfig") -> Optimizer:
    """Build the optimizer selected by ``cfg.optimizer``.

    Only parameters with ``requires_grad=True`` are handed to the optimizer, so
    frozen parts of the model are never touched.

    Args:
        model: the model whose parameters are optimised.
        cfg: a ``TrainConfig`` (``cfg.optimizer``, ``cfg.egd``, ``cfg.adamw``).

    Returns:
        An :class:`EGD` for ``"egd"`` or a ``torch.optim.AdamW`` for ``"adamw"``.

    Raises:
        ValueError: for an unknown ``cfg.optimizer`` or a model without
            trainable parameters.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError(
            "build_optimizer: the model has no parameters with requires_grad=True"
        )

    name = str(cfg.optimizer).strip().lower()
    if name == "egd":
        e = cfg.egd
        return EGD(
            params,
            lr=e.lr,
            eta=e.eta,
            F0=e.F0,
            auto_F0_margin=e.auto_F0_margin,
            nu=e.nu,
            eps1=e.eps1,
            eps2=e.eps2,
            weight_decay=e.weight_decay,
            consEn=e.consEn,
            seed=e.seed,
        )
    if name == "adamw":
        a = cfg.adamw
        return torch.optim.AdamW(
            params,
            lr=a.lr,
            betas=tuple(a.betas),
            eps=a.eps,
            weight_decay=a.weight_decay,
        )
    raise ValueError(
        f"unknown optimizer {cfg.optimizer!r}; expected 'egd' or 'adamw'"
    )
