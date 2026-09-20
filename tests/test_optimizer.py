"""Tests for ``transformer_sym.optimizer``: ``EGD``, the helpers and the builder.

Everything here is deliberately tiny -- at most a few dozen steps on CPU with a
handful of scalars -- so the whole file runs in a couple of seconds.  The tests
cover the ported reference behaviour, the two deliberate fixes described in
``INTERFACES.md`` section 2, and the save/restore round trip that makes
``train.py --resume`` faithful.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
import torch

from transformer_sym.config import TrainConfig
from transformer_sym.optimizer import EGD, build_optimizer, optimizer_requires_closure

# The quadratic 0.5 * ||w - target||^2 at the default starting point below.
QUADRATIC_INITIAL_LOSS = 0.5 * (4.0 + 9.0 + 1.0 + 0.25)  # == 7.125


# --------------------------------------------------------------------------- #
# tmp_path, made robust against this machine's pytest basetemp
# --------------------------------------------------------------------------- #
# Copied from ``tests/test_data.py`` (that file belongs to another module, so the
# fixture is duplicated rather than imported).  On this machine
# ``mkdir(mode=0o700)`` -- how pytest creates ``%TEMP%/pytest-of-<user>`` --
# yields a directory the process can no longer open, so the built-in ``tmp_path``
# fixture dies during setup.  Try the standard factory first (stock behaviour on
# a normal machine) and fall back to an ordinary directory under the system temp
# dir otherwise.
def _safe_name(request: pytest.FixtureRequest) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", request.node.name)[:60] or "test"


@pytest.fixture
def tmp_path(tmp_path_factory, request):
    """Per-test scratch directory, usable even when pytest's basetemp is not."""
    try:
        path = tmp_path_factory.mktemp(_safe_name(request))
    except OSError:
        path = None

    if path is not None:
        yield path
        return

    path = Path(tempfile.gettempdir()) / f"pytest-opt-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)  # default mode: 0o700 is unusable here
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _quadratic_problem(
    lr: float = 1.0,
    eta: float = 100.0,
    F0: float | None = -1.0,
    auto_F0_margin: float = 1.0,
    nu: float = 0.0,
    seed: int | None = 0,
):
    """``f(w) = 0.5 * ||w - 0||^2`` with ``w0 = (2, -3, 1, 0.5)`` -> ``f(w0) = 7.125``.

    Returns ``(w, optimizer, closure)``; the closure follows the required
    ``zero_grad -> backward -> return loss`` shape.
    """
    w = torch.nn.Parameter(torch.tensor([2.0, -3.0, 1.0, 0.5]))
    target = torch.zeros_like(w)
    opt = EGD(
        [w], lr=lr, eta=eta, F0=F0, auto_F0_margin=auto_F0_margin, nu=nu, seed=seed
    )

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = 0.5 * ((w - target) ** 2).sum()
        loss.backward()
        return loss

    return w, opt, closure


# --------------------------------------------------------------------------- #
# core behaviour
# --------------------------------------------------------------------------- #
def test_egd_descends_a_convex_quadratic():
    """EGD must drive the quadratic down substantially."""
    w, opt, closure = _quadratic_problem()
    history: list[float] = []
    for _ in range(30):
        loss = opt.step(closure)
        history.append(float(loss.detach()))

    print("quadratic loss trajectory:", [round(v, 6) for v in history])

    assert history[0] == pytest.approx(QUADRATIC_INITIAL_LOSS)
    assert min(history) < 1e-3                      # gets essentially to the minimum
    assert history[-1] < 0.05 * history[0]          # and stays down
    assert float(w.detach().norm()) < 0.5           # w moved from |w0| ~ 3.77
    assert opt.iteration == 30
    assert opt.skipped_updates == 0
    assert math.isfinite(history[-1])


def test_egd_trains_a_tiny_mlp():
    """A 2-layer MLP on a small synthetic regression task must improve."""
    torch.manual_seed(0)
    x = torch.randn(16, 4)
    y = (x.sum(dim=1, keepdim=True) > 0).float() * 2.0 - 1.0     # +-1 targets

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1)
    )
    opt = EGD(model.parameters(), lr=1.0, eta=100.0, F0=-1.0, seed=0)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = ((model(x) - y) ** 2).mean()
        loss.backward()
        return loss

    losses = [float(opt.step(closure).detach()) for _ in range(40)]

    assert all(math.isfinite(v) for v in losses)
    assert losses[-1] < losses[0]
    assert losses[-1] < 0.25 * losses[0]            # observed: ~1.4% of the start
    assert opt.iteration == 40


def test_F0_is_a_floor_and_freezes_the_parameters():
    """Once ``loss <= F0`` the update is skipped and ``skipped_updates`` grows."""
    w, opt, closure = _quadratic_problem(F0=-1.0)
    for _ in range(25):
        opt.step(closure)
    assert opt.skipped_updates == 0
    assert opt.iteration == 25

    # Put the floor well above the current loss: the dynamics is frozen.
    opt.F0 = 1.0
    frozen = w.detach().clone()
    for _ in range(4):
        loss = opt.step(closure)
        assert float(loss.detach()) <= opt.F0
    assert torch.equal(frozen, w.detach())          # parameters stop changing
    assert opt.skipped_updates == 4
    assert opt.iteration == 25                      # no successful update happened

    # Lowering the floor again resumes the descent.
    opt.F0 = -1.0
    opt.step(closure)
    assert not torch.equal(frozen, w.detach())
    assert opt.iteration == 26
    assert opt.skipped_updates == 4


def test_F0_none_resolves_via_auto_margin():
    """``F0=None`` becomes ``Finit - auto_F0_margin`` at the first step."""
    w, opt, closure = _quadratic_problem(F0=None, auto_F0_margin=0.25)
    assert opt.F0 is None                           # unresolved before stepping

    initial = float(closure().detach())
    assert initial == pytest.approx(QUADRATIC_INITIAL_LOSS)

    opt.step(closure)
    assert opt.Finit == pytest.approx(initial)
    assert opt.F0 == pytest.approx(initial - 0.25)
    assert opt.F0_resolved == pytest.approx(initial - 0.25)
    assert opt.stats()["F0"] == pytest.approx(initial - 0.25)


# --------------------------------------------------------------------------- #
# the two deliberate fixes
# --------------------------------------------------------------------------- #
def test_dimension_is_counted_once_when_the_first_step_is_skipped():
    """Regression test for the reference's dimension double-counting bug.

    In the reference ``self.dim += q.numel()`` lives inside
    ``if self.iteration == 0:`` while ``self.iteration`` only grows after a
    *successful* update.  A skipped first step therefore re-runs the whole
    initialisation branch on the next call: ``dim`` doubles and the ``lr``/``nu``
    rescalings are applied twice.  Here ``F0`` is put above the initial loss so
    that the very first step is skipped, and then lowered.
    """
    w, opt, closure = _quadratic_problem(F0=10.0, nu=0.4)
    total = w.numel()
    assert total == 4

    # First call: 7.125 - 10.0 <= eps2 -> initialise, but do not update.
    opt.step(closure)
    assert opt.iteration == 0
    assert opt.skipped_updates == 1
    assert opt.dim == total
    assert opt.lr == pytest.approx(1.0 / math.sqrt(100.0))
    assert opt.nu == pytest.approx(0.4 / math.sqrt(total))
    # The momenta were initialised, but the parameters did not move.
    assert opt.stats()["momentum_norm"] == pytest.approx(1.0)

    # Second call with a floor below the loss must not re-initialise.
    opt.F0 = -1.0
    opt.step(closure)
    assert opt.dim == total                                     # reference: 2 * total
    assert opt.lr == pytest.approx(1.0 / math.sqrt(100.0))      # reference: 1 / eta
    assert opt.nu == pytest.approx(0.4 / math.sqrt(total))      # reference: / dim twice
    assert opt.iteration == 1
    assert opt.skipped_updates == 1


def test_step_without_closure_raises_a_clear_value_error():
    """EGD needs a closure: the dynamics uses both the loss and ``param.grad``."""
    w, opt, _ = _quadratic_problem()
    with pytest.raises(ValueError, match="closure"):
        opt.step()
    with pytest.raises(ValueError, match="closure"):
        opt.step(None)


# --------------------------------------------------------------------------- #
# helpers / integration layer
# --------------------------------------------------------------------------- #
def test_optimizer_requires_closure():
    assert optimizer_requires_closure("egd") is True
    assert optimizer_requires_closure("EGD") is True
    assert optimizer_requires_closure("adamw") is False
    assert optimizer_requires_closure("sgd") is False


def test_build_optimizer_selects_the_optimizer_and_drops_frozen_params():
    model = torch.nn.Sequential(
        torch.nn.Linear(3, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1)
    )
    for p in model[0].parameters():                 # freeze the first layer
        p.requires_grad_(False)
    expected = {id(p) for p in model.parameters() if p.requires_grad}

    cfg = TrainConfig()
    assert cfg.optimizer == "egd"

    egd = build_optimizer(model, cfg)
    assert isinstance(egd, EGD)
    assert {id(p) for p in egd.param_groups[0]["params"]} == expected
    assert all(p.requires_grad for p in egd.param_groups[0]["params"])
    assert egd.eta == cfg.egd.eta
    assert egd.F0 == cfg.egd.F0
    assert egd.nu == cfg.egd.nu
    assert egd.lr == cfg.egd.lr                      # not yet rescaled
    assert egd.dim == 0                              # initialised at the first step

    cfg.optimizer = "adamw"
    adam = build_optimizer(model, cfg)
    assert isinstance(adam, torch.optim.AdamW)
    assert {id(p) for p in adam.param_groups[0]["params"]} == expected
    assert adam.defaults["lr"] == pytest.approx(cfg.adamw.lr)
    assert tuple(adam.defaults["betas"]) == tuple(cfg.adamw.betas)
    assert adam.defaults["eps"] == pytest.approx(cfg.adamw.eps)
    assert adam.defaults["weight_decay"] == pytest.approx(cfg.adamw.weight_decay)

    cfg.optimizer = "sgd"
    with pytest.raises(ValueError, match="optimizer"):
        build_optimizer(model, cfg)


def test_parameters_without_grad_are_skipped_safely():
    """A parameter that never receives a gradient must not crash the step."""
    used = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    unused = torch.nn.Parameter(torch.tensor([3.0]))
    opt = EGD([used, unused], lr=1.0, eta=100.0, F0=-1.0, seed=0)

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = 0.5 * (used ** 2).sum()               # ``unused`` never enters
        loss.backward()
        return loss

    for _ in range(5):
        opt.step(closure)

    assert unused.grad is None
    assert torch.equal(unused.detach(), torch.tensor([3.0]))
    assert not torch.equal(used.detach(), torch.tensor([1.0, 2.0]))
    assert opt.dim == 3                              # all parameters are counted
    assert opt.iteration == 5

    # A gradient that only appears after initialisation must be tolerated too
    # (the reference would key-error on the missing momenta).
    def closure_with_unused() -> torch.Tensor:
        opt.zero_grad()
        loss = 0.5 * (used ** 2).sum() + 0.5 * (unused ** 2).sum()
        loss.backward()
        return loss

    stale = unused.detach().clone()
    opt.step(closure_with_unused)
    assert opt.iteration == 6
    assert not torch.equal(stale, unused.detach())


def test_noise_is_seeded_and_leaves_the_global_rng_untouched():
    """``nu`` draws from a dedicated generator: reproducible, no global RNG use."""
    torch.manual_seed(1234)
    expected = torch.randn(3)

    torch.manual_seed(1234)
    w_a, opt_a, closure_a = _quadratic_problem(nu=0.5, seed=99)
    for _ in range(6):
        opt_a.step(closure_a)
    observed = torch.randn(3)
    assert torch.equal(expected, observed)           # the global stream is intact

    w_b, opt_b, closure_b = _quadratic_problem(nu=0.5, seed=99)
    for _ in range(6):
        opt_b.step(closure_b)
    assert torch.equal(w_a.detach(), w_b.detach())   # same seed -> same trajectory

    # A different seed gives a different (still finite) trajectory.
    w_c, opt_c, closure_c = _quadratic_problem(nu=0.5, seed=100)
    for _ in range(6):
        opt_c.step(closure_c)
    assert not torch.equal(w_a.detach(), w_c.detach())


def test_stats_returns_a_small_float_dict():
    w, opt, closure = _quadratic_problem(F0=-1.0)

    before = opt.stats()
    assert before["iteration"] == 0.0 and before["skipped_updates"] == 0.0
    assert math.isnan(before["loss"])               # nothing measured yet

    for _ in range(3):
        opt.step(closure)

    stats = opt.stats()
    assert set(stats) == {
        "iteration", "F0", "loss", "momentum_norm", "skipped_updates"
    }
    assert all(isinstance(v, float) for v in stats.values())
    assert stats["iteration"] == 3.0
    assert stats["skipped_updates"] == 0.0
    assert stats["F0"] == -1.0
    assert stats["loss"] < QUADRATIC_INITIAL_LOSS
    assert stats["momentum_norm"] == pytest.approx(1.0, rel=0.1)


# --------------------------------------------------------------------------- #
# save / restore (train.py --resume)
# --------------------------------------------------------------------------- #
#: Scalars that must survive a save/load round trip (see `_EGD_STATE_KEYS`).
ROUND_TRIP_SCALARS = (
    "iteration", "F0", "F0_resolved", "dim", "lr", "nu", "eps1", "eps2",
    "weight_decay", "consEn", "Finit", "p2", "_initialized", "skipped_updates",
)


def test_state_dict_round_trip_restores_the_scalars_and_the_rng():
    """Every scalar and the generator state must come back from ``state_dict()``."""
    w, opt, closure = _quadratic_problem(nu=0.5, seed=7)
    for _ in range(7):
        opt.step(closure)
    assert opt.iteration == 7

    saved = opt.state_dict()
    # The standard structure is untouched; exactly one extra key is added.
    assert set(saved) == {"state", "param_groups", "egd"}

    other_w, other, _ = _quadratic_problem(nu=0.5, seed=7)
    assert other.iteration == 0 and other._initialized is False

    result = other.load_state_dict(saved)
    assert result is None                            # same as torch.optim.Optimizer

    for key in ROUND_TRIP_SCALARS:
        assert getattr(other, key) == getattr(opt, key), key

    # ``lr`` and ``nu`` come back *already rescaled* by the original run ...
    assert other.lr == pytest.approx(1.0 / math.sqrt(100.0))
    assert other.nu == pytest.approx(0.5 / math.sqrt(other.dim))
    # ... and the initialisation block must not run again.
    assert other._initialized is True
    assert other._restored_egd_state is True
    assert other.restored_egd_state is True

    # The dedicated noise generator, including its stream position.
    assert other.generator is not None
    assert torch.equal(other.generator.get_state(), opt.generator.get_state())

    # The momenta travel in the ordinary per-parameter state.
    for old, new in zip(opt.param_groups[0]["params"], other.param_groups[0]["params"]):
        assert torch.equal(opt.state[old]["momenta"], other.state[new]["momenta"])

    # ``param_groups`` (what a checkpoint stores) agrees with the effective values.
    assert other.param_groups[0]["lr"] == pytest.approx(other.lr)
    assert other.param_groups[0]["nu"] == pytest.approx(other.nu)
    assert other.param_groups[0]["F0"] == other.F0


def test_resume_continues_the_exact_trajectory():
    """save -> fresh optimizer -> load must continue the run bit-for-bit.

    ``nu=0.5`` makes the noise (and therefore the saved RNG state) matter.
    """
    n_head, n_tail = 6, 6

    w_ref, opt_ref, closure_ref = _quadratic_problem(nu=0.5, seed=7)
    reference = [float(opt_ref.step(closure_ref).detach()) for _ in range(n_head + n_tail)]

    w_run, opt_run, closure_run = _quadratic_problem(nu=0.5, seed=7)
    head = [float(opt_run.step(closure_run).detach()) for _ in range(n_head)]
    saved = opt_run.state_dict()

    # A *fresh* optimizer (different lr/eta/F0/nu, random seed) is resumed from
    # the checkpoint, exactly like ``train.py --resume`` does.
    resumed = EGD([w_run], lr=1.0, eta=1.0, F0=42.0, nu=0.0, seed=None)
    resumed.load_state_dict(saved)

    def closure_resumed() -> torch.Tensor:
        resumed.zero_grad()
        loss = 0.5 * ((w_run - torch.zeros_like(w_run)) ** 2).sum()
        loss.backward()
        return loss

    tail = [float(resumed.step(closure_resumed).detach()) for _ in range(n_tail)]
    continued = head + tail

    print("reference trajectory:", [round(v, 8) for v in reference])
    print("resumed  trajectory:", [round(v, 8) for v in continued])

    torch.testing.assert_close(
        torch.tensor(continued), torch.tensor(reference), rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(w_run.detach(), w_ref.detach(), rtol=0.0, atol=0.0)
    assert resumed.iteration == opt_ref.iteration
    assert resumed.skipped_updates == opt_ref.skipped_updates

    # Stronger: after the continuation the two optimizers are indistinguishable,
    # scalars, momenta and RNG stream included.
    state_resumed = resumed.state_dict()
    state_reference = opt_ref.state_dict()
    assert set(state_resumed["egd"]) == set(state_reference["egd"])
    for key, value in state_resumed["egd"].items():
        expected = state_reference["egd"][key]
        if isinstance(value, torch.Tensor) or isinstance(expected, torch.Tensor):
            assert torch.equal(value, expected), key
        else:
            assert value == expected, key
    for src, dst in zip(
        opt_ref.param_groups[0]["params"], resumed.param_groups[0]["params"]
    ):
        assert torch.equal(
            opt_ref.state[src]["momenta"], resumed.state[dst]["momenta"]
        )

    # The test is sensitive: without the carried RNG state the noise sequence --
    # and therefore the trajectory -- differs.
    w_bad, opt_bad, closure_bad = _quadratic_problem(nu=0.5, seed=7)
    for _ in range(n_head):
        opt_bad.step(closure_bad)
    opt_bad.load_state_dict(saved)
    opt_bad.generator.manual_seed(123456)             # scramble the saved stream
    bad_tail = [float(opt_bad.step(closure_bad).detach()) for _ in range(n_tail)]
    assert not torch.allclose(
        torch.tensor(bad_tail), torch.tensor(reference[n_head:]), rtol=0.0, atol=0.0
    )


def test_load_state_dict_without_egd_key_is_tolerated():
    """Older / foreign checkpoints must load, with a clear signal and no crash."""
    w, opt, closure = _quadratic_problem(nu=0.5, seed=7)
    for _ in range(3):
        opt.step(closure)

    saved = opt.state_dict()
    saved.pop("egd")

    old_w, old, _ = _quadratic_problem(nu=0.5, seed=7)
    with pytest.warns(RuntimeWarning, match="egd"):
        result = old.load_state_dict(saved)

    assert result is None
    assert old._restored_egd_state is False
    # The EGD scalars keep their freshly-initialised values ...
    assert old._initialized is False
    assert old.iteration == 0
    assert old.lr == 1.0                              # raw, rescaled at init
    assert old.F0 == -1.0
    # ... while parameters and momenta were restored.
    for src, dst in zip(opt.param_groups[0]["params"], old.param_groups[0]["params"]):
        assert torch.equal(opt.state[src]["momenta"], old.state[dst]["momenta"])

    # It still runs: the next step initialises the dynamics again.
    def closure_old() -> torch.Tensor:
        old.zero_grad()
        loss = 0.5 * ((old_w - torch.zeros_like(old_w)) ** 2).sum()
        loss.backward()
        return loss

    old.step(closure_old)
    assert old._initialized is True and old.iteration == 1
    assert old.lr == pytest.approx(1.0 / math.sqrt(100.0))


def test_load_state_dict_accepts_an_adamw_style_dict():
    """A genuine ``AdamW`` state dict (no momenta, no ``egd``) must not raise."""
    w = torch.nn.Parameter(torch.tensor([2.0, -3.0, 1.0, 0.5]))
    adam = torch.optim.AdamW([w], lr=1e-3)
    (w ** 2).sum().backward()
    adam.step()

    target = torch.nn.Parameter(torch.tensor([2.0, -3.0, 1.0, 0.5]))
    egd = EGD([target], lr=1.0, eta=100.0, F0=-1.0, seed=0)
    with pytest.warns(RuntimeWarning, match="egd"):
        egd.load_state_dict(adam.state_dict())
    assert egd._restored_egd_state is False
    assert "exp_avg" in egd.state[target]             # AdamW state was restored

    def closure() -> torch.Tensor:
        egd.zero_grad()
        loss = 0.5 * ((target - torch.zeros_like(target)) ** 2).sum()
        loss.backward()
        return loss

    egd.step(closure)                                 # must not raise
    assert egd.iteration == 1
    assert egd.dim == 4


def test_state_dict_survives_torch_save_and_load_on_disk(tmp_path):
    """``torch.save``/``torch.load`` (default ``weights_only=True``) must work."""
    w, opt, closure = _quadratic_problem(nu=0.5, seed=7)
    for _ in range(5):
        opt.step(closure)

    path = tmp_path / "egd_optimizer.pt"
    torch.save(opt.state_dict(), path)
    assert path.exists() and path.stat().st_size > 0

    loaded = torch.load(path)                         # train.py does exactly this
    assert "egd" in loaded

    other_w, other, _ = _quadratic_problem(nu=0.5, seed=7)
    # ``train.py --resume`` restores the *parameters* from ``ckpt["model"]`` and
    # only then the optimizer state; do the same here.
    other_w.data.copy_(w.detach())
    other.load_state_dict(loaded)

    assert other._restored_egd_state is True
    for key in ROUND_TRIP_SCALARS:
        assert getattr(other, key) == getattr(opt, key), key
    assert torch.equal(other.generator.get_state(), opt.generator.get_state())

    # ... and the resumed optimizer keeps matching the original run.
    reference = [float(opt.step(closure).detach()) for _ in range(3)]

    def closure_other() -> torch.Tensor:
        other.zero_grad()
        loss = 0.5 * ((other_w - torch.zeros_like(other_w)) ** 2).sum()
        loss.backward()
        return loss

    continued = [float(other.step(closure_other).detach()) for _ in range(3)]
    torch.testing.assert_close(
        torch.tensor(continued), torch.tensor(reference), rtol=0.0, atol=0.0
    )
