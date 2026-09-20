"""Small shared helpers: seeding, device choice, logging, timing."""

from __future__ import annotations

import csv
import json
import os
import platform
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Reproducibility / device
# --------------------------------------------------------------------------- #
def seed_everything(seed: int, deterministic: bool = False) -> None:
    """Seed python, numpy and torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover - older torch
            pass


def resolve_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto``/``cpu``/``cuda``/``cuda:N``/``mps`` into a ``torch.device``."""
    requested = (requested or "auto").lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"[warn] device={requested!r} requested but CUDA is unavailable; using CPU")
        return torch.device("cpu")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        print("[warn] device='mps' requested but MPS is unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(requested)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def environment_report() -> str:
    """One-line environment summary, stored with every run for traceability."""
    return (
        f"python={sys.version.split()[0]} torch={torch.__version__} "
        f"cuda={torch.cuda.is_available()} os={platform.system()}-{platform.release()}"
    )


def configure_console_encoding() -> None:
    """Force UTF-8 on stdout/stderr.

    On this machine the Windows console defaults to **GBK**, so printing the
    corpus (multi30k German, with characters such as ``ß``) raises
    ``UnicodeEncodeError`` and kills an otherwise healthy run.  Anything that
    echoes source/target text calls this first.  Harmless where the streams are
    already UTF-8 or are not reconfigurable (e.g. captured pipes).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            pass


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
class CSVLogger:
    """Append-only CSV logger with a fixed header written once."""

    def __init__(self, path, fieldnames: Iterable[str]):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        # Truncate on first use so a re-run does not append to stale results.
        with self.path.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.fieldnames).writeheader()

    def log(self, row: Dict[str, Any]) -> None:
        clean = {k: row.get(k, "") for k in self.fieldnames}
        with self.path.open("a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=self.fieldnames).writerow(clean)

    def write_meta(self, meta: Dict[str, Any]) -> None:
        """Write run metadata next to the CSV (not inside it)."""
        meta_path = self.path.with_suffix(".meta.json")
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, ensure_ascii=False, default=str)


def format_seconds(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


# --------------------------------------------------------------------------- #
# Stall detection (the "is it hung?" guard)
# --------------------------------------------------------------------------- #
class StepWatchdog:
    """Track per-step wall-clock time and shout when a step looks stuck.

    A long run on CPU can appear frozen; this turns that into an explicit
    message naming the step, the elapsed time and what to check.
    """

    def __init__(self, stall_warn_seconds: float = 300.0, every: int = 1):
        self.stall_warn_seconds = float(stall_warn_seconds)
        self.every = max(1, int(every))
        self._last = time.time()
        self.max_seen = 0.0
        self.warned = 0

    def tick(self, step: int) -> float:
        now = time.time()
        dt = now - self._last
        self._last = now
        self.max_seen = max(self.max_seen, dt)
        if self.stall_warn_seconds > 0 and dt > self.stall_warn_seconds and step % self.every == 0:
            self.warned += 1
            print(
                f"[stall-watchdog] step {step} took {format_seconds(dt)} "
                f"(> {format_seconds(self.stall_warn_seconds)}). "
                "Not necessarily hung -- check CPU usage, batch size, and "
                "whether the data loader is still producing batches.",
                flush=True,
            )
        return dt


@dataclass
class Timer:
    """Simple elapsed-time helper."""

    start: float = 0.0

    def __post_init__(self) -> None:
        self.start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start

    @property
    def pretty(self) -> str:
        return format_seconds(self.elapsed)

    def reset(self) -> None:
        self.start = time.time()
