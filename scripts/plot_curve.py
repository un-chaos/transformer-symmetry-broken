#!/usr/bin/env python3
"""
Plot the training curve of a finished run.

Reads ``runs/<name>/training_log.csv`` -- the header-first CSV written by
``scripts/train.py`` -- and draws two panels: cross-entropy (train and val) and
the symmetry-breaking bias norms (``embed_b_norm``, ``bQ_norm``, ``bK_norm``,
``bV_norm``).  The x axis is the ``step`` column (one optimizer step); a log
that calls that column ``update`` instead also plots.  The figure is written as
``training_curve.png`` next to the CSV.

The plotting itself lives in :func:`plot_curve`, which ``scripts/train.py``
imports and calls at the end of a run, so the two never disagree about what a
curve looks like.  ``matplotlib`` is imported lazily: when it is not installed
the function prints a one-line note and returns ``None`` instead of raising.

Usage Examples:
    # a whole run directory (training_log.csv is looked up inside it)
    python scripts/plot_curve.py --run runs/small-egd-bgaussian-seed42

    # or the CSV itself, with an explicit output path and title
    python scripts/plot_curve.py --run runs/small-egd-symmetric-seed42/training_log.csv \
        --out /tmp/control.png --title "symmetric control"

    # positional form works too
    python scripts/plot_curve.py runs/small-egd-bconst-seed42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

#: X-axis column of the training log.  ``step`` is what ``scripts/train.py``
#: writes (one optimizer step); ``update`` is accepted too, so a log written by
#: the upstream-style header still plots instead of failing.
X_COLUMNS = ("step", "update")
#: Loss columns of the training log, plotted in this order.
LOSS_COLUMNS = ("train_loss", "val_loss")
#: Bias-norm columns of the training log, plotted in this order.
BIAS_COLUMNS = ("embed_b_norm", "bQ_norm", "bK_norm", "bV_norm")
#: Filename a run directory is expected to contain.
LOG_FILENAME = "training_log.csv"
#: Filename written next to the CSV when ``--out`` is not given.
DEFAULT_OUT = "training_curve.png"


def resolve_csv_path(target) -> Path:
    """Turn a run directory (or a CSV path) into the CSV path to plot.

    A directory resolves to ``<dir>/training_log.csv``; a file is taken as-is.
    Raises ``FileNotFoundError`` naming what was looked for, so a typo fails
    loudly instead of producing an empty figure.
    """
    path = Path(target)
    if path.is_dir():
        path = path / LOG_FILENAME
    if not path.exists():
        raise FileNotFoundError(
            f"no training log at {path}; pass a run directory (containing "
            f"{LOG_FILENAME}) or the CSV itself"
        )
    return path


def _read_log(csv_path: Path):
    """Read the CSV without pandas, returning ``{column: [values]}``.

    Blank fields (a train row has no ``val_loss`` and vice versa) become
    ``float("nan")``, which matplotlib simply skips, so the two row kinds can
    share one file.
    """
    import csv as _csv

    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = _csv.DictReader(fh)
        columns: List[str] = list(reader.fieldnames or [])
        data = {name: [] for name in columns}
        for row in reader:
            for name in columns:
                raw = (row.get(name) or "").strip()
                if raw == "":
                    data[name].append(float("nan"))
                    continue
                try:
                    data[name].append(float(raw))
                except ValueError:
                    data[name].append(float("nan"))
    return columns, data


def _has_values(data, column: str) -> bool:
    """True when ``column`` exists and has at least one non-NaN entry."""
    values = data.get(column) or []
    return any(value == value for value in values)  # NaN != NaN


def plot_curve(csv_path, out_path=None, title: str = "") -> Optional[Path]:
    """Plot train/val loss and the bias norms from a run's CSV log.

    Args:
        csv_path: the ``training_log.csv`` to read (a path, not a directory).
        out_path: where to write the PNG; defaults to ``training_curve.png``
            next to the CSV.
        title: figure title; the parent directory name is used when empty.

    Returns:
        The written PNG path, or ``None`` when matplotlib is unavailable, the
        CSV is missing, or the log is empty.
    """
    csv_path = Path(csv_path)
    out_path = Path(out_path) if out_path else csv_path.with_name(DEFAULT_OUT)
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: this machine has no display
        import matplotlib.pyplot as plt
    except Exception as exc:  # optional dependency, or no usable backend
        print(f"[plot] skipped ({type(exc).__name__}: {exc})")
        return None

    if not csv_path.exists():
        print(f"[plot] skipped, no CSV at {csv_path}")
        return None

    try:
        columns, data = _read_log(csv_path)
    except (OSError, ValueError) as exc:
        print(f"[plot] skipped ({type(exc).__name__}: {exc})")
        return None
    x_column = next((c for c in X_COLUMNS if _has_values(data, c)), None)
    if x_column is None:
        print(f"[plot] skipped, no {'/'.join(X_COLUMNS)} column in {csv_path.name}")
        return None

    loss_cols = [c for c in LOSS_COLUMNS if _has_values(data, c)]
    bias_cols = [c for c in BIAS_COLUMNS if _has_values(data, c)]
    panels = 1 + (1 if bias_cols else 0)
    fig, axes = plt.subplots(1, panels, figsize=(6.5 * panels, 4.2), squeeze=False)

    ax = axes[0][0]
    for column in loss_cols:
        ax.plot(data[x_column], data[column], label=column, linewidth=1.3)
    ax.set_xlabel(x_column)
    ax.set_ylabel("cross-entropy")
    ax.set_title(title or csv_path.parent.name)
    ax.grid(alpha=0.3)
    if loss_cols:
        ax.legend()

    if bias_cols:
        ax2 = axes[0][1]
        for column in bias_cols:
            ax2.plot(data[x_column], data[column], label=column, linewidth=1.2)
        ax2.set_xlabel(x_column)
        ax2.set_ylabel("||bias||")
        ax2.set_title("symmetry-breaking bias norms")
        ax2.grid(alpha=0.3)
        ax2.legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Plot the training curve of a finished run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "target",
        nargs="?",
        default=None,
        help=f"run directory or {LOG_FILENAME} path (positional form)",
    )
    ap.add_argument(
        "--run",
        default=None,
        help=f"run directory or {LOG_FILENAME} path (same as the positional form)",
    )
    ap.add_argument("--out", default=None, help=f"output PNG (default: {DEFAULT_OUT})")
    ap.add_argument("--title", default="", help="figure title (default: the run name)")
    return ap


def main() -> None:
    args = build_parser().parse_args()
    target = args.run or args.target
    if target is None:
        raise SystemExit("nothing to plot: pass a run directory or a CSV path (--run)")
    csv_path = resolve_csv_path(target)
    plot_curve(csv_path, out_path=args.out, title=args.title)


if __name__ == "__main__":
    main()
