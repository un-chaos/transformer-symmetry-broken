#!/usr/bin/env python3
"""
Compare every finished run under a log directory in one figure plus one
plain-language Chinese summary.

This is the "look at the effect of the bias" entry point for someone who does
not want to read CSVs.  It walks ``log_dir`` (default ``runs/``) for immediate
subdirectories that contain a ``training_log.csv`` -- the header-first log
written by ``scripts/train.py`` -- reads every per-run artifact that happens to
exist (``summary.json``, ``config.json``, ``bias.json``, ``args.json`` and the
optional ``eval_test.json`` written by ``scripts/evaluate.py``), and writes three
things into ``<log_dir>/_compare``:

- ``compare.png`` -- one figure, two panels: validation loss vs. step and
  training loss vs. step, one line per run (skipped with ``--no_plot``, or when
  matplotlib is unavailable).  Labels are Chinese when a CJK font is installed
  and English otherwise, so the figure never renders tofu boxes.
- ``compare.csv`` -- one row per run, for a spreadsheet.
- ``compare.txt`` -- the same content that is printed to stdout: an aligned
  table plus a "怎么看" reading guide and a "说明" legend, in Chinese.

Missing artifacts are never fatal: a run whose log is empty, or that was never
evaluated on the test split, still appears (sorted last / with ``-``).  A
``log_dir`` that does not exist, or that holds no run at all, prints a friendly
message and exits 0 instead of raising.

Usage Examples:
    # every run under runs/, figure + csv + txt into runs/_compare
    python scripts/report.py

    # a different log directory, an explicit output directory and a title
    python scripts/report.py --log_dir runs --out runs/_compare --title "b=0 vs b~N(0,0.02)"

    # metrics only, no matplotlib involved
    python scripts/report.py --log_dir runs --no_plot

    # as a library (no side effects on import)
    from scripts.report import build_report
    report = build_report("runs", out=None, make_plot=True, title=None)
    print(report["best_run"], report["n_runs"])
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# `main.py` dispatches through ``runpy.run_path``, which -- unlike running this
# file directly -- does not put this directory on ``sys.path``, so the same
# guard ``scripts/evaluate.py`` uses is repeated here.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

#: The per-run artifact that makes a directory a run.
LOG_FILENAME = "training_log.csv"
#: Outputs written into ``<out>``.
CSV_FILENAME = "compare.csv"
TXT_FILENAME = "compare.txt"
PNG_FILENAME = "compare.png"
#: Default output directory, relative to ``log_dir``.
OUT_DIRNAME = "_compare"

#: X-axis column of the training log (``step`` is what ``scripts/train.py``
#: writes; ``update`` is accepted so an upstream-style header still plots).
X_COLUMNS = ("step", "update")
TRAIN_COLUMN = "train_loss"
VAL_COLUMN = "val_loss"

#: compare.csv columns, in order.  ``epochs`` and ``steps`` stay separate here
#: (a spreadsheet prefers numbers); the text table joins them.
CSV_COLUMNS = (
    "run",
    "embed_mode",
    "attn_sectors",
    "model",
    "dataset_preset",
    "optimizer",
    "epochs",
    "steps",
    "parameters",
    "best_val_loss",
    "test_loss",
    "bleu",
    "elapsed_sec",
)

#: Placeholder for "this artifact was not produced".
MISSING = "-"

#: CJK families to look for, best first.  All three of ``Microsoft YaHei`` /
#: ``SimHei`` / ``SimSun`` ship with Windows; the rest make the list portable.
CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Source Han Sans CN",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Heiti SC",
    "Arial Unicode MS",
)

# --------------------------------------------------------------------------- #
# How big a validation-loss gap counts as real.  The README's own lr sweep on
# multi30k-tiny moves the val loss over a ~0.5 range, and repeated seeds land
# within a few hundredths of each other, so:
#: below this absolute gap (nats) the two runs are indistinguishable from noise.
NOISE_GAP = 0.05
#: at or above this absolute gap (nats) the difference is worth believing.
MEANINGFUL_GAP = 0.20
#: ... or this relative gap, whichever comes first.
MEANINGFUL_REL = 0.03

#: Extra columns of compare.png, used when no CJK font is installed.
ENGLISH_LABELS = {
    "x": "step",
    "val": "validation loss (val_loss)",
    "train": "training loss (train_loss)",
    "val_title": "validation loss vs step  (lower is better)",
    "train_title": "training loss vs step  (lower is better)",
    "default_title": "loss curves of every run under the log directory",
    "no_data": "no data for this panel",
}

#: Same, in Chinese.
CHINESE_LABELS = {
    "x": "训练步数（step）",
    "val": "验证损失（val_loss，越低越好）",
    "train": "训练损失（train_loss，仅供参考）",
    "val_title": "验证损失 vs 步数（越低越好）",
    "train_title": "训练损失 vs 步数（越低越好）",
    "default_title": "各次运行的损失曲线对比",
    "no_data": "这一栏没有可用数据",
}

DEFAULT_TITLE = "对称性破缺偏置对比报告"


# --------------------------------------------------------------------------- #
# Console / printing
# --------------------------------------------------------------------------- #
def _configure_console_encoding() -> None:
    """Force UTF-8 on stdout/stderr.

    This machine's Windows console defaults to GBK, which cannot encode the
    output of a Chinese report; ``scripts/train.py`` and the other scripts call
    the same helper (``symbreak_transformer.utils.configure_console_encoding``)
    for the same reason.  Implemented locally so importing this module stays
    free of the heavy ``torch`` import that ``utils`` pulls in.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            pass


def _safe_print(text: str) -> None:
    """Print ``text`` without ever dying on a non-UTF-8 console."""
    try:
        print(text)
        return
    except UnicodeEncodeError:
        pass
    try:
        sys.stdout.buffer.write(text.encode("utf-8", "replace") + b"\n")
        sys.stdout.flush()
    except Exception:  # pragma: no cover - last resort, keeps the CLI alive
        print(text.encode("ascii", "replace").decode("ascii"))


# --------------------------------------------------------------------------- #
# Small readers
# --------------------------------------------------------------------------- #
def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON object, returning ``None`` when it is absent or unusable."""
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_log(csv_path: Path) -> Tuple[List[str], List[Dict[str, Optional[float]]]]:
    """Read ``training_log.csv`` into ``(columns, rows)``.

    Blank fields -- a train row carries no ``val_loss`` and a val row carries no
    ``train_loss`` -- become ``None``; an unparseable value does too.  A file
    that is empty yields no columns, a header-only file yields columns and zero
    rows, which is exactly the distinction the caller reports on.
    """
    columns: List[str] = []
    rows: List[Dict[str, Optional[float]]] = []
    try:
        with Path(csv_path).open("r", newline="", encoding="utf-8-sig") as fh:
            reader = csv.DictReader(fh)
            columns = [name for name in (reader.fieldnames or []) if name]
            for raw_row in reader:
                row: Dict[str, Optional[float]] = {}
                for name in columns:
                    raw = (raw_row.get(name) or "").strip()
                    if raw == "":
                        row[name] = None
                        continue
                    try:
                        row[name] = float(raw)
                    except ValueError:
                        row[name] = None
                rows.append(row)
    except (OSError, UnicodeDecodeError, csv.Error):
        return [], []
    return columns, rows


def _series(
    rows: Sequence[Dict[str, Optional[float]]],
    columns: Sequence[str],
    loss_column: str,
) -> List[Tuple[float, float]]:
    """Collect ``(step, loss)`` pairs, skipping rows that lack either number.

    The x value is the first usable of ``step`` / ``update``; when the log has
    neither (an exotic header) the row index is used so a curve still appears.
    """
    x_column = next((c for c in X_COLUMNS if c in columns), None)
    points: List[Tuple[float, float]] = []
    for index, row in enumerate(rows):
        loss = row.get(loss_column)
        if loss is None or not math.isfinite(float(loss)):
            continue
        x = row.get(x_column) if x_column else None
        if x is None or not math.isfinite(float(x)):
            x = float(index + 1)
        points.append((float(x), float(loss)))
    return points


# --------------------------------------------------------------------------- #
# Per-run loading
# --------------------------------------------------------------------------- #
def _blank_run(run_dir: Path) -> Dict[str, Any]:
    """A run record with every field present, so downstream code never KeyErrors."""
    return {
        "name": run_dir.name,
        "run_dir": str(run_dir),
        "notes": [],
        "log_rows": 0,
        "log_columns": [],
        "train_series": [],
        "val_series": [],
        "best_val_loss": None,
        "best_val_step": None,
        # summary.json
        "summary": None,
        "summary_run_name": None,
        "steps": None,
        "parameters": None,
        "optimizer": None,
        "elapsed_sec": None,
        # config.json
        "n_embd": None,
        "n_head": None,
        "n_encoder_layer": None,
        "n_decoder_layer": None,
        "context_length": None,
        # bias.json
        "embed_mode": None,
        "use_q_bias": False,
        "use_k_bias": False,
        "use_v_bias": False,
        "attn_mode": None,
        "embed_learnable": False,
        "attn_learnable": False,
        # args.json
        "model": None,
        "dataset_preset": None,
        "epochs": None,
        "max_steps": None,
        "batch_size": None,
        "args_name": None,
        # eval_test.json
        "has_eval": False,
        "test_loss": None,
        "bleu": None,
    }


def _number(value: Any) -> Optional[float]:
    """Coerce ``value`` to a finite float, or ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_run(run_dir: Path) -> Dict[str, Any]:
    """Read every artifact of one run, tolerating each one being missing."""
    run = _blank_run(run_dir)
    log_path = run_dir / LOG_FILENAME

    columns, rows = _read_log(log_path)
    run["log_columns"] = columns
    run["log_rows"] = len(rows)
    run["train_series"] = _series(rows, columns, TRAIN_COLUMN)
    run["val_series"] = _series(rows, columns, VAL_COLUMN)
    if not columns:
        run["notes"].append(f"{LOG_FILENAME} 是空文件（连表头都没有）")
    elif not rows:
        run["notes"].append(f"{LOG_FILENAME} 只有表头，没有数据行")
    elif not run["val_series"]:
        run["notes"].append(
            f"{LOG_FILENAME} 里没有验证损失（{VAL_COLUMN}）数据，画不出验证曲线"
        )
    if run["val_series"]:
        step, loss = min(run["val_series"], key=lambda point: point[1])
        run["best_val_loss"] = loss
        run["best_val_step"] = step

    summary = _read_json(run_dir / "summary.json")
    run["summary"] = summary
    if summary:
        run["summary_run_name"] = summary.get("run_name")
        run["steps"] = _number(summary.get("steps"))
        run["parameters"] = _number(summary.get("parameters"))
        run["optimizer"] = summary.get("optimizer")
        run["elapsed_sec"] = _number(summary.get("elapsed_sec"))
        if run["best_val_loss"] is None:
            # A run whose log has no val row still knows its best loss here.
            run["best_val_loss"] = _number(summary.get("best_val_loss"))

    config = _read_json(run_dir / "config.json")
    if config:
        for key in (
            "n_embd",
            "n_head",
            "n_encoder_layer",
            "n_decoder_layer",
            "context_length",
        ):
            run[key] = _number(config.get(key))

    bias = _read_json(run_dir / "bias.json")
    if bias:
        run["embed_mode"] = bias.get("embed_mode")
        run["attn_mode"] = bias.get("attn_mode")
        run["use_q_bias"] = bool(bias.get("use_q_bias"))
        run["use_k_bias"] = bool(bias.get("use_k_bias"))
        run["use_v_bias"] = bool(bias.get("use_v_bias"))
        run["embed_learnable"] = bool(bias.get("embed_learnable"))
        run["attn_learnable"] = bool(bias.get("attn_learnable"))

    args = _read_json(run_dir / "args.json")
    if args:
        run["model"] = args.get("model")
        run["dataset_preset"] = args.get("dataset_preset")
        run["epochs"] = _number(args.get("epochs"))
        run["max_steps"] = _number(args.get("max_steps"))
        run["batch_size"] = _number(args.get("batch_size"))
        run["args_name"] = args.get("name")
        if run["optimizer"] is None:
            run["optimizer"] = args.get("optimizer")

    if run["steps"] is None:
        all_steps = [point[0] for point in run["train_series"] + run["val_series"]]
        if all_steps:
            run["steps"] = max(all_steps)
        elif run["max_steps"]:
            run["steps"] = run["max_steps"]

    eval_path = run_dir / "eval_test.json"
    if eval_path.exists():
        payload = _read_json(eval_path)
        if payload is None:
            run["notes"].append("eval_test.json 存在但读不出来（内容不是合法 JSON）")
        else:
            run["has_eval"] = True
            run["test_loss"] = _number(payload.get("loss"))
            bleu = payload.get("bleu")
            if isinstance(bleu, dict):
                run["bleu"] = _number(bleu.get("bleu"))
            else:
                run["bleu"] = _number(bleu)
    else:
        run["notes"].append("没有 eval_test.json（还没跑 evaluate，测试损失/BLEU 记为 -）")

    return run


def _sort_key(run: Dict[str, Any]) -> Tuple[int, float, str]:
    """Best validation loss first; runs with no usable curve last.

    The tier keeps the ordering honest: a run with real val rows always beats a
    run whose log has rows but no ``val_loss``, which in turn beats an empty log
    -- even when a ``summary.json`` supplies a best loss for the latter.
    """
    if run["val_series"]:
        tier = 0
    elif run["train_series"]:
        tier = 1
    else:
        tier = 2
    best = run["best_val_loss"]
    finite = isinstance(best, (int, float)) and math.isfinite(best)
    ranked = float(best) if finite else float("inf")
    return (tier, ranked, run["name"])


def _find_runs(log_dir: Path) -> List[Path]:
    """Immediate subdirectories of ``log_dir`` that contain a training log."""
    try:
        children = sorted(log_dir.iterdir(), key=lambda path: path.name)
    except OSError:
        return []
    return [path for path in children if path.is_dir() and (path / LOG_FILENAME).is_file()]


# --------------------------------------------------------------------------- #
# Presentation helpers
# --------------------------------------------------------------------------- #
def _display_width(text: str) -> int:
    """Terminal width of ``text``, counting CJK characters as two columns."""
    wide = ("W", "F")
    return sum(2 if unicodedata.east_asian_width(ch) in wide else 1 for ch in str(text))


def _fit(text: str, width: int) -> str:
    """Left-align ``text`` in ``width`` display columns, truncating if needed."""
    text = str(text)
    if _display_width(text) <= width:
        return text + " " * (width - _display_width(text))
    kept, used = [], 0
    for ch in text:
        step = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if used + step > max(width - 1, 1):
            break
        kept.append(ch)
        used += step
    return "".join(kept) + "…"


def _render_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """Render an aligned plain-text table (CJK-aware), header first."""
    widths = []
    for index, header in enumerate(headers):
        width = _display_width(header)
        for row in rows:
            if index < len(row):
                width = max(width, _display_width(row[index]))
        widths.append(min(width, 34))

    def render(row: Sequence[str]) -> str:
        cells = [row[i] if i < len(row) else "" for i in range(len(headers))]
        return "  ".join(_fit(cell, widths[i]) for i, cell in enumerate(cells)).rstrip()

    header_line = render(headers)
    lines = [header_line, "-" * _display_width(header_line)]
    lines.extend(render(row) for row in rows)
    return lines


def _fmt(value: Optional[float], digits: int = 4, missing: str = MISSING) -> str:
    """Format a finite number, or the placeholder."""
    number = _number(value)
    return missing if number is None else f"{number:.{digits}f}"


def _fmt_int(value: Optional[float], missing: str = MISSING, group: bool = False) -> str:
    """Format a count, or the placeholder."""
    number = _number(value)
    if number is None:
        return missing
    return f"{int(number):,}" if group else str(int(number))


def _attn_sectors(run: Dict[str, Any], missing: str = MISSING) -> str:
    """``"Q+V(gaussian)"``-style description of which attention biases are on."""
    sectors = "+".join(
        name
        for name, flag in (
            ("Q", run["use_q_bias"]),
            ("K", run["use_k_bias"]),
            ("V", run["use_v_bias"]),
        )
        if flag
    )
    if not sectors:
        return missing
    text = sectors
    if run["attn_mode"]:
        text += f"({run['attn_mode']})"
    if run["attn_learnable"]:
        text += "+learnable"
    return text


def _embed_mode(run: Dict[str, Any], missing: str = MISSING) -> str:
    """``"gaussian+learnable"``-style description of the embedding bias."""
    mode = run["embed_mode"]
    if not mode:
        return missing
    return f"{mode}+learnable" if run["embed_learnable"] else str(mode)


def _bias_phrase(run: Dict[str, Any]) -> str:
    """A Chinese one-liner describing a run's symmetry-breaking settings."""
    chinese_modes = {
        "zero": "zero（对照：b=0，对称性完全没有被破坏）",
        "gaussian": "gaussian（随机方向的高斯偏置 b）",
        "const": "const（每个维度都是同一个常数的偏置 b）",
    }
    mode = run["embed_mode"]
    text = chinese_modes.get(str(mode), str(mode) if mode else "未知（没有 bias.json）")
    if run["embed_learnable"]:
        text += "，而且 b 是可训练的"
    sectors = _attn_sectors(run, missing="")
    if sectors:
        text += f"，另外注意力里开着 {sectors} 偏置"
    else:
        text += "，注意力里没有额外偏置"
    return text


def _bias_signature(run: Dict[str, Any]) -> Tuple[Any, ...]:
    """Everything that defines the run's symmetry-breaking setting."""
    return (
        run["embed_mode"],
        run["embed_learnable"],
        run["use_q_bias"],
        run["use_k_bias"],
        run["use_v_bias"],
        run["attn_mode"],
        run["attn_learnable"],
    )


def _table_rows(runs: Sequence[Dict[str, Any]], group: bool = False) -> List[List[str]]:
    """Rows of the text table (``group`` adds thousand separators)."""
    rows = []
    for run in runs:
        epochs = _fmt_int(run["epochs"])
        steps = _fmt_int(run["steps"])
        if epochs == MISSING and steps == MISSING:
            epochs_steps = MISSING
        else:
            epochs_steps = f"{epochs}/{steps}"
        rows.append(
            [
                run["name"],
                _embed_mode(run),
                _attn_sectors(run),
                str(run["model"] or MISSING),
                str(run["dataset_preset"] or MISSING),
                str(run["optimizer"] or MISSING),
                epochs_steps,
                _fmt_int(run["parameters"], group=group),
                _fmt(run["best_val_loss"]),
                _fmt(run["test_loss"]),
                _fmt(run["bleu"], digits=2),
                _fmt(run["elapsed_sec"], digits=1),
            ]
        )
    return rows


TABLE_HEADERS = (
    "运行名称",
    "embed_mode",
    "注意力偏置",
    "模型",
    "数据集",
    "优化器",
    "epochs/steps",
    "参数量",
    "最佳验证损失",
    "测试损失",
    "BLEU",
    "耗时(秒)",
)


def _csv_rows(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    """compare.csv records: same data as the text table, empty for missing."""
    records = []
    for run in runs:
        records.append(
            {
                "run": run["name"],
                "embed_mode": _embed_mode(run, missing=""),
                "attn_sectors": _attn_sectors(run, missing=""),
                "model": run["model"] or "",
                "dataset_preset": run["dataset_preset"] or "",
                "optimizer": run["optimizer"] or "",
                "epochs": _fmt_int(run["epochs"], missing=""),
                "steps": _fmt_int(run["steps"], missing=""),
                "parameters": _fmt_int(run["parameters"], missing=""),
                "best_val_loss": _fmt(run["best_val_loss"], missing=""),
                "test_loss": _fmt(run["test_loss"], missing=""),
                "bleu": _fmt(run["bleu"], digits=2, missing=""),
                "elapsed_sec": _fmt(run["elapsed_sec"], digits=1, missing=""),
            }
        )
    return records


def _model_ladder() -> List[str]:
    """Model presets ordered smallest to largest, for the "next step" advice.

    Sorted by a size proxy (``n_embd x layers``) rather than alphabetically, so
    the printed sentence "越往右模型越大" is actually true.  Guarded: a broken
    ``symbreak_transformer`` install degrades the advice, it never takes the
    report down.
    """
    try:
        from symbreak_transformer.config import PRESETS

        def size(name: str) -> Tuple[int, int]:
            cfg = PRESETS[name]
            return (
                int(cfg.n_embd) * int(cfg.n_encoder_layer + cfg.n_decoder_layer),
                int(cfg.n_embd),
            )

        ranked = sorted(((name, size(name)) for name in PRESETS), key=lambda item: item[1])
        return [name for name, _ in ranked]
    except Exception:  # pragma: no cover - advice only
        return []


def _dataset_ladder() -> List[str]:
    """Dataset presets ordered from least to most real training data.

    Synthetic presets are toy tasks with no real text, so they sit at the cheap
    end; for the real ones ``max_train_samples`` of 0/absent means "the whole
    file", i.e. the slowest.  Ordered this way the advice is an actual ladder
    (cheap → expensive) instead of an alphabetical list.
    """
    try:
        from symbreak_transformer.config import DATASET_PRESETS
    except Exception:  # pragma: no cover - advice only
        return []
    sized: List[Tuple[str, float]] = []
    for name in DATASET_PRESETS:
        kwargs = DATASET_PRESETS[name]
        if kwargs.get("source") != "hf":
            sized.append((name, 0.0))  # toy task: no real parallel text
            continue
        limit = kwargs.get("max_train_samples", 0)
        sized.append((name, float("inf") if not limit else float(limit)))
    sized.sort(key=lambda item: (item[1], item[0]))
    return [name for name, _ in sized]


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _detect_cjk_font(font_manager) -> Optional[str]:
    """First installed CJK family from :data:`CJK_FONT_CANDIDATES`, else ``None``."""
    try:
        installed = {entry.name for entry in font_manager.fontManager.ttflist}
    except Exception:  # pragma: no cover - font subsystem trouble
        return None
    return next((name for name in CJK_FONT_CANDIDATES if name in installed), None)


def _plot_compare(
    runs: Sequence[Dict[str, Any]], out_path: Path, title: Optional[str]
) -> Tuple[Optional[Path], List[str], Optional[str], Optional[str]]:
    """Draw validation/training loss vs step, one line per run.

    Returns ``(png_path, glyph_warnings, font_used, skip_reason)``.  matplotlib
    is imported lazily, exactly like ``scripts/plot_curve.py``: without it the
    figure is skipped with a one-line note instead of an exception.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: this machine has no display
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception as exc:
        reason = f"matplotlib 不可用（{type(exc).__name__}: {exc}）"
        _safe_print(f"[图表] 跳过绘图：{reason}")
        return None, [], None, reason

    font = _detect_cjk_font(font_manager)
    cjk = font is not None
    if cjk:
        matplotlib.rcParams["font.sans-serif"] = [font] + [
            name for name in CJK_FONT_CANDIDATES if name != font
        ] + ["DejaVu Sans"]
        matplotlib.rcParams["font.family"] = "sans-serif"
        matplotlib.rcParams["axes.unicode_minus"] = False
        labels = CHINESE_LABELS
    else:
        # No CJK font: silently switch to English instead of drawing tofu boxes.
        labels = ENGLISH_LABELS
        _safe_print(
            "[图表] 没有找到中文字体（Microsoft YaHei / SimHei / SimSun 都没装上），"
            "图里的文字改用英文。"
        )

    if not any(run["val_series"] or run["train_series"] for run in runs):
        reason = "没有任何可用的损失数据"
        _safe_print(f"[图表] 跳过绘图：{reason}")
        return None, [], font, reason

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
    panels = (
        (axes[0], "val_series", labels["val_title"], labels["val"]),
        (axes[1], "train_series", labels["train_title"], labels["train"]),
    )
    for ax, key, panel_title, ylabel in panels:
        plotted = 0
        for run in runs:
            series = run[key]
            if not series:
                continue
            ax.plot(
                [point[0] for point in series],
                [point[1] for point in series],
                marker="o",
                markersize=2.6,
                linewidth=1.4,
                label=run["name"],
            )
            plotted += 1
        ax.set_xlabel(labels["x"])
        ax.set_ylabel(ylabel)
        ax.set_title(panel_title)
        ax.grid(alpha=0.3)
        if plotted == 0:
            ax.text(
                0.5, 0.5, labels["no_data"], transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="0.45",
            )
        elif plotted > 1:
            ax.legend(fontsize=8, loc="best", framealpha=0.9)

    main_title = title or labels["default_title"]
    if len(runs) == 1:
        # No legend clutter on a single run: the name goes in the title.
        main_title = f"{main_title}（{runs[0]['name']}）"
    if cjk:
        main_title += f" — 共 {len(runs)} 次运行，验证损失越低越好"
    else:
        main_title += f" - {len(runs)} run(s), lower loss is better"
    fig.suptitle(main_title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))

    glyph_warnings: List[str] = []
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fig.savefig(out_path, dpi=150)
            glyph_warnings = [
                str(item.message)
                for item in caught
                if "Glyph" in str(item.message) or "missing from" in str(item.message)
            ]
    except OSError as exc:  # pragma: no cover - unwritable output directory
        plt.close(fig)
        reason = f"写不出图片（{type(exc).__name__}: {exc}）"
        _safe_print(f"[图表] 跳过绘图：{reason}")
        return None, [], font, reason
    finally:
        plt.close(fig)

    if glyph_warnings:
        _safe_print(f"[图表] 注意：有字形缺失警告，中文字体可能没生效：{glyph_warnings[0]}")
    else:
        _safe_print(f"[图表] 已写出 {out_path}（字体：{font or 'English / DejaVu Sans'}）")
    return out_path, glyph_warnings, font, None


# --------------------------------------------------------------------------- #
# Text report
# --------------------------------------------------------------------------- #
def _reading_guide(runs: Sequence[Dict[str, Any]]) -> List[str]:
    """The "怎么看" section: who won, by how much, and what to do next."""
    lines: List[str] = ["怎么看", "------"]
    ranked = [run for run in runs if run["val_series"]]
    without = [run for run in runs if not run["val_series"]]

    if not ranked:
        lines.append("  目前没有任何一次运行留下可用的验证损失（val_loss），所以还比不出高低。")
        if without:
            lines.append(f"  有 {len(without)} 个运行目录的 {LOG_FILENAME} 是空的或只有表头：")
            lines.append("    " + "、".join(run["name"] for run in without))
        lines.append("  先把训练正常跑完（让它至少写下几次验证损失），再重新生成这份报告。")
        return lines

    best = ranked[0]
    best_loss = float(best["best_val_loss"])
    number = 0

    def item(text: str) -> None:
        """Append one numbered point, so the numbering never skips."""
        nonlocal number
        number += 1
        lines.append(f"  {number}. {text}")

    item(
        f"验证损失最低的是「{best['name']}」，最佳验证损失 {best_loss:.4f}"
        + (f"（出现在第 {int(best['best_val_step'])} 步）。" if best["best_val_step"] else "。")
    )

    if len(ranked) == 1:
        item("现在只有 1 次运行有验证曲线，没有第二个可以对比，所以看不出偏置到底有没有效果。")
        item("想看出「对称性破缺偏置」的效果，至少要有两次只差偏置设置的运行，例如：")
        lines.append(
            "       python main.py train --model small --dataset_preset multi30k-tiny "
            "--bias_preset symmetric  --name small-egd-symmetric"
        )
        lines.append(
            "       python main.py train --model small --dataset_preset multi30k-tiny "
            "--bias_preset b-gaussian --name small-egd-bgaussian"
        )
        lines.append(
            "     两次都用同样的模型、同样的数据、同样的步数，只改 --bias_preset，"
            "跑完后重新执行本脚本即可。"
        )
        return lines

    second = ranked[1]
    second_loss = float(second["best_val_loss"])
    gap = second_loss - best_loss
    gap_rel = gap / best_loss if best_loss > 0 else 0.0
    item(
        f"第二名是「{second['name']}」，最佳验证损失 {second_loss:.4f}，"
        f"两者相差 {gap:.4f}（约 {gap_rel * 100:.1f}%）。"
    )
    if len(ranked) > 2:
        rest = "、".join(
            f"{run['name']}（{float(run['best_val_loss']):.4f}）" for run in ranked[2:]
        )
        lines.append(f"     其余 {rest}。")

    same_setting = _bias_signature(best) == _bias_signature(second)
    item(
        f"偏置设置：「{best['name']}」是{_bias_phrase(best)}；"
        f"「{second['name']}」是{_bias_phrase(second)}。"
    )
    if not same_setting:
        lines.append("     两次运行的偏置设置不同，所以这个差距有可能正是偏置造成的。")
    else:
        lines.append(
            "     注意：两次运行的偏置设置其实是一样的，这个差距只反映随机种子"
            "（--seed）、数据顺序这类波动，不能算到偏置头上。"
        )

    if without:
        item(
            f"另外 {len(without)} 个运行没有可用的验证损失，没有参与比较："
            + "、".join(run["name"] for run in without)
        )

    if gap < NOISE_GAP:
        item(
            f"判断：{gap:.4f} 这么小的差距，在这个规模上基本就是噪声——"
            "换个随机种子重跑一遍，胜负很可能就反过来，不能当成偏置的效果。"
        )
        noisy = True
    elif gap >= MEANINGFUL_GAP or gap_rel >= MEANINGFUL_REL:
        item(
            f"判断：{gap:.4f}（约 {gap_rel * 100:.1f}%）的差距已经比随机种子造成的"
            "起伏大，看起来是真有效果，值得沿着这个方向继续。"
        )
        noisy = False
    else:
        item(
            f"判断：{gap:.4f}（约 {gap_rel * 100:.1f}%）不大不小——比噪声大一点，"
            "但还不至于一眼就信。只有当这个方向在多次运行里都稳定赢，才算真效果。"
        )
        noisy = True

    if noisy:
        models = _model_ladder()
        datasets = _dataset_ladder()
        lines.append("     想让差距变得能看清，可以按下面的顺序试：")
        lines.append(
            "       a) 训练更久：--max_steps 调大（例如 4000），或让 --valid_every_updates "
            "小一点，等验证损失走平再比；"
        )
        if models:
            lines.append(
                "       b) 换更大的模型：--model " + " → ".join(models)
                + "（越往右模型越大、跑得越慢）；"
            )
        else:  # pragma: no cover - advice only
            lines.append(
                "       b) 换更大的模型：--model smoke → tiny → small → base → large"
                "（越往右越大、越慢）；"
            )
        if datasets:
            lines.append(
                "       c) 用更多数据：--dataset_preset " + " → ".join(datasets)
                + "（越往右数据越多/越真实、跑得越慢）；"
            )
        else:  # pragma: no cover - advice only
            lines.append(
                "       c) 用更多数据：--dataset_preset multi30k-quick → multi30k-tiny "
                "→ multi30k（越往右数据越多、越慢）；"
            )
        lines.append(
            "       d) 每组设置换 2~3 个 --seed 各跑一次，看差距是不是每次都朝同一个方向。"
        )
    return lines


def _explanation(
    runs: Sequence[Dict[str, Any]],
    log_dir: Path,
    out_dir: Path,
    png_path: Optional[Path],
    png_skip: Optional[str],
    font: Optional[str],
) -> List[str]:
    """The "说明" section: what the numbers mean and where the runs live."""
    lines = ["说明", "----"]
    lines.append("  - 指标含义：验证损失（val_loss）和测试损失（test loss）越低越好；BLEU 越高越好；")
    lines.append("    训练损失（train_loss）是训练集上的滑动平均，只作参考，不能和验证损失直接比大小。")
    lines.append(
        "  - 「最佳验证损失」取 training_log.csv 里 val_loss 的最小值；"
        "「测试损失/BLEU」来自 eval_test.json（用 evaluate 子命令生成），没有就显示 "
        f"{MISSING}。"
    )
    lines.append(
        f"  - 运行目录（每个运行的日志、配置和权重都在自己的子目录里）：本次扫描了 {log_dir}"
    )
    for run in runs:
        lines.append(f"      {run['name']}  ->  {run['run_dir']}")
    lines.append(f"  - 本次输出目录：{out_dir}")
    if png_path is not None:
        drawn_with = f"中文字体：{font}" if font else "英文字体"
        lines.append(f"      {PNG_FILENAME}：上面两张损失曲线（{drawn_with}）；")
    else:
        lines.append(f"      {PNG_FILENAME}：本次没有生成（{png_skip or '未请求绘图'}）；")
    lines.append(f"      {CSV_FILENAME}：上面那张表的原始数据，可以直接用 Excel 打开；")
    lines.append(f"      {TXT_FILENAME}：就是你现在看到的这份文字报告。")
    notes = [(run["name"], note) for run in runs for note in run["notes"]]
    if notes:
        lines.append("  - 提示（有这些运行缺少部分结果，不影响其它运行）：")
        lines.extend(f"      {name}：{note}" for name, note in notes)
    return lines


def _build_text(
    runs: Sequence[Dict[str, Any]],
    log_dir: Path,
    out_dir: Path,
    title: Optional[str],
    png_path: Optional[Path],
    png_skip: Optional[str],
    font: Optional[str],
) -> str:
    """Assemble the whole plain-language Chinese report."""
    rule = "=" * 78
    lines = [rule, title or DEFAULT_TITLE, rule, ""]

    n_runs = len(runs)
    ranked = [run for run in runs if run["val_series"]]
    if n_runs == 1:
        lines.append(
            f"本次比较：在 {log_dir} 找到 1 个已完成的训练运行，"
            f"它用了 {_bias_phrase(runs[0])}。"
        )
    else:
        lines.append(
            f"本次比较：在 {log_dir} 找到 {n_runs} 个已完成的训练运行，"
            "按最佳验证损失从低到高排列（越低越好）。"
        )
    if ranked and len(ranked) < n_runs:
        lines.append(
            f"其中 {len(ranked)} 个有可用的验证曲线，"
            f"{n_runs - len(ranked)} 个没有，排在表格最后。"
        )
    lines.append("")

    lines.append("运行对比表")
    lines.append("")
    lines.extend(_render_table(TABLE_HEADERS, _table_rows(runs, group=True)))
    lines.append("")
    lines.extend(_reading_guide(runs))
    lines.append("")
    lines.extend(_explanation(runs, log_dir, out_dir, png_path, png_skip, font))
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _no_run_report(
    log_dir: Path, out_dir: Path, title: Optional[str], message: str
) -> Dict[str, Any]:
    """Friendly, file-writing, never-raising answer for "there is nothing here"."""
    text = "\n".join(
        [
            "=" * 78,
            title or DEFAULT_TITLE,
            "=" * 78,
            "",
            message,
            "",
            "怎么看",
            "------",
            "  1. 这份报告只统计「已经跑完并写下 training_log.csv」的运行。",
            "  2. 先训练一次（例如）：",
            "       python main.py train --model smoke --dataset_preset synthetic-reverse "
            "--max_steps 60",
            "  3. 训练结束后，运行目录里会出现 training_log.csv，再执行本脚本就能看到对比。",
            "",
            "说明",
            "----",
            f"  - 本次查找的位置：{log_dir}",
            f"  - 输出目录：{out_dir}",
            "  - 每次运行都应该有自己的子目录，例如 "
            f"{log_dir}{os.sep}my-run{os.sep}{LOG_FILENAME}。",
            "",
        ]
    )
    _safe_print(text)
    csv_path: Optional[Path] = None
    txt_path: Optional[Path] = None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        txt_path = out_dir / TXT_FILENAME
        with txt_path.open("w", encoding="utf-8") as fh:
            fh.write(text)
        csv_path = out_dir / CSV_FILENAME
        with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
            csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS)).writeheader()
    except OSError:
        pass
    return {
        "ok": False,
        "message": message,
        "log_dir": str(log_dir),
        "out": str(out_dir),
        "n_runs": 0,
        "runs": [],
        "best_run": None,
        "second_run": None,
        "ranked_runs": [],
        "csv": str(csv_path) if csv_path else None,
        "txt": str(txt_path) if txt_path else None,
        "png": None,
        "png_skip_reason": "没有可比较的运行",
        "font": None,
        "glyph_warnings": [],
        "text": text,
    }


def build_report(
    log_dir: str = "runs",
    out: Optional[str] = None,
    make_plot: bool = True,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Compare every finished run under ``log_dir`` and write the report.

    Args:
        log_dir: directory whose immediate subdirectories are runs (each holding
            a ``training_log.csv``).
        out: where ``compare.png`` / ``compare.csv`` / ``compare.txt`` go;
            defaults to ``<log_dir>/_compare``.
        make_plot: draw ``compare.png`` (matplotlib is imported lazily).
        title: title used for the figure and the text report.

    Returns:
        A JSON-friendly summary dict: ``runs`` (one record per run), ``best_run``,
        ``n_runs``, ``text`` and the written paths.  Edge cases -- a missing
        ``log_dir``, no run at all, an empty log, a missing ``eval_test.json`` --
        produce a friendly Chinese message and a normal return, never a raise.
    """
    _configure_console_encoding()
    log_path = Path(log_dir)
    out_dir = Path(out) if out else log_path / OUT_DIRNAME

    if not log_path.exists():
        return _no_run_report(
            log_path,
            out_dir,
            title,
            f"找不到目录「{log_path}」：还没有任何训练结果，所以没有可以对比的运行。",
        )
    if not log_path.is_dir():
        return _no_run_report(
            log_path,
            out_dir,
            title,
            f"「{log_path}」是一个文件，不是放训练运行的目录，所以没有可以对比的运行。",
        )

    run_dirs = _find_runs(log_path)
    if not run_dirs:
        return _no_run_report(
            log_path,
            out_dir,
            title,
            f"目录「{log_path}」里没有找到任何运行"
            f"（判断标准：子目录里有 {LOG_FILENAME}），所以没有可以对比的运行。",
        )

    runs = [_load_run(run_dir) for run_dir in run_dirs]
    runs.sort(key=_sort_key)

    png_path: Optional[Path] = None
    glyph_warnings: List[str] = []
    font: Optional[str] = None
    png_skip: Optional[str] = None
    if make_plot:
        png_path, glyph_warnings, font, png_skip = _plot_compare(
            runs, out_dir / PNG_FILENAME, title
        )
    else:
        png_skip = "命令行带了 --no_plot"

    text = _build_text(runs, log_path, out_dir, title, png_path, png_skip, font)

    csv_path: Optional[Path] = None
    txt_path: Optional[Path] = None
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / CSV_FILENAME
        with csv_path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS))
            writer.writeheader()
            writer.writerows(_csv_rows(runs))
        txt_path = out_dir / TXT_FILENAME
        with txt_path.open("w", encoding="utf-8") as fh:
            fh.write(text)
    except OSError as exc:
        _safe_print(f"[输出] 写文件失败（{type(exc).__name__}: {exc}）；下面只打印报告内容。")

    _safe_print(text)

    ranked = [run for run in runs if run["val_series"]]
    best = runs[0] if runs and runs[0]["val_series"] else None
    return {
        "ok": True,
        "message": None,
        "log_dir": str(log_path),
        "out": str(out_dir),
        "n_runs": len(runs),
        "runs": runs,
        "ranked_runs": [run["name"] for run in ranked],
        "best_run": best["name"] if best else None,
        "best_val_loss": best["best_val_loss"] if best else None,
        "second_run": ranked[1]["name"] if len(ranked) > 1 else None,
        "csv": str(csv_path) if csv_path else None,
        "txt": str(txt_path) if txt_path else None,
        "png": str(png_path) if png_path else None,
        "png_skip_reason": png_skip,
        "font": font,
        "glyph_warnings": glyph_warnings,
        "text": text,
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """The command line: the four documented flags."""
    ap = argparse.ArgumentParser(
        description=(
            "Compare every finished run under a log directory: one figure, "
            "one CSV and one plain-language Chinese summary."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--log_dir",
        default="runs",
        help="directory holding one subdirectory per run (default: runs)",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=f"output directory (default: <log_dir>/{OUT_DIRNAME})",
    )
    ap.add_argument(
        "--no_plot",
        action="store_true",
        help="skip compare.png (compare.csv and compare.txt are still written)",
    )
    ap.add_argument(
        "--title",
        default=None,
        help="figure and report title (default: a built-in Chinese title)",
    )
    return ap


def main() -> int:
    """CLI entry point; always returns 0, even when there is nothing to report."""
    args = build_parser().parse_args()
    build_report(
        log_dir=args.log_dir,
        out=args.out,
        make_plot=not args.no_plot,
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
