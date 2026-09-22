"""
A human-readable report of a model's structure, dimensions and parameter counts.

Written for the (non-programmer) user of this repository, whose complaint was that
the training output did not say what the model actually *is*.  The training script
prints :func:`model_summary` at startup and writes it next to the checkpoints.

The single most important property is that the component rows **add up exactly**
to the total: :func:`component_rows` groups every distinct parameter tensor exactly
once (deduplicating shared and tied weights, which ``nn.Module.parameters()`` also
deduplicates), so ``sum(row["params"]) == parameter_totals()["total"]`` always
holds.  ``tests/test_model_summary.py`` asserts it for the awkward configurations.

Output is deliberately **ASCII only**: this machine's console is GBK, and
non-ASCII box-drawing characters raise ``UnicodeEncodeError`` when the output is
piped (which is exactly what happens under a test runner or a shell redirect).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .utils import count_parameters

__all__ = [
    "parameter_totals",
    "component_rows",
    "buffer_rows",
    "architecture_lines",
    "model_summary",
    "write_model_summary",
]

#: Parameters of one component shown before the rest are collapsed.
_MAX_SHAPES_PER_ROW = 5
#: Buffers that are structural noise rather than experiment settings.
_BORING_BUFFERS = ("causal_mask",)


# --------------------------------------------------------------------------- #
# Totals
# --------------------------------------------------------------------------- #
def parameter_totals(model: nn.Module) -> Dict[str, int]:
    """
    Counts for the whole model.

    ``total`` counts every **distinct** parameter tensor once, so the tie between
    the output projection and the target embedding is not double counted;
    ``buffers`` counts the non-learned tensors (which is where this project's
    symmetry-breaking bias lives unless it was made learnable).
    """
    seen: Dict[int, torch.nn.Parameter] = {}
    for parameter in model.parameters():
        seen.setdefault(id(parameter), parameter)
    total = sum(p.numel() for p in seen.values())
    trainable = sum(p.numel() for p in seen.values() if p.requires_grad)
    buffers = 0
    seen_buffers: Dict[int, torch.Tensor] = {}
    for buffer in model.buffers():
        seen_buffers.setdefault(id(buffer), buffer)
    buffers = sum(b.numel() for b in seen_buffers.values())
    return {
        "total": int(total),
        "trainable": int(trainable),
        "frozen": int(total - trainable),
        "buffers": int(buffers),
    }


# --------------------------------------------------------------------------- #
# Grouping
# --------------------------------------------------------------------------- #
def _component_of(name: str) -> Tuple[str, str]:
    """
    Map a dotted parameter name to ``(component, short_name)``.

    Examples::

        src_embed.wte.weight            -> ("source embedding", "wte.weight")
        encoder.layers.3.ff.c_fc.weight -> ("encoder layer 3", "ff.c_fc.weight")
        attn_bias_encoder.q.b           -> ("attention bias (encoder)", "q.b")
    """
    if name.startswith("encoder.layers."):
        parts = name.split(".")
        return f"encoder layer {parts[2]}", ".".join(parts[3:])
    if name.startswith("decoder.layers."):
        parts = name.split(".")
        return f"decoder layer {parts[2]}", ".".join(parts[3:])
    if name.startswith("src_embed."):
        return "source embedding", name[len("src_embed.") :]
    if name.startswith("tgt_embed."):
        return "target embedding", name[len("tgt_embed.") :]
    if name.startswith("encoder."):
        return "encoder (other)", name[len("encoder.") :]
    if name.startswith("decoder."):
        return "decoder (other)", name[len("decoder.") :]
    if name.startswith("output_proj."):
        return "output projection", name[len("output_proj.") :]
    if name.startswith("attn_bias_"):
        site = name.split(".")[0][len("attn_bias_") :]
        return f"attention bias ({site})", ".".join(name.split(".")[1:])
    head, _, rest = name.partition(".")
    return head, rest or head


def _ordered_components(model: nn.Module) -> List[str]:
    """Component names in a stable, readable order (not dict order).

    Two passes, because ``named_parameters()`` de-duplicates by *tensor*: with
    ``tie_output_embedding=True`` the output projection shares the target
    embedding weight and its name never shows up there.  Scanning submodules for
    tensors they own directly recovers it, so the tied component still gets a row
    (with 0 own parameters) instead of silently disappearing from the table.
    """
    order: List[str] = []
    for name, _ in model.named_parameters():
        component, _ = _component_of(name)
        if component not in order:
            order.append(component)

    seen = set(order)
    for module_name, module in model.named_modules():
        if not module_name:
            continue
        for short, _ in module.named_parameters(recurse=False):
            component, _ = _component_of(f"{module_name}.{short}")
            if component not in seen:
                seen.add(component)
                order.append(component)

    def sort_key(component: str) -> Tuple[int, int, str]:
        if component == "source embedding":
            return (0, 0, component)
        if component == "target embedding":
            return (0, 1, component)
        if component.startswith("encoder layer "):
            return (1, int(component.rsplit(" ", 1)[1]), component)
        if component == "encoder (other)":
            return (2, 0, component)
        if component.startswith("decoder layer "):
            return (3, int(component.rsplit(" ", 1)[1]), component)
        if component == "decoder (other)":
            return (4, 0, component)
        if component == "output projection":
            return (5, 0, component)
        if component.startswith("attention bias"):
            return (6, 0, component)
        return (7, 0, component)

    return sorted(order, key=sort_key)


def _shorten(short: str) -> str:
    """``ff.c_fc.weight`` -> ``ff.c_fc``; drop the redundant ``.weight``."""
    return short[: -len(".weight")] if short.endswith(".weight") else short


def _dims_for(entries: List[Tuple[str, torch.Size]]) -> str:
    """
    A compact, semantic shape summary for one component.

    Compact matters: a raw listing of every tensor is 200 characters wide, which
    squeezes the component name out of the table and makes the whole report
    unreadable (that was the first version's failure).  Shapes are rendered the way
    a person reads them -- ``ff 256->1024`` rather than ``1024x256`` -- and the
    q/k/v/o attention block collapses into one clause.
    """
    # Bucket by the *first* path segment (``self_attn``, ``ff``, ``ln1``, ``wte``),
    # keeping the rest of the name so the sub-module can be recognised: the
    # useful grouping is at that level, not at ``...q_proj.weight``.
    buckets: List[Tuple[str, List[Tuple[str, torch.Size]]]] = []
    for short, shape in entries:
        head, _, sub = short.partition(".")
        if not sub:
            head, sub = short, "value"
        for name, items in buckets:
            if name == head:
                items.append((sub, shape))
                break
        else:
            buckets.append((head, [(sub, shape)]))

    clauses: List[str] = []
    attention_clauses: List[Tuple[str, Any, int]] = []
    plain_clauses: List[str] = []
    for head, items in buckets:
        subs = {sub for sub, _ in items}
        shapes = {sub: tuple(shape) for sub, shape in items}

        def forward_dims(sub_name: str):
            """``[out, in]`` -> ``(in, out)`` for a linear weight."""
            shape = shapes.get(sub_name)
            if shape and len(shape) == 2:
                return (shape[1], shape[0])
            return None

        attention = [p for p in ("q_proj", "k_proj", "v_proj", "out_proj")
                     if any(sub.startswith(p) for sub in subs)]
        if attention:
            span = forward_dims(f"{attention[0]}.weight")
            label = "/".join(p[0] for p in attention)
            biases = sum(1 for sub in subs if sub.endswith(".bias"))
            attention_clauses.append((label, span, biases))
            continue

        if {"c_fc.weight", "c_proj.weight"} & subs:
            hidden = forward_dims("c_fc.weight")
            back = forward_dims("c_proj.weight")
            text = f"ff {hidden[0]}->{hidden[1]}" if hidden and back else "ff"
            if "act.weight" in subs:
                text += " + act"
            plain_clauses.append(text)
            continue

        if head == "wte" and "weight" in shapes:
            plain_clauses.append("embed " + "x".join(map(str, shapes["weight"])))
            continue
        if head == "wpe" and "weight" in shapes:
            plain_clauses.append("pos " + "x".join(map(str, shapes["weight"])))
            continue
        if head.startswith("ln") or "norm" in head.lower():
            size = next((sh[0] for sh in shapes.values() if len(sh) == 1), None)
            plain_clauses.append(f"ln {size}" if size else "ln")
            continue
        seen_shapes = []
        for sub, shape in items:
            rendered = "x".join(map(str, shape)) or "scalar"
            clause = f"{sub} {rendered}"
            if clause not in seen_shapes:
                seen_shapes.append(clause)
        plain_clauses.extend(seen_shapes[:2])

    # A decoder layer has two attention blocks with identical shapes; saying
    # "q/k/v/o 256x256 x2" beats printing the same clause twice.
    merged: List[Tuple[str, Any, int, int]] = []
    for label, span, biases in attention_clauses:
        for index, (seen_label, seen_span, seen_bias, count) in enumerate(merged):
            if seen_label == label and seen_span == span:
                merged[index] = (seen_label, seen_span, seen_bias + biases, count + 1)
                break
        else:
            merged.append((label, span, biases, 1))
    for label, span, biases, count in merged:
        text = f"{label} " + ("x".join(map(str, span)) if span else "?")
        if count > 1:
            text += f" x{count}"
        if biases:
            text += f" (+{biases} bias)"
        clauses.append(text)
    # Collapse repeated clauses too ("ln 256, ln 256" -> "ln 256 x2").
    counted: List[Tuple[str, int]] = []
    for clause in plain_clauses:
        for index, (seen, count) in enumerate(counted):
            if seen == clause:
                counted[index] = (seen, count + 1)
                break
        else:
            counted.append((clause, 1))
    clauses.extend(text + (f" x{count}" if count > 1 else "") for text, count in counted)

    if len(clauses) > _MAX_SHAPES_PER_ROW:
        clauses = clauses[:_MAX_SHAPES_PER_ROW] + [f"+{len(clauses) - _MAX_SHAPES_PER_ROW} more"]
    return ", ".join(clauses)


def component_rows(model: nn.Module) -> List[Dict[str, Any]]:
    """
    One row per model component.

    Each row is ``{"name", "dims", "params", "trainable", "share"}``.  Shared and
    tied tensors are attributed to the *first* component that owns them and the
    later component gets a ``params`` of 0 with a note, so the column sums exactly
    (see the module docstring).
    """
    totals = parameter_totals(model)
    total = max(totals["total"], 1)

    owners: Dict[str, int] = {}
    params_by_id: Dict[int, torch.nn.Parameter] = {}
    for name, parameter in model.named_parameters():
        component, _ = _component_of(name)
        # First occurrence wins: `parameters()` is also a set, and a tied or shared
        # weight is one physical tensor however many names point at it.
        if id(parameter) in params_by_id:
            continue
        params_by_id[id(parameter)] = parameter
        owners[name] = id(parameter)

    grouped: Dict[str, List[Tuple[str, torch.Size]]] = {c: [] for c in _ordered_components(model)}
    for name, parameter in model.named_parameters():
        if owners.get(name) != id(parameter):
            continue  # a duplicate name for an already-counted tensor
        component, short = _component_of(name)
        grouped.setdefault(component, []).append((short, parameter.shape))

    ties: Dict[str, str] = {}
    for component in _ordered_components(model):
        if grouped.get(component):
            continue
        ties[component] = ""

    rows: List[Dict[str, Any]] = []
    for component in _ordered_components(model):
        entries = grouped.get(component) or []
        rows.append(
            {
                "name": component,
                "dims": _dims_for(entries) if entries else "(shared / tied)",
                "params": sum(
                    p.numel() for name, p in model.named_parameters()
                    if owners.get(name) == id(p) and _component_of(name)[0] == component
                ),
                "trainable": sum(
                    p.numel() for name, p in model.named_parameters()
                    if owners.get(name) == id(p)
                    and _component_of(name)[0] == component
                    and p.requires_grad
                ),
                "share": 0.0,
            }
        )
    for row in rows:
        row["share"] = row["params"] / total
    return rows


def buffer_rows(model: nn.Module) -> List[Dict[str, Any]]:
    """
    The non-learned tensors, reported separately from the parameters.

    This matters here more than in most projects: the symmetry-breaking bias ``b``
    and the per-head ``bQ``/``bK``/``bV`` are buffers by default, so omitting them
    would hide the very thing the experiment is about.
    """
    seen: Dict[int, torch.Tensor] = {}
    rows: List[Dict[str, Any]] = []
    for name, buffer in model.named_buffers():
        if id(buffer) in seen:
            continue
        seen[id(buffer)] = buffer
        short = name.rsplit(".", 1)[-1]
        if any(token in name for token in _BORING_BUFFERS):
            continue
        rows.append(
            {
                "name": name,
                "dims": "x".join(str(d) for d in buffer.shape) or "scalar",
                "params": int(buffer.numel()),
                "trainable": 0,
                "share": 0.0,
                "short": short,
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# Architecture facts
# --------------------------------------------------------------------------- #
def architecture_lines(model: nn.Module) -> List[str]:
    """One line per architecture fact, plus the bias configuration."""
    cfg = getattr(model, "cfg", None)
    bias_cfg = getattr(model, "bias_cfg", None)
    lines: List[str] = []
    if cfg is not None:
        head_dim = cfg.n_embd // cfg.n_head if cfg.n_head else 0
        lines += [
            f"n_embd (d_model)     : {cfg.n_embd}",
            f"n_head               : {cfg.n_head}   (head_dim {head_dim})",
            f"layers               : encoder {cfg.n_encoder_layer}, "
            f"decoder {cfg.n_decoder_layer}",
            f"d_ff (feed-forward)  : {cfg.d_ff}",
            f"context_length       : {cfg.context_length}",
            f"vocab                : source {model.src_vocab_size}, "
            f"target {model.tgt_vocab_size}",
            f"dropout              : {cfg.dropout}  "
            f"attention_dropout {cfg.attention_dropout}",
            f"activation           : {cfg.activation}"
            + (f" (prelu_random_init={cfg.prelu_random_init}, "
               f"slope mean={cfg.prelu_slope_mean} std={cfg.prelu_slope_std})"
               if cfg.activation == "prelu" else ""),
            f"norm_first           : {cfg.norm_first}",
            f"init_std             : {cfg.init_std}"
            f"   scaled_residual_init {cfg.scaled_residual_init}",
            f"scale_embedding      : {cfg.scale_embedding}",
            f"tie_output_embedding : {cfg.tie_output_embedding}",
            f"share_embeddings     : {cfg.share_embeddings}",
        ]
    else:  # pragma: no cover - a model without a config still gets a report
        lines.append(f"n_embd (d_model)     : {getattr(model, 'n_embd', '?')}")
    if bias_cfg is not None:
        lines.append(f"symmetry-breaking    : {bias_cfg.describe()}")
    return lines


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _rule(width: int, char: str = "-") -> str:
    return char * width


def _table(headers: List[str], rows: List[List[str]], widths: List[int]) -> List[str]:
    """
    A left-aligned ASCII table with **fixed** column widths.

    Fixed widths (rather than "as wide as the widest cell") are what keeps the
    report inside ``width`` and stops one long cell from squeezing the others out.
    Numeric columns are right-aligned so they can be compared by eye.
    """
    numeric = {"params", "trainable", "share", "values"}
    out: List[str] = []
    cells = []
    for index, header in enumerate(headers):
        text = header
        if len(text) > widths[index]:
            text = text[: widths[index]]
        cells.append(text.rjust(widths[index]) if header in numeric else text.ljust(widths[index]))
    out.append("  ".join(cells).rstrip())
    out.append(_rule(min(sum(widths) + 2 * (len(widths) - 1), 200)))
    for row in rows:
        cells = []
        for index, cell in enumerate(row):
            text = str(cell)
            if len(text) > widths[index]:
                text = text[: max(1, widths[index] - 1)] + "."
            header = headers[index]
            cells.append(
                text.rjust(widths[index]) if header in numeric else text.ljust(widths[index])
            )
        out.append("  ".join(cells).rstrip())
    return out


def _merge_for_display(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse runs of identical consecutive layers into one row.

    A 6-layer model otherwise prints six identical lines, which hides the shape of
    the model instead of revealing it.  The merged row keeps the **summed** params
    so the table's total still matches the model.
    """
    merged: List[Dict[str, Any]] = []
    index = 0
    while index < len(rows):
        current = rows[index]
        base = current["name"].rsplit(" ", 1)
        if len(base) != 2 or not base[1].isdigit():
            merged.append(dict(current))
            index += 1
            continue
        stem = base[0]
        first = int(base[1])
        run = [current]
        cursor = index + 1
        while cursor < len(rows):
            nxt = rows[cursor]
            parts = nxt["name"].rsplit(" ", 1)
            if (
                len(parts) == 2
                and parts[0] == stem
                and parts[1].isdigit()
                and int(parts[1]) == first + len(run)
                and nxt["dims"] == current["dims"]
                and nxt["params"] == current["params"]
            ):
                run.append(nxt)
                cursor += 1
            else:
                break
        if len(run) == 1:
            merged.append(dict(current))
        else:
            merged.append(
                {
                    "name": f"{stem} {first}-{first + len(run) - 1} (x{len(run)})",
                    "dims": f"each: {current['dims']}",
                    "params": current["params"] * len(run),
                    "trainable": current["trainable"] * len(run),
                    "share": sum(item["share"] for item in run),
                }
            )
        index = cursor
    return merged


def model_summary(model: nn.Module, width: int = 78) -> str:
    """
    The whole report as plain text.

    Three blocks: the architecture facts, a component table with a total row, and a
    short "where the parameters live" breakdown.  No trailing blank line.
    """
    totals = parameter_totals(model)
    lines: List[str] = []
    lines.append(_rule(width, "="))
    lines.append("MODEL SUMMARY".center(width).rstrip())
    lines.append(_rule(width, "="))

    lines.append("Architecture")
    for line in architecture_lines(model):
        lines.append(f"  {line}")

    lines.append("")
    lines.append("Components (learned parameters)")
    rows = component_rows(model)
    display = _merge_for_display(rows)
    # Two lines per component instead of a table: a table forces a fixed column
    # width, and the dimensions column then truncates ("q/k/v/o 256x256, ff."),
    # which is exactly the information the report exists to convey.
    for row in display:
        lines.append(f"  {row['name']}")
        lines.append(
            f"      params {row['params']:>13,}  ({100.0 * row['share']:4.1f}%)"
            f"   trainable {row['trainable']:>13,}"
        )
        lines.append(f"      dims   {row['dims']}")
    lines.append(
        f"  {'TOTAL':<28} params {totals['total']:>13,}  (100.0%)"
        f"   trainable {totals['trainable']:>13,}"
    )

    lines.append("")
    lines.append("Where the parameters live")
    groups = {
        "embeddings": 0,
        "encoder": 0,
        "decoder": 0,
        "output projection": 0,
        "bias": 0,
    }
    for row in rows:
        name = row["name"]
        if "embedding" in name:
            groups["embeddings"] += row["params"]
        elif name.startswith("encoder"):
            groups["encoder"] += row["params"]
        elif name.startswith("decoder"):
            groups["decoder"] += row["params"]
        elif name == "output projection":
            groups["output projection"] += row["params"]
        elif "bias" in name:
            groups["bias"] += row["params"]
    total = max(totals["total"], 1)
    for label, value in groups.items():
        lines.append(f"  {label:<20} {value:>14,}   {100.0 * value / total:5.1f}%")

    buffers = buffer_rows(model)
    lines.append("")
    lines.append(
        f"Non-learned buffers: {totals['buffers']:,} values in {len(buffers)} tensor(s) "
        "(these are NOT trained)"
    )
    if buffers:
        buffer_table = [
            [row["name"], row["dims"], f"{row['params']:,}"] for row in buffers[:12]
        ]
        if len(buffers) > 12:
            buffer_table.append([f"+{len(buffers) - 12} more", "", ""])
        lines += _table(
            ["buffer", "shape", "values"],
            buffer_table,
            [30, 22, 10],
        )

    lines.append("")
    lines.append(
        f"Totals: {totals['total']:,} parameters "
        f"({totals['trainable']:,} trainable, {totals['frozen']:,} frozen)"
    )
    lines.append(_rule(width, "="))
    return "\n".join(lines)


def write_model_summary(model: nn.Module, path, width: int = 78) -> Path:
    """Write :func:`model_summary` to ``path`` (UTF-8) and return the path."""
    path = Path(path)
    path.write_text(model_summary(model, width=width), encoding="utf-8")
    return path
