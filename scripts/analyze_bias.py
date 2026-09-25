#!/usr/bin/env python3
"""
Analyse the symmetry-breaking bias of one or more trained checkpoints.

Two questions are answered, for every checkpoint:

1. **What is in there?**  The inventory: the embedding bias ``b`` (mode, norm,
   distribution, which dimensions carry it) and the per-head ``bQ``/``bK``/``bV``
   norms per attention site.

2. **Does it matter?**  The effect: the model is run on the same real batches
   twice, once as trained and once with every bias temporarily zeroed, and the
   two outputs are compared. ``logit_rel_change`` is
   ``||logits_with - logits_without|| / ||logits_without||`` and ``loss_delta``
   is ``loss_without - loss_with`` (positive means the bias is helping).  For a
   run whose embedding bias is exactly zero both numbers are exactly zero, which
   is the control case.

Point 2 is the honest way to report symmetry breaking: the norm of ``b`` says
how large the perturbation is, but only the output difference says whether the
model uses it.

Usage Examples:
    # one checkpoint
    python scripts/analyze_bias.py --ckpt runs/small-egd-bgaussian-seed42/model_best.pt

    # the three-mode comparison
    python scripts/analyze_bias.py \
        --ckpt runs/small-egd-symmetric-seed42/model_best.pt \
               runs/small-egd-bgaussian-seed42/model_best.pt \
               runs/small-egd-bconst-seed42/model_best.pt \
        --split val --batches 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# This script is invoked by ``run.py``'s menu (the project's single entry point),
# which imports this package rather than putting ``scripts/`` on ``sys.path``, so
# the script-local ``_common`` helper needs it added explicitly.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from _common import (  # noqa: E402  (script-local helper)
    add_data_args,
    build_data_config,
    build_model,
    finite_or_none,
    load_checkpoint,
    load_tokenizers,
    zeroed_biases,
)
from symbreak_transformer.data import (  # noqa: E402
    PAD_ID,
    ParallelTextDataset,
    collate_batch,
    load_parallel_data,
)
from symbreak_transformer.utils import configure_console_encoding, resolve_device  # noqa: E402


# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
def bias_inventory(model) -> dict:
    """Norms, distributions and locations of every live bias in the model."""
    inventory: dict = {"report": dict(model.bias_report())}

    embed = model.embed_biases()
    if embed:
        first = embed[0]
        b = first.b.detach().float().flatten()
        top = torch.topk(b.abs(), k=min(8, b.numel()))
        inventory["embed"] = {
            "mode": str(first.mode),
            "n_dims": int(b.numel()),
            "norm": float(b.norm()),
            "abs_mean": float(b.abs().mean()),
            "std": float(b.std()) if b.numel() > 1 else 0.0,
            "min": float(b.min()),
            "max": float(b.max()),
            "nonzero_fraction": float((b != 0).float().mean()),
            "is_exactly_zero": bool((b == 0).all()),
            "top_dims_by_abs": [
                [int(i), float(v)] for v, i in zip(top.values.tolist(), top.indices.tolist())
            ],
            "instances": len(embed),
        }
    else:
        inventory["embed"] = {"mode": "none", "norm": 0.0, "is_exactly_zero": True}

    sectors: dict = {}
    for index, attn in enumerate(model.attention_biases()):
        for letter, sector in attn.sectors().items():
            key = f"b{letter.upper()}" if index == 0 else f"b{letter.upper()}_{index + 1}"
            b = sector.b.detach().float()
            sectors[key] = {
                "mode": str(sector.mode),
                "shape": list(b.shape),
                "norm": float(b.norm()),
                "abs_mean": float(b.abs().mean()),
                "is_exactly_zero": bool((b == 0).all()),
            }
    inventory["attention_sectors"] = sectors
    return inventory


# --------------------------------------------------------------------------- #
# Effect
# --------------------------------------------------------------------------- #
@torch.no_grad()
def measure_bias_effect(
    model,
    loader,
    device: torch.device,
    batches: int = 4,
    pad_id: int = PAD_ID,
) -> dict:
    """
    Compare the model with its bias against the same model with every bias zeroed.

    Returns:
        Dict with ``logit_rel_change``, ``loss_with_bias``, ``loss_without_bias``,
        ``loss_delta`` (positive == the bias helps) and the counts used.
    """
    was_training = model.training
    model.eval()

    relative_changes = []
    loss_with = 0.0
    loss_without = 0.0
    tokens = 0
    used_batches = 0

    for index, batch in enumerate(loader):
        if batches and index >= batches:
            break
        src = batch["src"].to(device)
        tgt_in = batch["tgt_in"].to(device)
        tgt_out = batch["tgt_out"].to(device)
        src_mask = batch["src_padding_mask"].to(device)
        tgt_mask = batch["tgt_padding_mask"].to(device)

        out_with = model(
            src, tgt_in,
            src_padding_mask=src_mask,
            tgt_padding_mask=tgt_mask,
            labels=tgt_out,
        )
        with zeroed_biases(model):
            out_without = model(
                src, tgt_in,
                src_padding_mask=src_mask,
                tgt_padding_mask=tgt_mask,
                labels=tgt_out,
            )
            difference = (out_with["logits"] - out_without["logits"]).float().norm()
            reference = out_without["logits"].float().norm().clamp_min(1e-12)
            relative_changes.append(float(difference / reference))

        n_tokens = int((tgt_out != pad_id).sum().item())
        if n_tokens == 0 or out_with["loss"] is None or out_without["loss"] is None:
            continue
        loss_with += float(out_with["loss"]) * n_tokens
        loss_without += float(out_without["loss"]) * n_tokens
        tokens += n_tokens
        used_batches += 1

    if was_training:
        model.train()

    mean_rel = sum(relative_changes) / len(relative_changes) if relative_changes else 0.0
    lw = loss_with / tokens if tokens else float("nan")
    lo = loss_without / tokens if tokens else float("nan")
    return {
        "logit_rel_change": mean_rel,
        "loss_with_bias": finite_or_none(lw),
        "loss_without_bias": finite_or_none(lo),
        "loss_delta": finite_or_none(lo - lw) if tokens else None,
        "batches": used_batches,
        "tokens": tokens,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def short_name(ckpt_path: Path) -> str:
    """Human label for a checkpoint: its run directory name."""
    return ckpt_path.parent.name or ckpt_path.stem


def print_inventory(name: str, inventory: dict) -> None:
    embed = inventory["embed"]
    print(f"  embedding b : mode={embed['mode']} |b|={embed['norm']:.4f} "
          f"abs_mean={embed['abs_mean']:.4f} nonzero={embed['nonzero_fraction']:.2f} "
          f"exactly_zero={embed['is_exactly_zero']}")
    if inventory["attention_sectors"]:
        parts = [
            f"{key}={sector['norm']:.4f}"
            for key, sector in inventory["attention_sectors"].items()
        ]
        print(f"  attention   : " + "  ".join(parts))
    else:
        print("  attention   : none")


def comparison_table(rows: list) -> str:
    """Fixed-width table comparing several checkpoints."""
    header = (
        f"{'run':<34} {'bias':<22} {'|b|':>8} {'|bQ|':>8} {'|bV|':>8} "
        f"{'relchange':>10} {'loss_d':>9} {'val':>8}"
    )
    lines = [header, "-" * len(header)]
    for row in rows:
        sectors = row["inventory"]["attention_sectors"]
        bq = sectors.get("bQ", {}).get("norm", 0.0)
        bv = sectors.get("bV", {}).get("norm", 0.0)
        bias_desc = row["bias"].get("embed_mode", "?")
        if row["bias"].get("attention_enabled") or sectors:
            bias_desc += "+attn"
        val = row.get("val_loss")
        lines.append(
            f"{row['name']:<34} {bias_desc:<22} "
            f"{row['inventory']['embed']['norm']:>8.4f} {bq:>8.4f} {bv:>8.4f} "
            f"{row['effect']['logit_rel_change']:>10.3e} "
            f"{(row['effect']['loss_delta'] if row['effect']['loss_delta'] is not None else float('nan')):>9.4f} "
            f"{(val if val is not None else float('nan')):>8.4f}"
        )
    return "\n".join(lines)


def plot_comparison(rows: list, out_path: Path) -> bool:
    """Bar chart of the bias norms and of the measured effect."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[plot] skipped ({type(exc).__name__}: {exc})")
        return False

    names = [row["name"] for row in rows]
    positions = range(len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(7.0 * 2, 4.5))

    ax = axes[0]
    ax.bar([p - 0.22 for p in positions], [r["inventory"]["embed"]["norm"] for r in rows],
           width=0.22, label="|b| (embedding)")
    ax.bar(list(positions),
           [r["inventory"]["attention_sectors"].get("bQ", {}).get("norm", 0.0) for r in rows],
           width=0.22, label="|bQ|")
    ax.bar([p + 0.22 for p in positions],
           [r["inventory"]["attention_sectors"].get("bV", {}).get("norm", 0.0) for r in rows],
           width=0.22, label="|bV|")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("bias norm")
    ax.set_title("symmetry-breaking bias magnitude")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    ax = axes[1]
    effects = [r["effect"]["logit_rel_change"] for r in rows]
    ax.bar(list(positions), effects, width=0.4, color="tab:red")
    ax.set_xticks(list(positions))
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("||dlogits|| / ||logits||")
    ax.set_title("measured effect of the bias (bias vs b=0)")
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")
    return True


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Analyse the symmetry-breaking bias of trained checkpoints.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument(
        "--ckpt",
        type=str,
        nargs="+",
        required=True,
        help="one or more checkpoint paths (the run name is taken from the parent dir)",
    )
    ap.add_argument("--split", choices=["train", "val", "test"], default="val")
    ap.add_argument("--batches", type=int, default=4, help="batches used per checkpoint")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=2000, help="0 loads the whole split")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default=None, help="combined JSON output path")
    ap.add_argument("--no_plot", action="store_true")
    add_data_args(ap)
    return ap


def main() -> None:
    configure_console_encoding()
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    data_cfg = build_data_config(args)

    train_pairs = load_parallel_data(data_cfg, "train")
    split = "train" if args.split == "train" else args.split
    pairs = train_pairs if split == "train" else load_parallel_data(data_cfg, split)
    if not pairs:
        raise SystemExit(f"the {split!r} split is empty")
    if args.max_samples:
        pairs = pairs[: args.max_samples]

    rows = []
    for ckpt_arg in args.ckpt:
        ckpt_path = Path(ckpt_arg)
        name = short_name(ckpt_path)
        print("")
        print(f"=== {name} ===")
        print(f"  checkpoint : {ckpt_path}")

        cfg, bias_cfg, ckpt = load_checkpoint(ckpt_path, device)
        src_tok, tgt_tok = load_tokenizers(ckpt_path, data_cfg, train_pairs)
        model = build_model(cfg, bias_cfg, src_tok.vocab_size, tgt_tok.vocab_size, device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        print(f"  bias       : {bias_cfg.describe()}")

        inventory = bias_inventory(model)
        print_inventory(name, inventory)

        dataset = ParallelTextDataset(
            pairs,
            src_tok,
            tgt_tok,
            max_src_len=cfg.context_length,
            max_tgt_len=cfg.context_length,
            default_max_len=cfg.context_length,
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, collate_fn=collate_batch
        )
        effect = measure_bias_effect(model, loader, device, batches=args.batches)
        print(
            f"  effect     : logit_rel_change={effect['logit_rel_change']:.4e} "
            f"loss_with={effect['loss_with_bias']} "
            f"loss_without={effect['loss_without_bias']} "
            f"loss_delta={effect['loss_delta']}  "
            f"({effect['batches']} batches / {effect['tokens']} tokens)"
        )

        row = {
            "name": name,
            "checkpoint": str(ckpt_path),
            "update": ckpt.get("update"),
            "val_loss": finite_or_none(ckpt.get("val_loss")),
            "bias": bias_cfg.to_dict(),
            "model": cfg.to_dict(),
            "inventory": inventory,
            "effect": effect,
        }
        rows.append(row)

        per_ckpt = ckpt_path.parent / "bias_analysis.json"
        with per_ckpt.open("w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2, ensure_ascii=False, default=str)
        print(f"[out] wrote {per_ckpt}")

    print("")
    print("=== comparison ===")
    print(comparison_table(rows))

    out_path = Path(args.out) if args.out else Path("bias_comparison.json")
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[out] wrote {out_path}")

    if not args.no_plot:
        plot_path = out_path.with_name("bias_comparison.png")
        plot_comparison(rows, plot_path)


if __name__ == "__main__":
    main()
