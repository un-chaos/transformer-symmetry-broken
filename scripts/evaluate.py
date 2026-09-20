#!/usr/bin/env python3
"""
Evaluate a trained checkpoint: validation/test loss and corpus BLEU.

The model shape and the symmetry-breaking settings are read straight out of the
checkpoint (its ``config`` / ``bias`` dicts), so only the data side has to be
supplied. The tokenizers saved next to the checkpoint by ``scripts/train.py``
are preferred; if they are missing the script rebuilds them from the data flags,
which reproduces the same vocabulary because tokenizer construction is
deterministic.

Usage Examples:
    # score the best checkpoint on the test split
    python scripts/evaluate.py --ckpt runs/small-egd-bgaussian-seed42/model_best.pt \
        --split test --dataset_preset multi30k-tiny

    # compare two runs on the same data
    python scripts/evaluate.py --ckpt runs/a/model_best.pt --split test
    python scripts/evaluate.py --ckpt runs/b/model_best.pt --split test

    # show a handful of greedy translations
    python scripts/evaluate.py --ckpt runs/a/model_best.pt --show_samples 5 --max_samples 200
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
# `main.py` dispatches through ``runpy.run_path``, which -- unlike running this
# file directly -- does not put this directory on ``sys.path``, so the
# script-local ``_common`` helper needs it added explicitly.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from _common import (  # noqa: E402  (script-local helper)
    add_data_args,
    build_data_config,
    build_model,
    load_checkpoint,
    load_tokenizers,
)
from symbreak_transformer.data import (  # noqa: E402
    PAD_ID,
    ParallelTextDataset,
    collate_batch,
    load_parallel_data,
)
from symbreak_transformer.evaluate import evaluate_bleu, evaluate_loss  # noqa: E402
from symbreak_transformer.utils import configure_console_encoding, resolve_device  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Evaluate a trained checkpoint (loss + corpus BLEU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--ckpt", type=str, required=True, help="checkpoint .pt path")
    ap.add_argument("--split", choices=["train", "val", "test"], default="test")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_samples", type=int, default=0, help="0 scores the whole split")
    ap.add_argument(
        "--show_samples", type=int, default=0, help="print this many src/ref/hyp triples"
    )
    ap.add_argument("--max_batches", type=int, default=0, help="0 uses the whole loader")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--out", type=str, default=None, help="JSON output path")
    add_data_args(ap)
    return ap


@torch.no_grad()
def main() -> None:
    configure_console_encoding()
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    ckpt_path = Path(args.ckpt)

    cfg, bias_cfg, ckpt = load_checkpoint(ckpt_path, device)
    print("=== Configuration ===")
    print(f"  checkpoint : {ckpt_path}")
    print(
        f"  model      : n_embd={cfg.n_embd} n_head={cfg.n_head} "
        f"enc/dec={cfg.n_encoder_layer}/{cfg.n_decoder_layer} d_ff={cfg.d_ff} "
        f"ctx={cfg.context_length}"
    )
    print(f"  bias       : {bias_cfg.describe()}")
    print(f"  trained    : update={ckpt.get('update')} epoch={ckpt.get('epoch')}")
    print(f"  device     : {device}")
    print("=====================")

    data_cfg = build_data_config(args)

    # The training split is always loaded: it is what the vocabulary is rebuilt
    # from when the saved tokenizers are missing.
    train_pairs = load_parallel_data(data_cfg, "train")
    split_pairs = (
        train_pairs if args.split == "train" else load_parallel_data(data_cfg, args.split)
    )
    if not split_pairs:
        raise SystemExit(f"the {args.split!r} split is empty")

    src_tokenizer, tgt_tokenizer = load_tokenizers(
        ckpt_path, data_cfg, train_pairs
    )
    print(
        f"[data] {args.split}: {len(split_pairs)} pairs | "
        f"src_vocab={src_tokenizer.vocab_size} tgt_vocab={tgt_tokenizer.vocab_size}"
    )

    model = build_model(
        cfg, bias_cfg, src_tokenizer.vocab_size, tgt_tokenizer.vocab_size, device
    )
    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = ParallelTextDataset(
        split_pairs,
        src_tokenizer,
        tgt_tokenizer,
        max_src_len=cfg.context_length,
        max_tgt_len=cfg.context_length,
        default_max_len=cfg.context_length,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_batch,
        num_workers=args.num_workers,
    )
    loss = evaluate_loss(
        model, loader, device, pad_id=PAD_ID, max_batches=args.max_batches
    )
    bleu = evaluate_bleu(
        model,
        split_pairs,
        src_tokenizer,
        tgt_tokenizer,
        device,
        max_len=cfg.context_length,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        show_samples=args.show_samples,
    )

    print("")
    print(f"=== {args.split} results ===")
    print(f"  loss        : {loss:.4f}")
    if math.isfinite(loss):
        print(f"  perplexity  : {math.exp(min(loss, 50.0)):.2f}")
    print(f"  BLEU        : {bleu['bleu']:.2f}  (strict {bleu['bleu_strict']:.2f})")
    if "sacrebleu" in bleu:
        print(f"  sacrebleu   : {bleu['sacrebleu']:.2f}")
    print(
        f"  precisions  : {[round(p, 2) for p in bleu['precisions']]}  "
        f"BP={bleu['bp']:.3f}  len_ratio={bleu['ratio']:.3f}"
    )
    print(f"  scored on   : {bleu['n_samples']} pairs")
    print("=====================")
    for sample in bleu.get("samples", []):
        print(f"  src: {sample['src']}")
        print(f"  ref: {sample['ref']}")
        print(f"  hyp: {sample['hyp']}")

    payload = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "update": ckpt.get("update"),
        "loss": loss,
        "bleu": {k: v for k, v in bleu.items() if k != "samples"},
        "bias": bias_cfg.to_dict(),
        "model": cfg.to_dict(),
    }
    out_path = (
        Path(args.out) if args.out else ckpt_path.parent / f"eval_{args.split}.json"
    )
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    print(f"[out] wrote {out_path}")


if __name__ == "__main__":
    main()
