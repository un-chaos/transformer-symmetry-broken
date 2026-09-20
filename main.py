#!/usr/bin/env python3
"""Main entry point for the symmetry-broken encoder-decoder Transformer.

Examples
--------
List the ready-made configs::

    python main.py --list-configs

Train with a preset (all network, bias and optimizer knobs live in the YAML)::

    python main.py --config configs/small_multi30k.yaml

The "bias b" experiment the project exists for -- three runs that differ only in
the embedding bias mode::

    python main.py --config configs/bias_zero.yaml
    python main.py --config configs/bias_embed_gaussian.yaml
    python main.py --config configs/bias_embed_const.yaml

Override any config field from the command line (values are parsed as YAML)::

    python main.py --config configs/small_multi30k.yaml \
        --set train.egd.lr=0.5 --set bias.embed.mode=gaussian \
        --set train.batch_size=32

Check the data pipeline without training, or score a checkpoint::

    python main.py --config configs/small_multi30k.yaml --data-only
    python main.py --config configs/small_multi30k.yaml --eval-only \
        --resume runs/<run-name>/checkpoint_best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the package importable when this file is run from anywhere.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformer_sym import __version__                      # noqa: E402
from transformer_sym.config import load_config               # noqa: E402
from transformer_sym.utils import (                          # noqa: E402
    configure_console_encoding,
    environment_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description=(
            "Encoder-decoder Transformer with a configurable symmetry-breaking "
            "embedding bias and an energy-conserving (EGD) optimizer."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to a YAML config (see configs/); required unless --list-configs",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config field, e.g. --set train.egd.lr=0.5 (repeatable)",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="override train.save_dir")
    parser.add_argument("--run-name", type=str, default=None, help="override train.run_name")
    parser.add_argument("--seed", type=int, default=None, help="override train.seed")
    parser.add_argument("--device", type=str, default=None, help="override train.device")
    parser.add_argument("--batch-size", type=int, default=None, help="override train.batch_size")
    parser.add_argument(
        "--max-steps", type=int, default=None, help="override train.max_steps (0 = from epochs)"
    )
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to continue from")
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="score --resume on a split instead of training",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="split used by --eval-only (default: test)"
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=0,
        help="limit the number of pairs scored by --eval-only (0 = all)",
    )
    parser.add_argument(
        "--bleu",
        action="store_true",
        help="after training, score the best checkpoint on the test split",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="only run the data pipeline (download, tokenize, show samples)",
    )
    parser.add_argument("--no-plot", action="store_true", help="do not write training_curve.png")
    parser.add_argument(
        "--list-configs", action="store_true", help="list available YAML configs and exit"
    )
    parser.add_argument("--version", action="version", version=f"transformer-sym {__version__}")
    return parser


def list_configs() -> int:
    config_dir = ROOT / "configs"
    files = sorted(config_dir.glob("*.yaml"))
    if not files:
        print(f"no configs found in {config_dir}")
        return 1
    print(f"configs in {config_dir}:")
    for path in files:
        print(f"  {path.relative_to(ROOT)}")
    return 0


def data_only(cfg) -> int:
    """Run the data pipeline and print a short sample -- a cheap sanity check."""
    from transformer_sym.data.download import load_all_splits
    from transformer_sym.data.tokenizer import build_tokenizers

    configure_console_encoding()  # samples contain non-ASCII German text
    print(cfg.describe())
    print(f"[env] {environment_report()}")
    pairs = load_all_splits(cfg.data, ("train", "val", "test"))
    for split, items in pairs.items():
        print(f"[data] {split}: {len(items)} pairs")
    src_tok, tgt_tok = build_tokenizers(cfg.data, pairs)
    print(
        f"[data] tokenizer={cfg.data.tokenizer} "
        f"src_vocab={src_tok.vocab_size} tgt_vocab={tgt_tok.vocab_size}"
    )
    for src, tgt in pairs["train"][:3]:
        ids = tgt_tok.encode(tgt, add_bos=True, add_eos=True, max_len=cfg.model.max_seq_len)
        print(f"  src : {src}")
        print(f"  tgt : {tgt}")
        print(f"  ids : {ids}")
        print(f"  back: {tgt_tok.decode(ids)}")
    return 0


def main(argv=None) -> int:
    configure_console_encoding()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_configs:
        return list_configs()
    if not args.config:
        parser.error("--config is required (or use --list-configs)")

    cfg = load_config(args.config, overrides=args.overrides)

    # Convenience overrides.
    if args.out_dir:
        cfg.train.save_dir = args.out_dir
    if args.run_name:
        cfg.train.run_name = args.run_name
    if args.seed is not None:
        cfg.train.seed = args.seed
    if args.device:
        cfg.train.device = args.device
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.max_steps is not None:
        cfg.train.max_steps = args.max_steps
    cfg.validate()

    if args.data_only:
        return data_only(cfg)

    if args.eval_only:
        if not args.resume:
            parser.error("--eval-only requires --resume <checkpoint>")
        from transformer_sym.train import evaluate_checkpoint

        result = evaluate_checkpoint(
            cfg,
            args.resume,
            split=args.split,
            max_samples=args.eval_samples,
        )
        print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
        return 0

    from transformer_sym.train import train

    result = train(cfg, resume=args.resume, plot=not args.no_plot, test_bleu=args.bleu)
    printable = {k: v for k, v in result.items() if k != "test_samples"}
    print(json.dumps(printable, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
