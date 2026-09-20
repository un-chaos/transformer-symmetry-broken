#!/usr/bin/env python3
"""
Single entry point for the symmetry-breaking Transformer project.

Every experiment is a subcommand; the actual implementations live in
``scripts/`` so each one can also be run directly
(``python scripts/train.py ...``).

Commands
--------
``train``         train an encoder-decoder Transformer
``evaluate``      score a checkpoint: validation/test loss and corpus BLEU
``analyze-bias``  measure how much the symmetry-breaking bias actually matters
``plot``          draw the training curve for a finished run

Usage Examples:
    # what is available?
    python main.py --help
    python main.py train --list_models

    # the three-mode bias experiment this repo exists for
    python main.py train --model small --bias_preset symmetric    --dataset_preset multi30k-tiny
    python main.py train --model small --bias_preset b-gaussian   --dataset_preset multi30k-tiny
    python main.py train --model small --bias_preset b-const      --dataset_preset multi30k-tiny

    # offline smoke test: no network, well under a minute
    python main.py train --model smoke --dataset_preset synthetic-reverse --max_steps 60

    # score and analyse a finished run
    python main.py evaluate     --ckpt runs/small-egd-bgaussian-seed42/model_best.pt --split test
    python main.py analyze-bias --ckpt runs/small-egd-bgaussian-seed42/model_best.pt
    python main.py plot         --run  runs/small-egd-bgaussian-seed42

Any arguments after the subcommand are passed straight through, so
``python main.py train --help`` shows the full training flag list.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbreak_transformer import __version__  # noqa: E402

#: subcommand -> script implementing it
COMMANDS = {
    "train": "scripts/train.py",
    "evaluate": "scripts/evaluate.py",
    "analyze-bias": "scripts/analyze_bias.py",
    "plot": "scripts/plot_curve.py",
}

_DESCRIPTIONS = {
    "train": "train an encoder-decoder Transformer",
    "evaluate": "score a checkpoint (loss + corpus BLEU)",
    "analyze-bias": "measure the effect of the symmetry-breaking bias",
    "plot": "draw the training curve of a finished run",
}


def print_help() -> None:
    print(__doc__.strip().split("Usage Examples:")[0].strip())
    print("\nusage: main.py <command> [options]\n\ncommands:")
    width = max(len(name) for name in COMMANDS)
    for name, description in _DESCRIPTIONS.items():
        print(f"  {name:<{width}}  {description}")
    print(f"\nRun 'main.py <command> --help' for the options of a command.")
    print(f"version: {__version__}")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    if not argv or argv[0] in ("-h", "--help"):
        print_help()
        return 0
    if argv[0] in ("-V", "--version"):
        print(f"symbreak-transformer {__version__}")
        return 0

    command = argv[0]
    if command not in COMMANDS:
        print(f"error: unknown command {command!r}", file=sys.stderr)
        print_help()
        return 2

    script = ROOT / COMMANDS[command]
    if not script.exists():
        print(f"error: {script} is missing", file=sys.stderr)
        return 2

    # Run the script as if it had been invoked directly, forwarding the
    # remaining arguments untouched.
    sys.argv = [str(script)] + argv[1:]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
