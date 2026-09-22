#!/usr/bin/env python3
"""
Download a training corpus.

The default is the one requested for this project: **FineWeb-Edu `sample/10BT`**,
the ~10-billion-token English web corpus (14 parquet shards, ~28.5 GB), the same
corpus the companion project calls ``finewebedu10B``.

The download is resumable, which matters because the mirror on this machine drops
connections regularly and the shards are ~2 GB each.  Re-running the command
simply continues; nothing is downloaded twice and a partial file is never thrown
away.

Note that FineWeb-Edu is **monolingual English**, so it is used with the
span-corruption (denoising) objective rather than translation -- see
``--dataset_preset fineweb-10b`` in ``scripts/train.py``.

Usage Examples:
    # the full 10B-token sample (~28.5 GB, about 1.5 h at 6 MB/s)
    python scripts/download_data.py

    # only the first two shards (~4.3 GB), enough to try the pipeline
    python scripts/download_data.py --max_files 2

    # how much is already on disk?
    python scripts/download_data.py --status

    # a different corpus / mirror
    python scripts/download_data.py --repo HuggingFaceFW/fineweb-edu \
        --config sample/100BT --dest data/fineweb-edu/sample-100BT
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from symbreak_transformer.data.fineweb import (  # noqa: E402
    FINEWEB_CONFIG,
    FINEWEB_ENDPOINT,
    FINEWEB_REPO,
    download_fineweb,
    fineweb_local_files,
    list_fineweb_files,
    read_manifest,
)
from symbreak_transformer.utils import configure_console_encoding  # noqa: E402

#: Where the corpus presets in ``symbreak_transformer/config.py`` expect to find
#: their shards, keyed by dataset preset name.
PRESET_DIRS = {
    "fineweb-10b": "data/fineweb-edu/sample-10BT",
}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Download a training corpus (default: FineWeb-Edu 10B sample).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--repo", type=str, default=FINEWEB_REPO)
    ap.add_argument("--config", type=str, default=FINEWEB_CONFIG)
    ap.add_argument("--endpoint", type=str, default=FINEWEB_ENDPOINT)
    ap.add_argument(
        "--dest",
        type=str,
        default=None,
        help=f"target directory (default: {PRESET_DIRS['fineweb-10b']})",
    )
    ap.add_argument(
        "--max_files",
        type=int,
        default=None,
        help="download only the first N shards (useful for a quick trial)",
    )
    ap.add_argument("--max_attempts", type=int, default=8, help="retries per shard")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument(
        "--status",
        action="store_true",
        help="report what is on disk and what is still missing, then exit",
    )
    return ap


def show_status(args) -> int:
    dest = Path(args.dest or PRESET_DIRS.get("fineweb-10b", "data/fineweb-edu/sample-10BT"))
    manifest = read_manifest(dest)
    local = fineweb_local_files(dest)
    on_disk = sum(path.stat().st_size for path in local)
    print(f"[fineweb] directory : {dest.resolve()}")
    print(f"[fineweb] shards    : {len(local)} on disk")
    print(f"[fineweb] size      : {on_disk / 1e9:.2f} GB")
    if manifest:
        print(f"[fineweb] manifest  : complete={manifest.get('complete')}")
    try:
        remote = list_fineweb_files(args.repo, args.config, args.endpoint)
        expected = sum(int(item["size"]) for item in remote)
        print(f"[fineweb] remote    : {len(remote)} shards, {expected / 1e9:.2f} GB")
        missing = [
            item["path"]
            for item in remote
            if not (dest / Path(str(item["path"])).name).exists()
            or (dest / Path(str(item["path"])).name).stat().st_size
            < int(item["size"])
        ]
        print(f"[fineweb] missing   : {len(missing)} shard(s)")
        for name in missing[:5]:
            print(f"            {name}")
    except Exception as exc:  # noqa: BLE001 - offline status is still useful
        print(f"[fineweb] remote listing failed ({type(exc).__name__}); disk info only")
    return 0


def main() -> int:
    configure_console_encoding()
    args = build_parser().parse_args()

    if args.status:
        return show_status(args)

    dest = args.dest or PRESET_DIRS.get("fineweb-10b", "data/fineweb-edu/sample-10BT")
    manifest = download_fineweb(
        dest_dir=dest,
        repo=args.repo,
        config=args.config,
        endpoint=args.endpoint,
        max_files=args.max_files,
        max_attempts=args.max_attempts,
        timeout=args.timeout,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))
    if not manifest.get("complete"):
        print(
            "\n[fineweb] not everything is present yet. Re-run the same command to "
            "continue where it stopped (nothing is downloaded twice).",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
