"""
FineWeb-Edu: the large pre-training corpus.

``HuggingFaceFW/fineweb-edu`` is a filtered crawl of English web text.  Its
``sample/10BT`` configuration is the ~10-billion-token sample (14 parquet shards,
~28.5 GB), which is the corpus the companion project calls ``finewebedu10B``.

Two things live here:

* :func:`download_fineweb` -- a **resumable** downloader.  The mirror on this
  machine drops connections regularly, and the shards are ~2 GB each, so every
  file is fetched with an HTTP range request that continues from whatever is
  already on disk instead of starting over.  Re-running it is always safe.
* :func:`iter_fineweb_texts` -- a streaming reader that walks the parquet shards
  row-group by row-group, so a 28 GB corpus never has to fit in memory.

The corpus is **monolingual English**, so it cannot be used for translation; it is
paired with the span-corruption objective in
:mod:`symbreak_transformer.data.denoising`, which turns raw text into
encoder/decoder pairs.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

__all__ = [
    "FINEWEB_REPO",
    "FINEWEB_CONFIG",
    "FINEWEB_ENDPOINT",
    "FINEWEB_USER_AGENT",
    "list_fineweb_files",
    "download_fineweb",
    "fineweb_local_files",
    "read_manifest",
    "iter_fineweb_texts",
]

#: The dataset repository and the ~10B-token configuration inside it.
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample/10BT"
#: huggingface.co is unreachable from here; the mirror is required, and it
#: answers 403 unless a User-Agent is sent.
FINEWEB_ENDPOINT = "https://hf-mirror.com"
FINEWEB_USER_AGENT = "Mozilla/5.0 (compatible; symbreak-transformer/1.0)"

_MANIFEST = "manifest.json"
_CHUNK = 262144


# --------------------------------------------------------------------------- #
# Remote listing
# --------------------------------------------------------------------------- #
def _request(url: str, user_agent: str, extra: Optional[Dict[str, str]] = None,
             timeout: float = 60.0):
    headers = {"User-Agent": user_agent}
    if extra:
        headers.update(extra)
    return urllib.request.Request(url, headers=headers)


def list_fineweb_files(
    repo: str = FINEWEB_REPO,
    config: str = FINEWEB_CONFIG,
    endpoint: str = FINEWEB_ENDPOINT,
    user_agent: str = FINEWEB_USER_AGENT,
    timeout: float = 60.0,
) -> List[Dict[str, object]]:
    """
    List the parquet shards of a configuration, with their sizes.

    Returns:
        ``[{"path": "sample/10BT/000_00000.parquet", "size": 2152837621}, ...]``
        sorted by path (which is also shard order).

    Raises:
        RuntimeError: if the listing cannot be fetched after retrying.
    """
    api = f"{endpoint}/api/datasets/{repo}/tree/main/{config}"
    last: Optional[Exception] = None
    for attempt in range(5):
        try:
            with urllib.request.urlopen(
                _request(api, user_agent, timeout=timeout), timeout=timeout
            ) as response:
                entries = json.loads(response.read())
            files = [
                {"path": entry["path"], "size": int(entry.get("size") or 0)}
                for entry in entries
                if entry.get("type") == "file" and entry["path"].endswith(".parquet")
            ]
            if not files:
                raise RuntimeError(f"no parquet files under {api}")
            return sorted(files, key=lambda item: str(item["path"]))
        except Exception as exc:  # noqa: BLE001 - retried below
            last = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"could not list {api}: {last}")


def _download_url(repo: str, path: str, endpoint: str) -> str:
    return f"{endpoint}/datasets/{repo}/resolve/main/{path}"


# --------------------------------------------------------------------------- #
# Resumable download
# --------------------------------------------------------------------------- #
def _download_one(
    url: str,
    target: Path,
    expected_size: int,
    user_agent: str,
    timeout: float,
    max_attempts: int,
    progress_every: float,
) -> Dict[str, object]:
    """Download ``url`` to ``target``, resuming a partial file. Retries on reset."""
    attempt = 0
    while True:
        have = target.stat().st_size if target.exists() else 0
        if expected_size and have >= expected_size:
            return {"file": target.name, "bytes": have, "status": "cached"}
        attempt += 1
        if attempt > max_attempts:
            return {"file": target.name, "bytes": have, "status": "failed"}

        headers = {"Range": f"bytes={have}-"} if have else {}
        mode = "ab" if have else "wb"
        started = time.time()
        done = have
        try:
            with urllib.request.urlopen(
                _request(url, user_agent, headers, timeout=timeout), timeout=timeout
            ) as response:
                # A server that ignores Range answers 200 with the whole body, so
                # the partial file must be discarded rather than appended to.
                if have and response.status == 200:
                    mode = "wb"
                    done = 0
                last_report = time.time()
                with target.open(mode) as handle:
                    while True:
                        chunk = response.read(_CHUNK)
                        if not chunk:
                            break
                        handle.write(chunk)
                        done += len(chunk)
                        now = time.time()
                        if progress_every and now - last_report >= progress_every:
                            rate = (done - have) / max(now - started, 1e-9) / 1e6
                            percent = 100.0 * done / expected_size if expected_size else 0.0
                            print(
                                f"    {target.name}: {done / 1e9:.2f}/"
                                f"{expected_size / 1e9:.2f} GB ({percent:.1f}%) "
                                f"{rate:.1f} MB/s",
                                flush=True,
                            )
                            last_report = now
            if expected_size and done < expected_size:
                # Short read: loop again and continue from where we stopped.
                continue
            return {"file": target.name, "bytes": done, "status": "downloaded"}
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            have_now = target.stat().st_size if target.exists() else 0
            print(
                f"    {target.name}: {type(exc).__name__} after "
                f"{have_now / 1e9:.2f} GB; retry {attempt}/{max_attempts}",
                flush=True,
            )
            if attempt >= max_attempts:
                return {"file": target.name, "bytes": have_now, "status": "failed"}
            time.sleep(min(30.0, 2.0 * attempt))


def read_manifest(dest_dir) -> Dict[str, object]:
    """Read the manifest written by :func:`download_fineweb` (``{}`` if absent)."""
    path = Path(dest_dir) / _MANIFEST
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt manifest is not fatal
        return {}


def download_fineweb(
    dest_dir,
    repo: str = FINEWEB_REPO,
    config: str = FINEWEB_CONFIG,
    endpoint: str = FINEWEB_ENDPOINT,
    user_agent: str = FINEWEB_USER_AGENT,
    max_files: Optional[int] = None,
    max_attempts: int = 8,
    timeout: float = 120.0,
    progress_every: float = 20.0,
) -> Dict[str, object]:
    """
    Download (or top up) the parquet shards of a FineWeb-Edu configuration.

    Resumable in three ways: an already-complete file is skipped, a partially
    downloaded file continues from its current size, and re-running after a
    failure just carries on.  A ``manifest.json`` next to the shards records what
    is present so the training code can find them without hitting the network.

    Args:
        dest_dir: where the shards go (``data/fineweb-edu/sample-10BT`` by default
            when called from the CLI).
        max_files: stop after this many shards (``None`` = all of them).
        max_attempts: per-file retry budget for the flaky mirror.
        progress_every: seconds between progress lines.

    Returns:
        The manifest dict: ``{"repo", "config", "endpoint", "files": [...],
        "complete": bool, "total_bytes": int}``.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    remote = list_fineweb_files(repo, config, endpoint, user_agent, timeout)
    everything = len(remote)
    if max_files:
        remote = remote[: int(max_files)]

    print(
        f"[fineweb] {repo} :: {config} -> {dest}\n"
        f"[fineweb] {len(remote)} of {everything} shard(s), "
        f"{sum(int(f['size']) for f in remote) / 1e9:.2f} GB",
        flush=True,
    )

    results = []
    for index, entry in enumerate(remote, start=1):
        path = str(entry["path"])
        size = int(entry["size"])
        target = dest / Path(path).name
        print(f"[fineweb] shard {index}/{len(remote)}: {target.name}", flush=True)
        outcome = _download_one(
            _download_url(repo, path, endpoint),
            target,
            size,
            user_agent,
            timeout,
            max_attempts,
            progress_every,
        )
        outcome["path"] = path
        outcome["expected_size"] = size
        results.append(outcome)
        print(
            f"[fineweb]   -> {outcome['status']} ({outcome['bytes'] / 1e9:.2f} GB)",
            flush=True,
        )

    complete = len(remote) == everything and all(
        int(item["bytes"]) >= int(item["expected_size"]) for item in results
    )
    manifest = {
        "repo": repo,
        "config": config,
        "endpoint": endpoint,
        "files": results,
        "total_bytes": sum(int(item["bytes"]) for item in results),
        "complete": bool(complete),
    }
    (dest / _MANIFEST).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[fineweb] done: {manifest['total_bytes'] / 1e9:.2f} GB present, "
        f"complete={complete}",
        flush=True,
    )
    return manifest


# --------------------------------------------------------------------------- #
# Local access
# --------------------------------------------------------------------------- #
def fineweb_local_files(dest_dir) -> List[Path]:
    """The local parquet shards, in shard order."""
    return sorted(Path(dest_dir).glob("*.parquet"))


def iter_fineweb_texts(
    dest_dir,
    column: str = "text",
    files: Optional[Sequence[Path]] = None,
    row_group_batch: int = 1000,
    limit: Optional[int] = None,
) -> Iterator[str]:
    """
    Stream documents out of the shards without loading them into memory.

    Iterates parquet **row groups** and yields the text column batch by batch, so
    peak memory is one row group rather than 28 GB.

    Args:
        dest_dir: directory holding the shards (from :func:`download_fineweb`).
        column: the text column (FineWeb-Edu uses ``text``).
        files: explicit shard list; defaults to every ``*.parquet`` in ``dest_dir``.
        row_group_batch: rows per internal batch.
        limit: stop after this many documents.

    Yields:
        One document (``str``) at a time.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise ImportError(
            "reading the FineWeb-Edu parquet shards needs pyarrow: "
            "pip install pyarrow"
        ) from exc

    shards = list(files) if files is not None else fineweb_local_files(dest_dir)
    if not shards:
        raise FileNotFoundError(
            f"no parquet shards under {dest_dir}; download the corpus first "
            f"with menu item 7) in run.py (or run.bat)"
        )

    emitted = 0
    for shard in shards:
        parquet = pq.ParquetFile(shard)
        if column not in parquet.schema_arrow.names:
            raise KeyError(
                f"{shard.name} has no column {column!r}; columns are "
                f"{parquet.schema_arrow.names}"
            )
        for batch in parquet.iter_batches(batch_size=row_group_batch, columns=[column]):
            for value in batch.column(0).to_pylist():
                if value is None:
                    continue
                yield value
                emitted += 1
                if limit is not None and emitted >= limit:
                    return
