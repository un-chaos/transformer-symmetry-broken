"""Training loop.

The only unusual part is EGD: the energy-conserving step needs *both* the loss
value and ``param.grad`` at the current point, so it is driven through a
closure that zeroes gradients, re-runs the forward pass and back-propagates.
AdamW follows the ordinary ``closure() -> optimizer.step()`` order.  Both paths
go through the same loop, selected by
:func:`transformer_sym.optimizer.optimizer_requires_closure`.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from .config import ExperimentConfig, save_config
from .data.dataset import build_dataloaders
from .data.download import load_all_splits, load_parallel_data
from .data.tokenizer import PAD_ID, Tokenizer, build_tokenizers
from .evaluate import evaluate_bleu, evaluate_loss, plot_curve
from .model import Seq2SeqTransformer
from .optimizer import build_optimizer, optimizer_requires_closure
from .utils import (
    CSVLogger,
    StepWatchdog,
    Timer,
    configure_console_encoding,
    count_parameters,
    environment_report,
    format_seconds,
    resolve_device,
    seed_everything,
)

__all__ = ["train", "evaluate_checkpoint", "auto_run_name", "make_run_dir"]

LOG_FIELDS: List[str] = [
    "step",
    "epoch",
    "train_loss",
    "train_ppl",
    "val_loss",
    "val_ppl",
    "elapsed_sec",
    "sec_per_step",
    "tokens_per_sec",
    "egd_iteration",
    "egd_momentum_norm",
    "egd_skipped",
    "embed_b_norm",
    "bQ_norm",
    "bK_norm",
    "bV_norm",
    "best_val_loss",
]


# --------------------------------------------------------------------------- #
# Run bookkeeping
# --------------------------------------------------------------------------- #
def auto_run_name(cfg: ExperimentConfig) -> str:
    """Readable run name encoding model shape, both biases, optimizer and seed."""
    m, emb, attn = cfg.model, cfg.bias.embed, cfg.bias.attention

    emb_tag = f"emb{emb.mode}"
    if emb.resample == "per_step":
        emb_tag += "-rs"
    if emb.learnable:
        emb_tag += "-learn"

    if not attn.enabled or attn.mode == "zero":
        attn_tag = "attnoff"
    else:
        sectors = "".join(
            name
            for name, on in (
                ("Q", attn.q_enabled),
                ("K", attn.k_enabled),
                ("V", attn.v_enabled),
            )
            if on
        )
        attn_tag = f"attn{attn.mode}b{sectors or 'none'}"
        if attn.resample == "per_step":
            attn_tag += "-rs"
        if attn.learnable:
            attn_tag += "-learn"

    return (
        f"{cfg.train.optimizer}"
        f"-d{m.d_model}-e{m.n_encoder_layers}d{m.n_decoder_layers}h{m.n_heads}"
        f"-{emb_tag}-{attn_tag}-seed{cfg.train.seed}"
    )


def make_run_dir(cfg: ExperimentConfig) -> Tuple[Path, str]:
    """Create (or reuse) the output directory for this run.

    A finished run (one that already has ``checkpoint_final.pt``) is never
    overwritten: a timestamp suffix is appended instead, so results are not lost
    by re-running the same config.
    """
    name = cfg.train.run_name or auto_run_name(cfg)
    root = Path(cfg.train.save_dir)
    run_dir = root / name
    if (run_dir / "checkpoint_final.pt").exists():
        stamp = time.strftime("%Y%m%d-%H%M%S")
        name = f"{name}-{stamp}"
        run_dir = root / name
        print(f"[run] previous run found, writing to a new directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, name


def _bias_norms(report: Dict[str, Any]) -> Dict[str, float]:
    """Collapse ``model.bias_report()`` into the four CSV norm columns.

    The report names its entries ``embed.bias_norm`` and ``<site>.bQ_norm`` with
    one entry per attention site (``encoder``, ``decoder_self``,
    ``decoder_cross``, ...).  Keys are normalised (non-alphanumerics dropped,
    lower-cased) and each column takes the **largest** norm seen for that
    sector, so the CSV keeps exactly one column per bias no matter how many
    sites or layers carry it.
    """
    wanted = {
        "embed_b_norm": ("embedbiasnorm", "embedbnorm", "embednorm"),
        "bQ_norm": ("bqnorm",),
        "bK_norm": ("bknnorm",),
        "bV_norm": ("bvnorm",),
    }
    normalised: Dict[str, float] = {}
    for key, value in report.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue  # modes and other strings carry no norm
        normalised["".join(ch for ch in str(key).lower() if ch.isalnum())] = numeric

    out: Dict[str, float] = {}
    for field, candidates in wanted.items():
        matches = [
            value
            for key, value in normalised.items()
            if any(candidate in key for candidate in candidates)
        ]
        out[field] = max(matches) if matches else 0.0
    return out


def _egd_log_fields(optimizer) -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    if hasattr(optimizer, "stats"):
        try:
            stats = optimizer.stats() or {}
        except Exception:
            stats = {}
    return {
        "egd_iteration": stats.get("iteration", ""),
        "egd_momentum_norm": stats.get("momentum_norm", ""),
        "egd_skipped": stats.get("skipped_updates", ""),
    }


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer,
    cfg: ExperimentConfig,
    step: int,
    epoch: int,
    val_loss: float,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": cfg.to_dict(),
        "step": step,
        "epoch": epoch,
        "val_loss": val_loss,
        "environment": environment_report(),
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic-ish: never leave a half-written checkpoint
    return path


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(
    cfg: ExperimentConfig,
    resume: Optional[str] = None,
    plot: bool = True,
    test_bleu: bool = False,
) -> Dict[str, Any]:
    """Run one complete training experiment described by ``cfg``.

    Args:
        cfg: a validated :class:`~transformer_sym.config.ExperimentConfig`.
        resume: optional checkpoint path to continue from.
        plot: write ``training_curve.png`` into the run directory at the end.
        test_bleu: score the best checkpoint on the ``test`` split when done.

    Returns:
        A summary dict (also written to ``summary.json`` in the run directory).
    """
    cfg.validate()
    configure_console_encoding()  # corpus text is non-ASCII on a GBK console
    seed_everything(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    timer = Timer()

    run_dir, run_name = make_run_dir(cfg)
    print(cfg.describe())
    print(f"[run] name={run_name}\n[run] dir ={run_dir}\n[run] {environment_report()}")

    # ---------------- data ---------------- #
    print("[data] loading parallel text ...")
    splits_needed = ["train", "val"]
    pairs = load_all_splits(cfg.data, splits_needed)
    train_pairs = pairs.get("train") or []
    if not train_pairs:
        raise ValueError("the training split is empty; check data.source / data.data_dir")
    print(
        f"[data] source={cfg.data.source} train={len(train_pairs)} "
        f"val={len(pairs.get('val') or [])}"
    )

    src_tokenizer, tgt_tokenizer = build_tokenizers(cfg.data, pairs)
    print(
        f"[data] tokenizer={cfg.data.tokenizer} "
        f"src_vocab={src_tokenizer.vocab_size} tgt_vocab={tgt_tokenizer.vocab_size}"
    )

    # Persist tokenizers next to the checkpoints so a run can be reproduced.
    src_tokenizer.save(run_dir / "tokenizer_src.json")
    tgt_tokenizer.save(run_dir / "tokenizer_tgt.json")

    cfg.model.vocab_size_src = src_tokenizer.vocab_size
    cfg.model.vocab_size_tgt = tgt_tokenizer.vocab_size
    cfg.validate()
    save_config(cfg, run_dir / "config.yaml")

    loaders = build_dataloaders(cfg, src_tokenizer, tgt_tokenizer, splits=splits_needed)
    train_loader = loaders["train"]
    val_loader = loaders.get("val")

    # ---------------- model ---------------- #
    model = Seq2SeqTransformer(
        cfg.model,
        cfg.bias,
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        pad_id=PAD_ID,
        bos_id=Tokenizer.bos_id,
        eos_id=Tokenizer.eos_id,
    ).to(device)
    n_params = count_parameters(model)
    print(f"[model] {n_params:,} trainable parameters on {device}")

    optimizer = build_optimizer(model, cfg.train)
    needs_closure = optimizer_requires_closure(cfg.train.optimizer)
    print(f"[opt] {type(optimizer).__name__} (closure={needs_closure})")

    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as exc:
                print(f"[resume] could not restore optimizer state ({exc}); starting fresh")
        start_step = int(ckpt.get("step", 0))
        print(f"[resume] loaded {resume} at step {start_step}")

    # ---------------- loop setup ---------------- #
    steps_per_epoch = max(1, len(train_loader))
    total_steps = cfg.train.max_steps or (cfg.train.epochs * steps_per_epoch)
    if start_step >= total_steps:
        print(f"[warn] start_step {start_step} >= total_steps {total_steps}; nothing to do")

    per_step_bias = (
        cfg.bias.embed.mode != "zero" and cfg.bias.embed.resample == "per_step"
    ) or (
        cfg.bias.attention.enabled
        and cfg.bias.attention.mode != "zero"
        and cfg.bias.attention.resample == "per_step"
    )
    if per_step_bias:
        print("[bias] per_step resampling enabled: biases are redrawn every step")

    logger = CSVLogger(run_dir / "training_log.csv", LOG_FIELDS)
    logger.write_meta(
        {
            "run_name": run_name,
            "environment": environment_report(),
            "device": str(device),
            "parameters": n_params,
            "config": cfg.to_dict(),
        }
    )
    print(f"[log] {run_dir / 'training_log.csv'}")

    watchdog = StepWatchdog(cfg.train.stall_warn_seconds)
    scheduler_note = f"{total_steps} steps ({cfg.train.epochs} epoch(s) x {steps_per_epoch})"
    print(f"[train] {scheduler_note}")

    def closure(batch):
        optimizer.zero_grad(set_to_none=True)
        out = model(
            batch["src"].to(device),
            batch["tgt_in"].to(device),
            src_padding_mask=batch["src_padding_mask"].to(device),
            tgt_padding_mask=batch["tgt_padding_mask"].to(device),
            labels=batch["tgt_out"].to(device),
            label_smoothing=cfg.train.label_smoothing,
        )
        loss = out["loss"]
        if loss is None:
            raise RuntimeError("model returned no loss; labels were not wired through")
        loss.backward()
        if cfg.train.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        return loss

    history: List[Dict[str, Any]] = []
    best_val = float("inf")
    best_step = start_step
    model.train()
    step = start_step
    stop_reason = "completed"
    stop_training = False
    running_loss = 0.0
    running_tokens = 0
    window_start = time.time()
    epoch = 0

    try:
        # ``stop_training`` (not just ``break``) is required: the ``max_seconds``
        # guard fires inside the inner ``for`` loop, and breaking only that loop
        # would leave the outer ``while`` spinning without ever stepping again.
        while step < total_steps and not stop_training:
            for batch in train_loader:
                if step >= total_steps:
                    break
                if cfg.train.max_seconds > 0 and timer.elapsed > cfg.train.max_seconds:
                    stop_reason = (
                        f"stopped by train.max_seconds={cfg.train.max_seconds} "
                        f"after {format_seconds(timer.elapsed)}"
                    )
                    print(f"[stop] {stop_reason}")
                    stop_training = True
                    break

                if per_step_bias:
                    model.resample_biases()

                step_started = time.time()
                if needs_closure:
                    loss_tensor = optimizer.step(lambda: closure(batch))
                else:
                    loss_tensor = closure(batch)
                    optimizer.step()
                if loss_tensor is None:
                    raise RuntimeError("optimizer returned no loss")
                loss_value = float(loss_tensor.detach().item())
                dt = watchdog.tick(step)

                n_tokens = int((batch["tgt_out"] != PAD_ID).sum().item())
                running_loss += loss_value * max(n_tokens, 1)
                running_tokens += max(n_tokens, 1)

                step += 1

                log_now = cfg.train.log_every > 0 and step % cfg.train.log_every == 0
                if log_now:
                    window = max(time.time() - window_start, 1e-9)
                    mean_loss = running_loss / max(running_tokens, 1)
                    ppl = math.exp(min(mean_loss, 50.0))
                    tps = running_tokens / window
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "train_loss": mean_loss,
                        "train_ppl": ppl,
                        "elapsed_sec": timer.elapsed,
                        "sec_per_step": dt,
                        "tokens_per_sec": tps,
                    }
                    row.update(_bias_norms(model.bias_report()))
                    row.update(_egd_log_fields(optimizer))
                    row["best_val_loss"] = best_val if math.isfinite(best_val) else ""
                    logger.log(row)
                    history.append(row)
                    print(
                        f"[step {step:6d}/{total_steps}] loss {mean_loss:.4f} "
                        f"ppl {ppl:8.2f} | {tps:7.0f} tok/s | {timer.pretty}",
                        flush=True,
                    )
                    running_loss = 0.0
                    running_tokens = 0
                    window_start = time.time()

                do_eval = val_loader is not None and (
                    (cfg.train.eval_every > 0 and step % cfg.train.eval_every == 0)
                    or step == total_steps
                )
                if do_eval:
                    val_loss = evaluate_loss(
                        model,
                        val_loader,
                        device,
                        pad_id=PAD_ID,
                        max_batches=cfg.train.max_eval_batches,
                    )
                    if math.isfinite(val_loss) and val_loss < best_val:
                        best_val = val_loss
                        best_step = step
                        _save_checkpoint(
                            run_dir / "checkpoint_best.pt",
                            model,
                            optimizer,
                            cfg,
                            step,
                            epoch,
                            val_loss,
                        )
                        flag = " (best, saved)"
                    else:
                        flag = ""
                    val_ppl = (
                        math.exp(min(val_loss, 50.0)) if math.isfinite(val_loss) else float("nan")
                    )
                    row = {
                        "step": step,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                        "elapsed_sec": timer.elapsed,
                        "best_val_loss": best_val if math.isfinite(best_val) else "",
                    }
                    row.update(_bias_norms(model.bias_report()))
                    row.update(_egd_log_fields(optimizer))
                    logger.log(row)
                    print(
                        f"[step {step:6d}] val loss {val_loss:.4f} ppl {val_ppl:.2f}{flag} "
                        f"| {timer.pretty}",
                        flush=True,
                    )
                    model.train()

                    if cfg.train.bleu_every > 0 and step % cfg.train.bleu_every == 0:
                        bleu = evaluate_bleu(
                            model,
                            pairs["val"],
                            src_tokenizer,
                            tgt_tokenizer,
                            device,
                            max_len=cfg.model.max_seq_len,
                            max_samples=cfg.train.bleu_samples,
                        )
                        print(
                            f"[step {step:6d}] val BLEU {bleu['bleu']:.2f} "
                            f"(strict {bleu['bleu_strict']:.2f}) on {bleu['n_samples']} pairs"
                        )
                        model.train()

            epoch += 1
            if step >= total_steps:
                break
    except KeyboardInterrupt:
        stop_reason = "interrupted by user"
        print(f"[stop] {stop_reason}; saving what we have")

    # ---------------- wrap up ---------------- #
    if val_loader is not None and math.isinf(best_val):
        best_val = evaluate_loss(
            model, val_loader, device, pad_id=PAD_ID, max_batches=cfg.train.max_eval_batches
        )
        best_step = step

    _save_checkpoint(
        run_dir / "checkpoint_final.pt", model, optimizer, cfg, step, epoch, best_val
    )

    result: Dict[str, Any] = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "steps": step,
        "epochs": epoch,
        "parameters": n_params,
        "best_val_loss": best_val,
        "best_step": best_step,
        "elapsed_sec": timer.elapsed,
        "stop_reason": stop_reason,
        "device": str(device),
        "optimizer": type(optimizer).__name__,
        "bias_report": model.bias_report(),
        "csv": str(run_dir / "training_log.csv"),
    }
    if isinstance(model.bias_report(), dict):
        result.update(
            {
                "bias_embed_mode": cfg.bias.embed.mode,
                "bias_attention_mode": cfg.bias.attention.mode if cfg.bias.attention.enabled else "zero",
            }
        )

    if test_bleu:
        best_path = run_dir / "checkpoint_best.pt"
        use_path = best_path if best_path.exists() else run_dir / "checkpoint_final.pt"
        ckpt = torch.load(use_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        test_pairs = load_parallel_data(cfg.data, "test")
        bleu = evaluate_bleu(
            model,
            test_pairs,
            src_tokenizer,
            tgt_tokenizer,
            device,
            max_len=cfg.model.max_seq_len,
            max_samples=cfg.train.bleu_samples,
            show_samples=5,
        )
        result["test_bleu"] = {k: v for k, v in bleu.items() if k != "samples"}
        result["test_samples"] = bleu.get("samples", [])
        print(
            f"[test] BLEU {bleu['bleu']:.2f} (strict {bleu['bleu_strict']:.2f}) "
            f"on {bleu['n_samples']} pairs from {use_path.name}"
        )
        for sample in bleu.get("samples", [])[:5]:
            print(f"    src: {sample['src']}")
            print(f"    ref: {sample['ref']}")
            print(f"    hyp: {sample['hyp']}")

    with (run_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False, default=str)

    if plot:
        plot_curve(run_dir / "training_log.csv", title=run_name)

    if watchdog.warned:
        print(
            f"[stall-watchdog] {watchdog.warned} step(s) exceeded "
            f"{format_seconds(cfg.train.stall_warn_seconds)}; slowest step "
            f"{format_seconds(watchdog.max_seen)}"
        )
    print(
        f"[done] {stop_reason} | steps={step} | best val loss="
        f"{best_val if math.isfinite(best_val) else 'n/a'} | {format_seconds(timer.elapsed)}"
    )
    return result


# --------------------------------------------------------------------------- #
# Standalone evaluation of a checkpoint
# --------------------------------------------------------------------------- #
def evaluate_checkpoint(
    cfg: ExperimentConfig,
    checkpoint: str,
    split: str = "test",
    max_samples: int = 0,
    show_samples: int = 5,
) -> Dict[str, Any]:
    """Score a saved checkpoint on one split (loss + BLEU)."""
    cfg.validate()
    configure_console_encoding()
    device = resolve_device(cfg.train.device)
    seed_everything(cfg.train.seed)

    train_pairs = load_parallel_data(cfg.data, "train")
    eval_pairs = train_pairs if split == "train" else load_parallel_data(cfg.data, split)
    vocab_by_split = {"train": train_pairs}
    src_tokenizer, tgt_tokenizer = build_tokenizers(cfg.data, vocab_by_split)
    cfg.model.vocab_size_src = src_tokenizer.vocab_size
    cfg.model.vocab_size_tgt = tgt_tokenizer.vocab_size

    model = Seq2SeqTransformer(
        cfg.model,
        cfg.bias,
        src_vocab_size=src_tokenizer.vocab_size,
        tgt_vocab_size=tgt_tokenizer.vocab_size,
        pad_id=PAD_ID,
        bos_id=Tokenizer.bos_id,
        eos_id=Tokenizer.eos_id,
    ).to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    from .data.dataset import ParallelTextDataset, collate_batch
    from torch.utils.data import DataLoader

    dataset = ParallelTextDataset(
        eval_pairs,
        src_tokenizer,
        tgt_tokenizer,
        max_src_len=cfg.model.max_seq_len,
        max_tgt_len=cfg.model.max_seq_len,
    )
    loader = DataLoader(dataset, batch_size=cfg.train.batch_size, collate_fn=collate_batch)
    loss = evaluate_loss(model, loader, device, pad_id=PAD_ID)
    bleu = evaluate_bleu(
        model,
        eval_pairs,
        src_tokenizer,
        tgt_tokenizer,
        device,
        max_len=cfg.model.max_seq_len,
        max_samples=max_samples,
        show_samples=show_samples,
    )
    print(f"[eval] {checkpoint} on {split}: loss={loss:.4f} BLEU={bleu['bleu']:.2f}")
    return {"split": split, "loss": loss, "bleu": {k: v for k, v in bleu.items() if k != "samples"}, "samples": bleu.get("samples", [])}
