# Symmetry-breaking encoder-decoder Transformer with an energy-conserving optimizer

## 快速上手（中文）

不想碰命令行、不想读代码的话：**双击 `run.bat`**，然后按提示输入数字就行。

```
双击 run.bat  →  输入 1（跑三种 bias 做对比）  →  输入 1（玩具任务，15 秒，不用联网）
```

跑完会自动生成对比图和一份中文报告：

- `runs\_compare\compare.png` —— 三种 bias 的验证损失曲线叠在一起
- `runs\_compare\compare.txt` —— 中文表格 + 「怎么看」的结论
- `runs\_compare\compare.csv` —— 同样的表格，可用 Excel 打开

想换设置再跑：再运行一次 `run.bat`，选 `2`（只跑一种 bias），用数字选「跑多久」
和「用哪种 bias」；或者直接编辑 `my_config.py`（每项都有中文注释），
然后选菜单 `3`。

**完整中文说明见 [`使用说明.md`](使用说明.md)**（包含常见问题、名词解释、
每种 bias 是什么意思、结果怎么看）。下面英文部分是这个项目的技术说明和完整参数表。

---

## Overview (English)

This package implements an **encoder-decoder Transformer** together with an
explicit, configurable **embedding-bias symmetry-breaking mechanism** and the
**EGD (Energy-conserving Gradient Descent)** optimizer, so that the effect of
deliberately breaking attention's rotational symmetry can be studied directly.
It includes the training code, two alternative optimizers for comparison
(AdamW, SGD with momentum), an evaluation script, an analysis script that
measures how much the injected bias actually changes what the model computes,
and a comparison report that turns a directory full of runs into one figure and
one table.

The layout, the configuration style (`config.py` presets + command-line flags +
`examples/*.sh`), the run-output conventions and the training-script shape follow
the companion project
[`evasilverstein/Symmetry-breaking-attention-bias`](https://github.com/evasilverstein/Symmetry-breaking-attention-bias),
which studies the same physics on a decoder-only GPT. AI coding assistance
contributed substantially to this package, including code generation and
facilitating testing and analysis.

### Two ways to drive it

| | for | how |
|---|---|---|
| **Foolproof** | running the bias comparison without touching code | double-click `run.bat`, or `python run.py`, and answer the numbered menu |
| **Scriptable** | scripting, sweeping, reproducing a published setting | `python main.py <command> [flags]`, or `examples/*.sh` |

`run.py` is a thin front-end: it asks a couple of questions, then invokes
`main.py` for you and shows the comparison report at the end. Anything it can do
you can also do by hand — see "Quick Start" below.

## Key Idea

Attention never sees the embedding `x` directly; it only ever sees it through the
learned projections `W_q`, `W_k`, `W_v`:

```
x = Embed(tokens) * sqrt(n_embd) + position
q, k, v = W_q x, W_k x, W_v x
scores  = q k^T / sqrt(head_dim)
```

With no offset the Q-K sector has an `O(n_embd)` rotational symmetry: for any
orthogonal `R`, a rotated embedding `R x` can be absorbed into the weights
(`W_q R^T` is just another matrix of the same shape), so the hypothesis class is
invariant and a whole orbit of parameter settings computes the same function.
Those redundant directions are conserved Noether currents, and they can limit the
chaotic exploration that Energy-conserving Descent relies on.

Adding a **fixed, unlearned** bias `b` breaks that symmetry:

```
R (x + b) = R x + R b  !=  R x + b        (unless R b = b, i.e. b ~ 0)
```

so `b` singles out a preferred direction in feature space that no rotation can
undo. This package exposes that knob three ways:

1. **The embedding bias `b`**, added once to the embedding (`--bias_mode`), with
   the three settings asked for: `zero` (`b = 0`, the symmetric control),
   `gaussian` (`b ~ N(mean, std^2)`), and `const` (`b = c` in every dimension).
2. **Reference-style per-head biases `bQ` / `bK` / `bV`** inside attention.
   `bQ` enters through the softmax and is therefore *exponentially* amplified;
   `bV` only passes through a linear map and has a power-law effect; `bK` is off
   by default because the key-independent part of a constant shift cancels in the
   softmax normalisation.
3. **Learnable variants** of either (`--bias_learnable`, `--attn_learnable`),
   which turn the fixed buffer into a trained `nn.Parameter`.

## Installation

```bash
pip install -r requirements.txt
```

There is no GPU dependency anywhere: the presets are sized for CPU.

### Included Optimizers

- **EGD** (`symbreak_transformer/optimizer.py`) — energy-conserving descent, a
  port of the companion project's `ECD_q1_scaled` (its initialisation
  double-counting bug is fixed, and its state round-trips faithfully through a
  checkpoint).
- **AdamW** — PyTorch's, as a strong conventional baseline.
- **SGDM** — SGD with momentum, as a cheap baseline.

The companion project's vendored SOAP optimizer is *not* included here; the
training script accepts `--optimizer egd|adamw|sgdm`.

## Quick Start

### The three bias modes (the experiment this repo exists for)

```bash
# control: b = 0, O(n_embd) is an exact symmetry
python main.py train --model small --dataset_preset multi30k-tiny \
    --bias_preset symmetric --name small-egd-symmetric

# b ~ N(0, 0.02^2), drawn once at initialisation
python main.py train --model small --dataset_preset multi30k-tiny \
    --bias_preset b-gaussian --name small-egd-bgaussian

# b = 1 in every dimension
python main.py train --model small --dataset_preset multi30k-tiny \
    --bias_preset b-const --name small-egd-bconst
```

The same runs can be reproduced from the archived flag sets in `examples/`:

```bash
bash examples/train_symmetric.sh
bash examples/train_b_gaussian.sh
bash examples/train_b_const.sh
```

(On Windows these need Git Bash, which ships with Git for Windows; `bash` is
usually at `C:\Program Files\Git\bin\bash.exe` and may not be on `PATH`. Every
example is a single `python` invocation, so it can also be run by copying the
flags onto `python main.py train` directly.)

### Reference-style per-head attention biases

```bash
python main.py train --model small --dataset_preset multi30k-tiny \
    --optimizer egd --egd_lr 1.0 --egd_eta 100 --egd_F0 -1.0 \
    --use_q_bias --use_v_bias --mean_Q 0.5 --std_Q 0.05 --mean_V 0.5 --std_V 0.05 \
    --name small-egd-bQbV
```

### AdamW baseline

```bash
python main.py train --model small --dataset_preset multi30k-tiny \
    --optimizer adamw --adam_lr 1e-3 --adam_wd 0.01 \
    --bias_preset symmetric --name small-adamw-symmetric
```

### Offline smoke test (no network, well under a minute)

```bash
python main.py train --model smoke --dataset_preset synthetic-reverse --max_steps 60
```

### Evaluation and bias analysis

```bash
# loss + corpus BLEU on a split
python main.py evaluate --ckpt runs/small-egd-bgaussian/model_best.pt \
    --split test --dataset_preset multi30k-tiny --show_samples 5

# how much does the injected bias actually change the model?
python main.py analyze-bias --ckpt runs/small-egd-bgaussian/model_best.pt

# compare the three modes side by side
python main.py analyze-bias \
    --ckpt runs/small-egd-symmetric/model_best.pt \
           runs/small-egd-bgaussian/model_best.pt \
           runs/small-egd-bconst/model_best.pt

# plot a finished run
python main.py plot --run runs/small-egd-bgaussian
```

### What is available

```bash
python main.py --help                # the four subcommands
python main.py train --list_models   # model / bias / dataset presets + parameter counts
```

## Training Output and Logging

Each training run saves to `runs/<run-name>/` with:

- `model_best.pt` — checkpoint at the best validation loss
- `model_final.pt` — final checkpoint
- `model_<update>.pt` — periodic checkpoints, when `--save_every_updates` is set
- `training_log.csv` — the training curves, header first and then appended
- `losses.csv` — an append-only `tag,loss` dump of every training step, handy for
  a quick loss-versus-step check or a scripted comparison
- `summary.json` — the run summary (steps, parameters, optimizer, best val loss)
- `config.json`, `bias.json`, `args.json` — the resolved configuration
- `training_log.meta.json` — run name, parameter count, config and bias, written
  when the run starts, so an interrupted run is still identifiable
- `tokenizer_src.json`, `tokenizer_tgt.json` — the exact vocabularies used
- `training_curve.png` — loss and bias-norm curves (skip with `--no_plot`)

`training_log.csv` columns:

```
step,epoch,train_loss,train_ppl,val_loss,val_ppl,elapsed_sec,sec_per_step,
tokens_per_sec,egd_iteration,egd_momentum_norm,egd_skipped,
embed_b_norm,bQ_norm,bK_norm,bV_norm,best_val_loss
```

Training rows carry `train_*`, evaluation rows carry `val_*` (both are appended
to the same file, so a row leaves the other block empty).

The four `*_norm` columns are the point: they let you watch the
symmetry-breaking term alongside the loss. To plot by hand:

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/my-run/training_log.csv")
plt.plot(df["step"], df["train_loss"], label="train")
plt.plot(df["step"], df["val_loss"], label="val")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend()
plt.savefig("training_curve.png")
```

`analyze-bias` writes `bias_analysis.json` next to each checkpoint and a combined
`bias_comparison.json` (+ `bias_comparison.png`) for the set.

### Resuming an interrupted run

A CPU run is long enough that losing it to an interruption hurts, so training can
pick up from a checkpoint:

```bash
python main.py train --model small --dataset_preset multi30k-tiny \
    --bias_preset b-gaussian --max_steps 4000 --log_dir runs --name small-egd-bgaussian
# ... interrupted or stopped at update N ...
python main.py train --model small --dataset_preset multi30k-tiny \
    --bias_preset b-gaussian --max_steps 4000 --log_dir runs --name small-egd-bgaussian \
    --resume runs/small-egd-bgaussian/model_final.pt
```

Both the model **and** the optimizer are restored, and EGD's own bookkeeping
(iteration, `F0`, the rescaled `lr`/`nu`, the momentum norm and its RNG state)
travels with it, so a resumed run continues the same trajectory rather than
restarting the dynamics. Resuming into the same `--log_dir`/`--name` appends to
the existing `training_log.csv` instead of truncating it. The flags you pass on
the resume command must match the original run (different model shape or
vocabulary means the checkpoint will not load).

## Package Structure

```
  .
  ├── run.bat                     # Windows: double-click this
  ├── run.py                      # the numbered menu behind run.bat
  ├── my_config.py                # optional control panel (Chinese comments)
  ├── 使用说明.md                  # plain-language Chinese guide
  ├── main.py                     # scriptable entry point: train / evaluate / analyze-bias / report / plot
  ├── symbreak_transformer/
  │   ├── config.py               # Seq2SeqConfig + PRESETS, BiasConfig + BiasPresets, DataConfig
  │   ├── optimizer.py            # EGD (energy-conserving descent), AdamW/SGDM builders
  │   ├── utils.py                # device, seeding, CSV logging, stall watchdog, optimizer reporting
  │   ├── evaluate.py             # validation loss, greedy decoding, corpus BLEU
  │   ├── bias.py                 # the embedding bias b and the per-head bQ / bK / bV
  │   ├── model/
  │   │   ├── embedding.py        # token + positional embedding (this is where x + b happens)
  │   │   ├── attention.py        # multi-head attention, self and cross, with optional biases
  │   │   ├── feedforward.py      # position-wise MLP (gelu / relu / prelu)
  │   │   ├── encoder.py          # encoder layer + encoder stack
  │   │   ├── decoder.py          # decoder layer + decoder stack
  │   │   └── seq2seq.py          # Seq2SeqTransformer tying encoder and decoder together
  │   └── data/
  │       ├── download.py         # parallel text: HuggingFace mirror / synthetic / local
  │       ├── tokenizer.py        # word and char tokenizers, vocabulary building
  │       └── dataset.py          # torch Dataset, padding collate, DataLoader factory
  ├── scripts/
  │   ├── train.py                # unified training script (all flags)
  │   ├── evaluate.py             # checkpoint scoring: loss + BLEU
  │   ├── analyze_bias.py         # inventory and measured effect of the bias
  │   ├── report.py               # every run -> one comparison figure + one Chinese table
  │   ├── plot_curve.py           # single-run training curve
  │   └── _common.py              # shared data flags / checkpoint loading for the scripts
  ├── examples/                   # archived flag sets, one per experiment
  ├── tests/                      # pytest suite
  └── requirements.txt
```

## Key Hyperparameter defaults, can be varied

### EGD Optimizer (`--optimizer egd`)

- `--egd_lr`: **1.0** (rescaled internally by `1/sqrt(eta)`)
- `--egd_eta`: 100 (concentration parameter)
- `--egd_F0`: **-1.0** — a loss offset that is guaranteed to sit below any
  cross-entropy loss
- `--egd_nu`: 0.0 (momentum noise amplitude)
- `--egd_consEn`: True (energy-conservation rescaling)

**`F0` must stay below the smallest loss the model can reach.** The step is only
taken while `loss - F0 > eps2`, so an `F0` above the reachable loss freezes
training without any error message. Cross-entropy is always `>= 0`, so `F0 = -1`
is always safe and the denominator `loss - F0 = loss + 1` can never collapse.
That is the default.

`--egd_auto_F0` is available for the companion project's convention
(`F0 = initial_loss - auto_F0_margin`), but **read the warning**: it stops
training as soon as the loss has improved by the margin. With the default margin
of `1.0` that truncates a run after a single nat — measured on this repo, val
loss pinned at `5.6905` from step 64 onward instead of continuing to `5.49`. It is
useful only when you set the margin larger than the total improvement you expect.

**`lr` and `F0` are coupled, so they were retuned together.** `lr` multiplies the
kick while `loss - F0` divides it; a smaller `F0` makes the denominator larger and
therefore needs a larger `lr`. An `lr` tuned against the old automatic `F0` is
roughly **ten times too small** once `F0 = -1`. Measured on `multi30k-quick`
(2 000 pairs, `n_embd=128`, `tiny`, 126 steps, `eta=100`, `F0=-1`, validation loss
at each checkpoint):

| `--egd_lr` | step 40 | step 80 | step 120 | best val loss | behaviour |
|---|---|---|---|---|---|
| 0.1 | 6.612 | 6.001 | 5.553 | 5.493 | stable but far too slow |
| 0.3 | 5.497 | 4.785 | 4.523 | 4.497 | stable |
| **1.0** (default) | 4.480 | 4.331 | 4.203 | 4.177 | stable, fast |
| 2.0 | 4.378 | 4.072 | 3.953 | **3.722** | stable, best here |
| 4.0 | 20.70 | 14.48 | 5.019 | 5.019 | early blow-up, then recovers |

So the usable window with `F0 = -1` is roughly **`0.3 - 2.0`**, and the default
`1.0` sits comfortably inside it (it is also the companion project's value). The
`4.0` row is not a bug: EGD is a Hamiltonian flow designed to explore and then
concentrate, so a large excursion that recovers is expected behaviour — but it
makes the run's outcome seed-dependent, so it is not a good default. If you change
`F0`, the model size or the dataset substantially, re-check `lr` the same way.

### Embedding Bias `b` (`--bias_mode`)

- `--bias_mode`: `zero` | `gaussian` | `const`
- `--bias_mean`: 0.0, `--bias_std`: 0.02 (gaussian)
- `--bias_const`: 1.0 (const)
- `--bias_resample`: `fixed` (drawn once) | `per_step` (redrawn every optimizer step)
- `--bias_learnable`: turn `b` into a trained `nn.Parameter`

### Attention Biases `bQ` / `bK` / `bV`

- `--use_q_bias`, `--use_k_bias` (off by default), `--use_v_bias`
- `--mean_Q`: 0.5, `--std_Q`: 0.05
- `--mean_V`: 0.5, `--std_V`: 0.05
- `--attn_mode`: `gaussian` (also `zero` / `const`), `--attn_resample`: `fixed` | `per_step`
- `--share_across_heads` (default on), `--no_share_across_layers` to give each layer its own
- `--attn_learnable`: train the biases instead of fixing them

### Model

- `--model`: `smoke` | `tiny` | `small` | `base` | `large` (see `--list_models`)
- `--ctx`: 128 context length (per preset), `--n_embd`, `--n_head`, `--n_encoder_layer`,
  `--n_decoder_layer`, `--d_ff`, `--dropout`
- `--activation`: `gelu` | `relu` | `prelu` (`--use_prelu`, `--prelu_random_init`)

## Architecture Modes

### Symmetric Mode (`--symmetric`, or `--bias_preset symmetric`)

- `b = 0` and every attention bias off, so `O(n_embd)` is an exact symmetry
- this is the control run every comparison needs

### Embedding-bias Mode

- `b` is added to the embedding, breaking the `O(n_embd)` rotational symmetry
- three settings: `zero` / `gaussian` / `const`, each optionally redrawn per step
- measured by `analyze-bias`, which runs the model with the bias and with every
  bias zeroed and reports `||dlogits|| / ||logits||` — the honest measure of
  whether the model actually uses the injected direction

### Attention-bias Mode

- `bQ` breaks the `O(head_dim)` symmetry of the Q-K sector, exponentially
  amplified through the softmax
- `bV` acts on the V-O sector with a power-law effect
- `bK` is available but off by default (it partly cancels in the softmax)

## Data

`--dataset` selects one of three providers:

| `--dataset` | behaviour |
|---|---|
| `hf` | downloads parallel text from a HuggingFace **mirror** (default `https://hf-mirror.com`), caching the raw file under `data/raw/` |
| `synthetic` | generates a deterministic toy task (`--synthetic_task copy/reverse/sort`) — no network at all |
| `local` | reads `data/local/<split>.tsv` |

The default task is `bentrevett/multi30k`, German → English image captions
(29 000 train pairs, ~4.6 MB), fields selected with `--src_field` / `--tgt_field`.

> **Why a mirror?** `huggingface.co` is unreachable from some networks (DNS
> returns a poisoned address on the development machine), so the pipeline talks to
> `hf-mirror.com` instead and sends a `User-Agent` header, without which the
> mirror answers `403`. Both are configurable (`--hf_endpoint`, `--hf_repo`).
> Nothing needs HuggingFace's `datasets` or `transformers` packages.
> If a download fails and `--no_fallback_to_synthetic` is not passed, the run
> warns and continues on the synthetic task instead of crashing.

## Tests

```bash
python -m pytest tests -q
```

142 tests, ~2.5 minutes on CPU. They cover the bias semantics (each mode,
resampling, learnable, per-sector independence), attention mask polarity and
padding invariance, causality, weight tying, the EGD dynamics (`F0` floor,
initialisation, checkpoint round trip), the tokenizer and collate contract,
hand-computed BLEU values, and a full CLI end-to-end run — including a guard that
every flag used in `examples/*.sh` exists in `scripts/train.py`.
