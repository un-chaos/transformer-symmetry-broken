# Symmetry-breaking encoder-decoder Transformer with an energy-conserving optimizer

## 快速上手（中文）

**这个项目只有一种用法**：**双击 `run.bat`**（或者运行 `python run.py`，完全一样），
然后按提示输入数字。

```
双击 run.bat  →  输入 1（跑三种 bias 做对比）  →  回车（玩具任务，15 秒，不用联网）
```

跑完会自动生成对比图和一份中文报告：

- `runs\_compare\compare.png` —— 三种 bias 的验证损失曲线叠在一起
- `runs\_compare\compare.txt` —— 中文表格 + 「怎么看」的结论
- `runs\_compare\compare.csv` —— 同样的表格，可用 Excel 打开

**每次训练的结果都会存下来**，在 `runs\<实验名>\` 里，一定有这几样：

- `training_curve.png` —— 损失曲线图（训练 + 验证）
- `model_summary.txt` —— 模型说明书：结构、维度、参数量花在哪
- `training_log.csv` / `config.json` / `bias.json` / `args.json` / 模型权重

**想自由调初始参数**（模型大小、学习率、batch size、数据集……）：选菜单 `3`，
它会列出当前设置，输入编号就能改 —— **没有任何参数是写死的**，
`train.py` 支持多少参数（100 多项）就能改多少，选 `a` 看全部。
改过的项会被自动记住，不需要编辑任何文件。

**想用大规模语料**（FineWeb-Edu 10B，约 28.5 GB）：选菜单 `8` 下载
（断点续传，会另开一个窗口显示进度），然后在菜单 `3` 里把 `--dataset_preset`
改成 `fineweb-10b`（或先试 `fineweb-quick`）。

**没有第二个入口、没有要手改的配置文件、不需要懂代码。**

**完整中文说明见 [`使用说明.md`](使用说明.md)**（包含常见问题、名词解释、
每种 bias 是什么意思、结果怎么看、模型说明书怎么读、大数据怎么下）。
下面英文部分是这个项目的技术说明和完整参数表。

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

The layout, the configuration style (`config.py` presets + a flat set of named
parameters), the run-output conventions and the training-script shape follow the
companion project
[`evasilverstein/Symmetry-breaking-attention-bias`](https://github.com/evasilverstein/Symmetry-breaking-attention-bias),
which studies the same physics on a decoder-only GPT. AI coding assistance
contributed substantially to this package, including code generation and
facilitating testing and analysis.

### One way to drive it

There is exactly **one** user-facing entry point, and deliberately so:

```
double-click run.bat          # or: python run.py   (identical)
        ↓
numbered Chinese menu  →  answer with digits  →  results land in runs/
```

Every capability is a menu item: training (1, 2), free configuration of any
parameter (3, 4), re-reporting finished runs (5), measuring what the bias actually
does (6), the results folder (7), downloading the large corpus (8) and help (9). There is no second entry point —
no dispatcher script, no shell-script archive, no config file to hand-edit — and
`tests/test_cli.py::test_the_menu_is_the_only_root_entry_point` fails if one is
reintroduced.

Parameters are *not* hard-coded behind the menu: menu item 3) edits every flag
`scripts/train.py` accepts (~128 of them, listed straight from its argparse), and
remembers the edits in a generated `run_settings.json` the user never opens. The
internal scripts under `scripts/` are implementation detail, invoked by the menu
for isolation (a crash returns to the menu rather than killing it).

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

Everything below is done from the menu; the parameter names are the ones menu 3)
shows and lets you change.

### The three bias modes (the experiment this repo exists for)

Menu **1)** runs all three back to back with otherwise identical settings and then
draws the comparison — that is the whole experiment, and it is the reason the
foolproof path exists:

```
menu 1)  →  pick how long  →  enter
   b = 0            control: O(n_embd) is an exact symmetry
   b ~ N(0, 0.02^2) drawn once at initialisation
   b = 1            in every dimension
```

To run a single mode instead, use menu **2)** and pick the bias. Anyone who wants
a different model size or step budget changes it first in menu **3)**
(`--model`, `--dataset_preset`, `--max_steps`, …), then runs 1) or 2).

### Reference-style per-head attention biases

Menu **3)** → set `--bias_preset` to `attn-bQbV` (or `attn-full` for bQ+bK+bV).
The individual knobs (`--use_q_bias`, `--use_v_bias`, `--mean_Q`, `--std_Q`,
`--mean_V`, `--std_V`) are in the same editor under group "symmetry-breaking
bias", and `a)` on that screen lists all of them.

### AdamW baseline

Menu **3)** → `--optimizer` → `adamw`, plus `--adam_lr` / `--adam_wd`. Then run
menu 2) with `--bias_preset` = `symmetric` to get the conventional baseline the
EGD runs are compared against.

### Offline smoke test (no network, well under a minute)

Menu **1)** or **2)** → "玩具任务" (the toy task). It is synthetic, needs no
network and takes about 15 seconds per run.

### Evaluation and bias analysis

Both happen from the menu:

- after training, the menu scores the best checkpoint (test loss + corpus BLEU)
  and stores it as `eval_test.json` inside the run directory;
- menu **5)** re-scans every finished run and rewrites the comparison figure,
  table and Chinese report into `<log_dir>/_compare/`;
- menu **6)** measures the bias itself: it picks a finished run and re-runs the
  model with the injected bias zeroed, reporting `logit_rel_change` (how much the
  output moved) as `bias_analysis.json` inside that run directory. A larger number
  means the model actually uses the bias, which is the quantitative half of the
  "is the symmetry breaking doing anything?" question.

### What is available

Menu **3)** → `a)` prints every parameter `scripts/train.py` accepts, grouped,
with its default and its help text. Menu **9)** explains what each menu item does.
The preset tables (model sizes, bias presets, dataset presets and their parameter
counts) are printed by the training script's `--list_models`, which the menu does
not need to expose because the same names are offered as numbered choices in the
editor.

## Training Output and Logging

Each training run saves to `runs/<run-name>/` with:

- `model_best.pt` — checkpoint at the best validation loss
- `model_final.pt` — final checkpoint
- `model_<update>.pt` — periodic checkpoints, when `--save_every_updates` is set
- `training_log.csv` — the training curves, header first and then appended
- `losses.csv` — an append-only `tag,loss` dump of every training step, handy for
  a quick loss-versus-step check or a scripted comparison
- `summary.json` — the run summary (steps, parameters, optimizer, best val loss)
- `model_summary.txt` — the full structure/dimension/parameter report (see below)
- `config.json`, `bias.json`, `args.json` — the resolved configuration
- `training_log.meta.json` — run name, parameter count, config and bias, written
  when the run starts, so an interrupted run is still identifiable
- `tokenizer_src.json`, `tokenizer_tgt.json` — the exact vocabularies used
- `training_curve.png` — loss and bias-norm curves (skip with `--no_plot`)

### What did I just build? (`model_summary.txt`)

Every run prints a model report before training starts and writes the same text to
`runs/<run-name>/model_summary.txt`, so a result is interpretable long after the
console has scrolled away. It answers four questions in one screen: the
**architecture** (depth, width, heads, vocab, activation, dropout, the
symmetry-breaking setting), the **per-component parameter counts** with the
semantic shapes behind them, **where the parameters live** as a percentage
breakdown, and which tensors are **non-learned buffers** (this project's bias `b`
is a buffer: it is the object of study, not something training updates).

```
Architecture
  n_embd (d_model)     : 128
  n_head               : 4   (head_dim 32)
  layers               : encoder 2, decoder 2
  vocab                : source 400, target 400
  tie_output_embedding : True
  symmetry-breaking    : embedding b: mode=gaussian std=0.02 ...

Components (learned parameters)
  source embedding
      params        55,296  ( 7.2%)   dims   embed 400x128, pos 32x128
  encoder layer 0-1 (x2)
      params       264,960  (34.3%)   dims   each: q/k/v/o 128x128 (+4 bias), ff 128->256, ln 128 x2
  decoder layer 0-1 (x2)
      params       397,568  (51.4%)   dims   each: q/k/v/o 128x128 x2 (+8 bias), ff 128->256, ln 128 x3
  output projection
      params             0  ( 0.0%)   dims   (shared / tied)
  TOTAL              773,120  (100.0%)

Where the parameters live
  embeddings                  110,592    14.3%
  encoder                     264,960    34.3%
  decoder                     397,568    51.4%

Non-learned buffers: 128 values in 1 tensor(s) (these are NOT trained)
  src_embed.bias.b                128
```

Identical layers are merged into one `xN` row so the table stays readable at
depth, and a tied weight is counted once and reported as `(shared / tied)` with 0
own parameters, so the component column sums exactly to the total.

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
pick up from a checkpoint. Resume is a parameter like any other, so it is reached
from the menu: **3)** → `a)` (all parameters) → find `--resume` and enter the
checkpoint path, e.g. `runs/small-egd-bgaussian/model_final.pt`, then run 2).
Keep the other settings identical to the original run.

Both the model **and** the optimizer are restored, and EGD's own bookkeeping
(iteration, `F0`, the rescaled `lr`/`nu`, the momentum norm and its RNG state)
travels with it, so a resumed run continues the same trajectory rather than
restarting the dynamics. Resuming into the same log directory and run name appends
to the existing `training_log.csv` instead of truncating it. The settings must
match the original run (a different model shape or vocabulary means the checkpoint
will not load).

## Package Structure

```
  .
  ├── run.bat                     # Windows: double-click this
  ├── run.py                      # the numbered menu behind run.bat: the ONLY entry point
  ├── run_settings.json           # generated: what the menu remembers for you (gitignored)
  ├── 使用说明.md                  # plain-language Chinese guide
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
  ├── tests/                      # pytest suite
  └── requirements.txt
```

`scripts/` is internal: the menu spawns those scripts as subprocesses so that a
crash in a training run returns to the menu instead of killing it. Nothing there
is a documented way to use the project.

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

`--dataset` selects the **provider**, and `--dataset_preset` selects a named
provider-plus-size combination:

| `--dataset` | behaviour |
|---|---|
| `hf` | downloads parallel text from a HuggingFace **mirror** (default `https://hf-mirror.com`), caching the raw file under `data/raw/` |
| `synthetic` | generates a deterministic toy task (`--synthetic_task copy/reverse/sort`) — no network at all |
| `local` | reads `data/local/<split>.tsv` |
| `fineweb` | reads **FineWeb-Edu** parquet shards from `--fineweb_dir` (downloaded separately) |

Named presets: `synthetic-copy`, `synthetic-reverse`, `synthetic-sort`,
`multi30k-quick` (2 000 pairs), `multi30k-tiny` (8 000), `multi30k` (all 29 000),
`fineweb-quick`, `fineweb-10b`.

The default task is `bentrevett/multi30k`, German → English image captions
(29 000 train pairs, ~4.6 MB), fields selected with `--src_field` / `--tgt_field`.

> **Why a mirror?** `huggingface.co` is unreachable from some networks (DNS
> returns a poisoned address on the development machine), so the pipeline talks to
> `hf-mirror.com` instead and sends a `User-Agent` header, without which the
> mirror answers `403`. Both are configurable (`--hf_endpoint`, `--hf_repo`).
> Nothing needs HuggingFace's `datasets` or `transformers` packages.
> If a download fails and `--no_fallback_to_synthetic` is not passed, the run
> warns and continues on the synthetic task instead of crashing.

### Large-scale corpus: FineWeb-Edu 10B

`HuggingFaceFW/fineweb-edu`, config `sample/10BT` — roughly **10 B tokens in 14
parquet shards (~28.5 GB)**. Menu item **8)** fetches it through a separate,
**resumable** downloader in a **separate, visible console window**, so progress
(or a stall) is observable rather than hidden behind the menu; re-running it
continues where it stopped instead of starting over, and it first reports how much
is already on disk. The destination defaults to
`data/fineweb-edu/sample-10BT`.

FineWeb-Edu is **monolingual English**, so it cannot be used for translation. It
is used with the span-corruption (T5-style denoising) objective instead, which
needs no parallel data. Select it from the menu: item **3)** → set
`--dataset_preset` to `fineweb-quick` (a 20 000-document slice, for a first try)
or `fineweb-10b` (everything), optionally adjust `--bpe_vocab_size`,
`--tokenizer_train_documents`, `--max_documents` and bound the run with
`--max_steps` — then run 1) or 2). Choosing a `fineweb-*` preset selects the
denoising objective automatically, because that is the only objective the corpus
supports.

Under `--objective denoising` the encoder reads a document with a few spans
replaced by `<extra_id_k>` sentinels and the decoder writes those spans back, so
`src` and `tgt` share one BPE tokenizer and one vocabulary. The realized span
count follows `--noise_density` (fraction of tokens removed, default 0.15) and
`--mean_span_length`; `--tokenizer_train_documents` bounds how many documents the
BPE tokenizer is trained on, and `--max_documents` bounds the corpus actually
indexed. Validation documents are held out deterministically by `--val_every`.

> **This machine trains on CPU only.** 10 B tokens is not a CPU budget: use
> `--max_steps` (and `--max_documents`) to bound a run, and treat the corpus as
> the thing the pipeline is *capable* of consuming rather than something to
> exhaust. Building the shard index takes ~35 s over all 14 shards because only
> row-group metadata is read up front.

## Tests

```bash
python -m pytest tests -q
```

238 tests, ~3 minutes on CPU. They cover the bias semantics (each mode,
resampling, learnable, per-sector independence), attention mask polarity and
padding invariance, causality, weight tying, the EGD dynamics (`F0` floor,
initialisation, checkpoint round trip), the tokenizer and collate contract,
hand-computed BLEU values, the model-summary invariants (component rows sum to
the distinct-parameter total, tied tensors reported once), the FineWeb/denoising
path (BPE id layout, sentinel framing, span recovery, deterministic shuffling),
the menu itself (scripted input, graceful EOF, settings persistence, and a guard
that **no second entry point exists**), a guard that every `scripts/train.py`
option is reachable from the menu's parameter editor, and full end-to-end training
runs through the same scripts the menu spawns.
