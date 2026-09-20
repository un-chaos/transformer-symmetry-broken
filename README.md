# transformer-symmetry-broken

A small, readable **encoder-decoder Transformer** built to study one question:

> what happens to learning when you deliberately **break the rotation symmetry**
> of the embedding by adding a fixed bias `b`?

Every structure lives in its own file (encoder, decoder, attention, embeddings,
feed-forward, the optimizer, the data pipeline), everything is driven from a
single YAML config, and one main program (`main.py`) ties it together. The
optimizer is an **energy-conserving descent** method (EGD), with AdamW available
as a baseline for comparison.

This runs **CPU-only** — it is deliberately sized so that a full experiment
finishes in minutes, not hours.

---

## 1. The idea: `x -> x + b` breaks `O(d_model)`

Attention never sees the embedding `x` directly; it only ever sees it through the
learned linear maps `W_q`, `W_k`, `W_v`:

```
x = Embed(tokens) * sqrt(d_model) + pos_embed
q, k, v = W_q x, W_k x, W_v x
scores  = q kᵀ / sqrt(d_head)
```

If `b = 0`, a **global rotation** of the embedding space is a symmetry of the
model: for any orthogonal `R`, the rotated embedding `R x` can be absorbed into
the weights (`W_q Rᵀ` is just another matrix of the same shape), so the
hypothesis class is invariant under the action of `O(d_model)`. That symmetry is
a genuine constraint — and a plausible source of slow or degenerate training.

Now add a bias: `x -> x + b` with `b != 0` **fixed** (not learned). A rotation no
longer commutes with the offset,

```
R (x + b) = R x + R b   !=   R x + b        (unless R b = b, i.e. b ~ 0),
```

so the symmetry is *explicitly broken*. `b` picks a preferred direction in
feature space, and the model can no longer rotate its way out of it.

`b` is a **non-learned buffer** by default. It is part of the experiment
definition, not part of what the optimizer trains — so you can turn the symmetry
breaking on and off from a YAML file without touching a line of code.

### The three choices for `b`

`bias.embed.mode` is exactly the switch requested for these experiments:

| `mode` | what `b` is | symmetry |
|---|---|---|
| `zero` | `b = 0` (all `d_model` entries) | `O(d_model)` **preserved** — the control run |
| `gaussian` | `b ~ N(mean, std²)` element-wise over `d_model` | broken; a random preferred direction |
| `const` | `b = const_value` in every dimension | broken; a fully isotropic rank-one direction |

`bias.embed.resample` decides *when* `b` is (re)drawn:

* `fixed` (default) — drawn once at init, then held constant. The breaking is a
  fixed property of the parameterisation.
* `per_step` — redrawn on every optimizer step. The breaking becomes a source of
  stochastic perturbation in the update rule, which is a different experiment.

Set `bias.embed.learnable: true` to make `b` an `nn.Parameter` that the
optimizer trains (requires `resample: fixed`).

### Reference-style per-head biases `bQ` / `bK` / `bV`

The same machinery is also available *inside* attention, mirroring
[`Symmetry-breaking-attention-bias`](https://github.com/un-chaos): additive
per-head biases of shape `(n_heads, head_dim)`.

* `bQ` acts in the **Q-K sector**, whose rotation symmetry is `O(d_head)`
  (`Q -> RQ`, `K -> RK`). It enters through the softmax, so its effect is
  **exponentially amplified**.
* `bV` acts in the **V-O sector** (`V -> VR`, `O -> RᵀO`) and only passes through
  a linear map, so its effect is **power-law**.
* `bK` exists but is **off by default**: the part of a constant shift that does
  not depend on the key cancels in the softmax normalisation.

Turn them on with `bias.attention.enabled: true` plus a `mode` and per-sector
switches; see [`configs/attn_bias_bQ.yaml`](configs/attn_bias_bQ.yaml) and
[`configs/attn_bias_bQbV.yaml`](configs/attn_bias_bQbV.yaml).

---

## 2. Layout

One concept per file:

```
main.py                        # single entry point: train / eval / inspect data
transformer_sym/
  config.py                    # typed config objects + YAML load/save + validation
  bias.py                      # the embedding bias b  and  the per-head bQ/bK/bV
  embedding.py                 # token + positional embedding (this is where x+b happens)
  attention.py                 # hand-written multi-head attention (self & cross)
  feedforward.py               # position-wise MLP (gelu / relu / prelu)
  encoder.py                   # encoder layer + encoder stack
  decoder.py                   # decoder layer + decoder stack
  model.py                     # Seq2SeqTransformer tying encoder + decoder together
  optimizer.py                 # EGD (energy-conserving descent) + AdamW baseline
  train.py                     # the training loop (closure-driven for EGD)
  evaluate.py                  # validation loss, greedy decoding, corpus BLEU, curves
  utils.py                     # seeding, CSV logging, timing, stall watchdog
  data/
    download.py                # fetch parallel text (HuggingFace mirror) / synthetic / local
    tokenizer.py               # word & char tokenizer, vocabulary building, save/load
    dataset.py                 # torch Dataset, padding collate, DataLoader factory
configs/                       # 12 ready-to-run presets (see below)
tests/                         # pytest suite
```

---

## 3. Quick start

```bash
# 1. install (CPU wheels are enough; there is no GPU dependency anywhere)
pip install -r requirements.txt

# 2. list the presets
python main.py --list-configs

# 3. end-to-end smoke test: synthetic copy/reverse task, ~30 steps, well under a minute
python main.py --config configs/smoke_synthetic.yaml

# 4. the real thing: German -> English on Multi30k
python main.py --config configs/small_multi30k.yaml --bleu
```

Every knob can be overridden from the command line (values are parsed as YAML,
so `null`, `true` and `[0.9, 0.98]` all work):

```bash
python main.py --config configs/small_multi30k.yaml \
    --set bias.embed.mode=gaussian \
    --set bias.embed.std=0.05 \
    --set train.egd.lr=0.5 \
    --set train.batch_size=32
```

---

## 4. Data

`data.source` selects one of three providers:

| `source` | what it does |
|---|---|
| `hf` | downloads parallel text from a HuggingFace **mirror** (default `https://hf-mirror.com`), caches the raw file under `data/raw/` |
| `synthetic` | generates a deterministic toy task (`copy` / `reverse` / `sort`) — no network at all |
| `local` | reads `data/local/<split>.tsv` (or `.jsonl`) |

Default task: `bentrevett/multi30k`, German → English image captions
(29 000 train pairs, ~4.6 MB). Fields are selected with `src_field` / `tgt_field`.

> **Why a mirror?** On this machine `huggingface.co` is unreachable (DNS returns a
> poisoned address), so the pipeline talks to `hf-mirror.com` instead and sends a
> `User-Agent` header, without which the mirror answers `403`. Both are
> configurable: `data.hf_endpoint`, `data.hf_repo`, `data.hf_files`,
> `data.user_agent`. Nothing needs HuggingFace's `datasets` or `transformers`
> packages — the download is plain `urllib` + JSON Lines.
>
> If a download fails and `data.fallback_to_synthetic: true` (the default), the
> run prints a warning and continues on the synthetic task instead of crashing.

Check the pipeline without training:

```bash
python main.py --config configs/small_multi30k.yaml --data-only
```

Tokenization is a small in-repo word-level (or character-level) vocabulary:
`<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`. Word mode lowercases and splits with
`\w+|[^\w\s]`, keeps tokens with `count >= data.min_freq`, and caps the vocabulary
at `data.max_vocab` by descending frequency.

---

## 5. The optimizer: EGD

`train.optimizer: egd` uses an energy-conserving descent step: a Hamiltonian
flow on `(q, p)` where the "energy denominator" is the current loss offset from a
reference level `F0`, with the momenta renormalised every step so the dynamics
cannot run away. It needs **both** the loss value and the gradient at the current
point, which is why the training loop drives it through a closure.

```yaml
train:
  optimizer: egd
  egd:
    lr: 0.1              # internally rescaled by 1/sqrt(eta); see the sweep below
    eta: 100.0           # concentration parameter
    F0: null             # null -> F0 = initial_loss - auto_F0_margin
    auto_F0_margin: 1.0
    nu: 0.0              # momentum noise amplitude (0 = deterministic)
    eps1: 1.0e-10
    eps2: 1.0e-40
    weight_decay: 0.0
    consEn: true         # energy-conservation rescaling
```

**The one thing to get right is `F0`.** The step is only taken while
`loss - F0 > eps2`; if `F0` sits *above* the loss the model can reach, updates
stop and training silently freezes. Worse, as `loss -> F0+` the denominator
`loss - F0` collapses and the step explodes. So either leave `F0: null` (it is
then set to `initial_loss - auto_F0_margin`, which is always safe) or set a fixed
value that is safely below the smallest loss you expect. Cross-entropy is always
positive, so a fixed `F0: 1.0` is only safe if the loss will stay well above 1.0.
**`F0: null` is the default and the recommended setting.**

### `lr` is not the reference value — here is the measured sweep

The reference implementation ships `lr=1.0`, which it tuned on a 124M-parameter
GPT. On the small models here that diverges immediately, so do not copy it.

A 150-step sweep on Multi30k (4 000 train pairs, `d_model=128`, 2+2 layers,
target vocab 2 041 so `ln(V) = 7.62`, `eta=100`, `F0: null`, batch 64) measured
the following — each entry is the training loss at that step:

| `train.egd.lr` | step 25 | step 50 | step 100 | step 150 | val loss | behaviour |
|---|---|---|---|---|---|---|
| `0.05` | 7.454 | 7.281 | 6.847 | **6.268** | 6.300 | stable, steady |
| **`0.1`** (default) | 7.272 | 6.373 | 6.285 | 6.382 | 6.312 | stable, fastest early |
| `0.15` | 6.883 | 6.419 | 6.343 | 6.424 | 6.343 | stable |
| `0.2` | 36.69 | 28.50 | 7.485 | 6.244 | **5.822** | early excursion, then recovers |

A separate shorter sweep (30 steps, 2 000 pairs, vocab 1 303) probed the upper
end, where the dynamics stops being usable: `lr=0.3` went 6.94, 6.41, then
**2859.8** by step 20 and 2629 by step 30, and `lr=1.0` oscillated
11.4 -> 30.0 -> 16.4 -> 10.1 without ever settling.

So the usable window is roughly **`0.05 - 0.15`**; the default is `0.1`. Two
caveats worth knowing:

* The `0.2` row is not a bug. Energy-conserving descent is a *Hamiltonian flow*,
  not a monotone descent; it is designed to explore and then concentrate, so an
  early excursion that recovers is expected behaviour. It still reached the best
  validation loss of the whole sweep. Lower `lr` buys predictability, not
  necessarily a better optimum.
* The window is model- and task-dependent. If you change `d_model` or the
  dataset substantially, re-check `lr` the same way: run a few hundred steps at
  `0.05`, `0.1` and `0.2` and watch for a loss that climbs for more than a few
  dozen steps.

`eta` is the concentration parameter from the method and is left at `100`; note
`lr` is internally rescaled by `1/sqrt(eta)`, so `eta` also scales the effective
step. Prefer tuning `lr`.

Checkpoint resuming is faithful: `EGD.state_dict()` carries its scalars
(`iteration`, `F0`, `dim`, rescaled `lr`/`nu`, `_initialized`) and its RNG state,
so a resumed run continues the same trajectory instead of silently restarting the
dynamics.

For comparison runs, `train.optimizer: adamw` uses `torch.optim.AdamW` with
`train.adamw.*`. On the short runs here AdamW descends faster per step; EGD's
appeal is its asymptotic concentration behaviour, which needs longer runs.

---

## 6. Config reference

The loader rejects **unknown keys**, so a typo fails loudly instead of silently
changing your experiment.

### `model`
| key | default | meaning |
|---|---|---|
| `d_model` | 256 | embedding / hidden width (must divide by `n_heads`) |
| `n_encoder_layers`, `n_decoder_layers` | 3, 3 | stack depths |
| `n_heads` | 8 | attention heads |
| `d_ff` | 1024 | feed-forward hidden width |
| `dropout` | 0.1 | residual/dropout rate |
| `attention_dropout` | 0.0 | dropout on attention probabilities |
| `max_seq_len` | 128 | positional table size and length cap |
| `activation` | `gelu` | `gelu` / `relu` / `prelu` |
| `prelu_random_init` | `false` | randomise per-feature PReLU slopes (asymmetric MLP) |
| `norm_first` | `false` | pre-LN instead of post-LN |
| `scale_embedding` | `true` | multiply embeddings by `sqrt(d_model)` |
| `tie_output_embedding` | `true` | tie the output projection to the target embedding |
| `share_embeddings` | `false` | one embedding matrix for both sides |
| `init_std` | 0.02 | weight init std |
| `scaled_residual_init` | `true` | nanoGPT-style `(2 * n_layers)^-0.5` residual scaling |

### `bias.embed` — the bias `b` on `x`
| key | default | meaning |
|---|---|---|
| `mode` | `zero` | **`zero` / `gaussian` / `const`** |
| `resample` | `fixed` | `fixed` (once at init) / `per_step` (every step) |
| `mean`, `std` | 0.0, 0.02 | Gaussian parameters (scalar, or a list of length `d_model`) |
| `const_value` | 1.0 | value used by `const` mode |
| `learnable` | `false` | make `b` a trained `nn.Parameter` (requires `resample: fixed`) |
| `seed` | 1234 | dedicated RNG seed, so bias draws never disturb global seeding |

### `bias.attention` — per-head `bQ` / `bK` / `bV`
| key | default | meaning |
|---|---|---|
| `enabled` | `false` | master switch |
| `mode` | `gaussian` | `zero` / `gaussian` / `const` |
| `resample` | `fixed` | `fixed` / `per_step` |
| `q_enabled`, `k_enabled`, `v_enabled` | `true`, `false`, `true` | per-sector switches (`bK` off: it partly cancels in the softmax) |
| `q_mean`/`q_std`, `k_mean`/`k_std`, `v_mean`/`v_std` | 0.5/0.05, 0.3/0.1, 0.5/0.05 | per-sector distributions |
| `share_across_heads` | `true` | one `head_dim` vector expanded to all heads. With `learnable: true` this ties only the **initial** values — the per-head copies are separate parameters and then train independently |
| `share_across_layers` | `true` | one bias object reused by every layer |
| `apply_encoder`, `apply_decoder_self`, `apply_decoder_cross` | all `true` | which attention sites get a bias |
| `learnable` | `false` | trainable biases instead of fixed buffers |
| `const_value`, `seed` | 1.0, 1234 | as above |

### `data`, `train`
| key | default | meaning |
|---|---|---|
| `data.source` | `hf` | `hf` / `synthetic` / `local` |
| `data.hf_repo` / `hf_endpoint` | `bentrevett/multi30k` / `https://hf-mirror.com` | remote source |
| `data.src_field` / `tgt_field` | `de` / `en` | JSON keys |
| `data.max_train_samples` / `max_val_samples` | 0 | 0 = no limit |
| `data.max_vocab`, `min_freq`, `lowercase`, `tokenizer` | 30000, 2, `true`, `word` | vocabulary building |
| `data.fallback_to_synthetic` | `true` | degrade gracefully when offline |
| `train.batch_size`, `epochs` | 32, 10 | schedule |
| `train.max_steps` | 0 | 0 = `epochs × steps_per_epoch` |
| `train.log_every`, `eval_every` | 50, 200 | logging cadence |
| `train.bleu_every`, `bleu_samples` | 0, 200 | 0 disables periodic BLEU |
| `train.save_dir`, `run_name`, `seed`, `device` | `runs`, auto, 42, `auto` | run identity |
| `train.grad_clip` | 0.0 | 0 disables (EGD normalises its own step) |
| `train.label_smoothing` | 0.0 | passed to cross-entropy |
| `train.max_seconds` | 0.0 | hard wall-clock stop |
| `train.stall_warn_seconds` | 300.0 | warn when a single step exceeds this |

---

## 7. Presets

| config | purpose |
|---|---|
| `smoke_synthetic.yaml` | 30-step end-to-end verification on CPU, no network |
| `tiny_multi30k.yaml` | smallest real translation run (~8k train pairs) |
| `small_multi30k.yaml` | the recommended starting point |
| `base_multi30k.yaml` | largest CPU-feasible preset |
| `base_adamw_multi30k.yaml` | same model/data, AdamW baseline |
| `bias_zero.yaml` | **control**: `b = 0`, attention biases off |
| `bias_embed_gaussian.yaml` | `b ~ N(0, 0.02²)`, drawn once |
| `bias_embed_gaussian_per_step.yaml` | `b` redrawn every step |
| `bias_embed_const.yaml` | `b = 1` in every dimension |
| `attn_bias_bQ.yaml` | per-head `bQ` only (Q-K sector) |
| `attn_bias_bQbV.yaml` | `bQ` + `bV` (Q-K and V-O) |
| `attn_bias_full.yaml` | all three sectors, **learnable** biases |

A minimal bias comparison (identical in every other respect):

```bash
for c in bias_zero bias_embed_gaussian bias_embed_const; do
  python main.py --config configs/$c.yaml
done
```

---

## 8. Outputs

Each run writes a self-contained directory under `runs/<run-name>/`:

```
config.yaml             # the fully resolved config (defaults included)
tokenizer_src.json      # the exact vocabularies used
tokenizer_tgt.json
training_log.csv        # step-indexed metrics
training_log.meta.json  # environment + config
checkpoint_best.pt      # best validation loss
checkpoint_final.pt
summary.json            # final summary incl. bias norms
training_curve.png      # loss + ||bias|| curves
```

The CSV records the loss/perplexity, throughput, EGD internals
(`egd_iteration`, `egd_momentum_norm`, `egd_skipped`) and the **bias norms**
(`embed_b_norm`, `bQ_norm`, `bK_norm`, `bV_norm`) so you can see the
symmetry-breaking term evolve alongside the loss.

A run that already contains a `checkpoint_final.pt` is never overwritten: the new
run gets a timestamped directory instead.

---

## 9. Tests

```bash
python -m pytest tests -q
```

The suite covers the bias semantics (each mode, resampling, learnable),
attention mask polarity, padding invariance, causality, weight tying, the EGD
dynamics (`F0` floor, dimension handling, closure requirement), the tokenizer /
collate contract, and a fast end-to-end smoke run.

---

## 10. Notes and caveats

* **CPU only.** There is no GPU on the target machine; all presets are sized for
  it. `train.device: auto` resolves to `cuda`/`mps` automatically if you move the
  code somewhere with a GPU.
* Attention is written out by hand (explicit projections, scores, mask, softmax)
  rather than using `scaled_dot_product_attention`, so that the bias injection
  points stay visible and auditable. That is a deliberate readability-over-speed
  trade-off, and it is the main reason the configs stay small.
* BLEU is implemented in-tree (clipped n-gram precisions + brevity penalty); the
  reported `bleu_strict` is the unsmoothed score, while `bleu` applies add-1
  smoothing to zero precisions, which matters for very short hypotheses. If
  `sacrebleu` happens to be installed it is used to cross-check.
* Masks are boolean: `key_padding_mask` uses `True = PAD (ignore)` and
  `attn_mask` uses `True = BLOCKED`. Float/additive masks raise `TypeError`.

## Acknowledgements

The per-head `bQ`/`bK`/`bV` formulation and the energy-conserving descent
optimizer follow the companion project
`Symmetry-breaking-attention-bias`; the EGD implementation here is a port with an
initialisation bug fixed (the reference double-counted the parameter dimension if
the very first update was skipped).
