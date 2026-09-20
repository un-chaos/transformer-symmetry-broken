#!/bin/bash
# Embedding bias drawn once from a Gaussian: b ~ N(0, 0.02^2), fixed thereafter.
#
# A rotation R no longer commutes with the offset (R(x + b) != Rx + b), so the
# O(n_embd) symmetry of the symmetric control is broken by a *random* preferred
# direction in feature space. Resampling is `fixed`, so the breaking is a fixed
# property of the parameterisation rather than a per-step perturbation.

python scripts/train.py \
    --model small \
    --bias_preset b-gaussian \
    --bias_mode gaussian \
    --bias_mean 0.0 \
    --bias_std 0.02 \
    --bias_resample fixed \
    --bias_seed 1234 \
    --dataset_preset multi30k-tiny \
    --optimizer egd \
    --egd_lr 0.1 \
    --egd_eta 100.0 \
    --egd_consEn \
    --batch_size 32 \
    --epochs 3 \
    --valid_every_updates 200 \
    --log_every 50 \
    --seed 42 \
    --log_dir runs
    # --wandb
