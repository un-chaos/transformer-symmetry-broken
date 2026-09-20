#!/bin/bash
# Isotropic rank-one breaking: b = const_value in every embedding dimension.
#
# Same breaking as b-gaussian but with a *known* direction (the all-ones vector)
# instead of a random one. It isolates "some breaking is present" from "which
# direction was drawn", and it is the cleanest place to watch ||b|| in the CSV:
# embed_b_norm is sqrt(n_embd) * const_value and never moves.

python scripts/train.py \
    --model small \
    --bias_preset b-const \
    --bias_mode const \
    --bias_const 1.0 \
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
    --log_dir runs \
    --name small-egd-bconst-seed42
    # --wandb
