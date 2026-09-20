#!/bin/bash
# Symmetric control run: b = 0 and every attention bias off.
#
# This is the baseline the whole repository is compared against. With b = 0 the
# embedding keeps its exact O(n_embd) rotation symmetry, so any difference seen
# in the b-gaussian / b-const runs is attributable to the symmetry breaking and
# not to the model, the data or the optimizer.
#
# EGD at lr = 0.1 (the measured default for this model size; see README).

python scripts/train.py \
    --model small \
    --bias_preset symmetric \
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
    --name small-egd-symmetric-seed42
    # --wandb
