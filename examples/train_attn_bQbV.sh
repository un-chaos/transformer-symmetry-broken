#!/bin/bash
# Reference-style per-head biases bQ + bV (the companion project's setting).
#
# bQ enters through the softmax and is exponentially amplified (Q-K sector,
# O(head_dim) symmetry); bV only passes through a linear map and acts with
# power-law strength (V-O sector). bK stays off: the key-independent part of a
# constant shift cancels in the softmax normalisation.
#
# --no_share_across_heads gives every head its own bias vector instead of one
# head_dim vector expanded to all of them, which is the per-head variant the
# reference paper studies.

python scripts/train.py \
    --model small \
    --bias_preset attn-bQbV \
    --use_q_bias \
    --use_v_bias \
    --attn_mode gaussian \
    --attn_resample fixed \
    --mean_Q 0.5 \
    --std_Q 0.05 \
    --mean_V 0.5 \
    --std_V 0.05 \
    --no_share_across_heads \
    --bias_seed 1234 \
    --dataset_preset multi30k-tiny \
    --optimizer egd \
    --egd_lr 0.1 \
    --egd_eta 100.0 \
    --egd_consEn \
    --batch_size 32 \
    --epochs 3 \
    --valid_every_updates 200 \
    --seed 42 \
    --log_dir runs
    # --wandb
