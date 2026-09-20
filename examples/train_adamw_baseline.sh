#!/bin/bash
# AdamW comparison run, same model / data / bias as train_b_gaussian.sh.
#
# EGD is the method under study; AdamW is the ordinary-descent reference it has
# to beat. On short CPU runs AdamW usually descends faster per step, while EGD's
# appeal is its asymptotic concentration near low loss -- so a fair comparison
# wants the same seed, the same b and the same number of updates.
#
# 1e-4 is the upstream AdamW default for these small models; raise it only if
# the loss curve is flat after the first hundred updates.

python scripts/train.py \
    --model small \
    --bias_preset b-gaussian \
    --dataset_preset multi30k-tiny \
    --optimizer adamw \
    --adam_lr 1e-4 \
    --adam_wd 0.01 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --batch_size 32 \
    --epochs 3 \
    --valid_every_updates 200 \
    --log_every 50 \
    --seed 42 \
    --log_dir runs \
    --name small-adamw-bgaussian-seed42
    # --wandb
