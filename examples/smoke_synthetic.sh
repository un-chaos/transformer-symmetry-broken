#!/bin/bash
# Offline smoke test: synthetic reverse task, no network, well under a minute.
#
# The synthetic splits are generated in-process (synthetic-reverse: the target
# is the source token list reversed), so this needs no download and no cache. It
# is the run to use when checking that a code change did not break the training
# loop, the CSV log, the checkpoints or the curve: 60 updates of the `smoke`
# preset finish in seconds on CPU.
#
# The bias is left symmetric so the smoke run also covers the b = 0 code path;
# swap in --bias_preset b-gaussian to exercise the bias machinery instead.

python scripts/train.py \
    --model smoke \
    --bias_preset symmetric \
    --dataset_preset synthetic-reverse \
    --synthetic_task reverse \
    --synthetic_train_size 512 \
    --synthetic_val_size 128 \
    --synthetic_test_size 128 \
    --synthetic_vocab 20 \
    --synthetic_len 8 \
    --tokenizer word \
    --dataset synthetic \
    --optimizer egd \
    --egd_lr 0.1 \
    --egd_eta 100.0 \
    --batch_size 32 \
    --max_steps 60 \
    --valid_every_updates 20 \
    --log_every 20 \
    --seed 42 \
    --device cpu \
    --log_dir runs \
    --name smoke-synthetic
    # --wandb
