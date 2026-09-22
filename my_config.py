# -*- coding: utf-8 -*-
# =====================================================================
#  这是你的实验设置 —— 想怎么调就怎么调。
#
#  改完保存，然后运行 run.bat（或 python run.py）选菜单 3)。
#
#  规则很简单：
#    * SETTINGS 里每一项都是一个真实的命令行参数（--xxx）。train.py 支持多少
#      参数，这里就能写多少 —— 没有任何参数是写死的。
#    * 值就是 Python 的值：数字写 0.1，字符串写 "abc"，开关写 True / False。
#    * 想用的参数不在 SETTINGS 里？看文件最下面的"全部参数"参考块，把它搬上来。
#    * EXTRA_ARGS 是万能兜底：直接写一串命令行参数，原样追加到最后。
# =====================================================================

FORMAT = 2

# ============ 常用设置：改这里就够了 ============
# 当前 bias 设置：b-gaussian
SETTINGS = {
    "--dataset_preset": 'multi30k-quick',   # 数据集：multi30k-quick(2000句) / multi30k-tiny(8000句) / multi30k(全部) / fineweb-quick / fineweb-10b(10B词) / synthetic-*(玩具)
    "--objective": None,   # 任务类型：translation=翻译（要平行语料）/ denoising=去噪（原始文本）
    "--model": 'tiny',   # 模型大小：smoke / tiny / small / base / large（越大越强、越慢）
    "--bias_preset": 'b-gaussian',   # 对称性破缺设置：symmetric / b-gaussian / b-const / attn-bQbV / attn-full / ...
    "--optimizer": 'egd',   # 优化器：egd（本项目的能量守恒下降法）/ adamw / sgdm
    "--egd_lr": None,   # EGD 学习率（和 --egd_F0 配套，改一个通常要一起调）
    "--egd_F0": None,   # EGD 的 loss 偏移，必须低于能达到的最小 loss；默认 -1 对交叉熵永远安全
    "--batch_size": None,   # 每批多少条数据（内存不够就调小）
    "--epochs": 3,   # 训练几轮（越大越慢、一般也越好）
    "--max_steps": None,   # 最多训练多少步（0 = 由 epochs 决定；快速试跑就写个小数字）
    "--ctx": None,   # 上下文长度（一句话最多多少个 token）
    "--n_embd": None,   # 隐藏维度（不写就用模型预设的值）
    "--n_head": None,   # 注意力头数（必须能整除 --n_embd）
    "--log_dir": 'runs',   # 结果保存到哪个文件夹
    "--name": 'quick-bgaussian',   # 这次实验的名字（空字符串 = 自动起名）

    # 这次用到的其它参数（也都可以改）：
    "--log_every": 4,
    "--valid_every_updates": 8,
}

# ============ 万能兜底：想加什么参数就写在这里 ============
# 例：EXTRA_ARGS = ["--seed", "123", "--grad_clip", "1.0"]
EXTRA_ARGS = []

# =====================================================================
#  全部参数（参考用，默认不用动）。
#
#  想调哪个：把那一行前面的 # 去掉、改成你要的值，然后把它搬到上面的
#  SETTINGS 里（或者直接在 SETTINGS 里照着写一行）。等号右边是 train.py
#  的默认值。参数清单是从 train.py 自动生成的，不会过期。
# =====================================================================
# === options ===
#   "-h": None,   # show this help message and exit
# === model shape ===
#   "--ctx": None,   # context length
#   "--n_embd": None,   # 
#   "--n_head": None,   # 
#   "--n_encoder_layer": None,   # 
#   "--n_decoder_layer": None,   # 
#   "--d_ff": None,   # 
#   "--dropout": None,   # 
#   "--attention_dropout": None,   # 
#   "--activation": None,   # 
#   "--use_prelu": None,   # shorthand for --activation prelu
#   "--prelu_random_init": None,   # 
#   "--prelu_slope_mean": None,   # mean of the random PReLU slopes (--prelu_random_init)
#   "--prelu_slope_std": None,   # std of the random PReLU slopes (--prelu_random_init)
#   "--norm_first": None,   # 
#   "--no_scaled_residual_init": None,   # disable the (2*n_layer)**-0.5 residual-output init scaling
#   "--no_scale_embedding": None,   # do not scale the embedding by sqrt(n_embd)
#   "--no_tie_output_embedding": None,   # use a separate output projection
#   "--share_embeddings": None,   # one embedding matrix for source and target (equal vocabs)
#   "--init_std": None,   # 
#   "--vocab": 12000,   # vocabulary size used for the --list_models parameter count
#   "--list_models": False,   # print the model / bias / dataset presets and exit
# === symmetry-breaking bias ===
#   "--symmetric": None,   # force b = 0 and every attention bias off (control run)
#   "--bias_mode": None,   # how the embedding bias b is drawn
#   "--bias_mean": None,   # 
#   "--bias_std": None,   # 
#   "--bias_const": None,   # value used by const mode
#   "--bias_resample": None,   # 
#   "--bias_learnable": None,   # train b instead of keeping it a fixed buffer
#   "--use_q_bias": None,   # 
#   "--use_k_bias": None,   # 
#   "--use_v_bias": None,   # 
#   "--attn_mode": None,   # 
#   "--attn_resample": None,   # 
#   "--attn_learnable": None,   # 
#   "--attn_const": None,   # value used by attn_mode=const
#   "--mean_Q": None,   # 
#   "--std_Q": None,   # 
#   "--mean_K": None,   # 
#   "--std_K": None,   # 
#   "--mean_V": None,   # 
#   "--std_V": None,   # 
#   "--no_share_across_heads": None,   # give every head its own bias vector
#   "--no_share_across_layers": None,   # give every layer its own bias object
#   "--no_apply_encoder": None,   # no per-head bias in the encoder self-attention
#   "--no_apply_decoder_self": None,   # no per-head bias in the decoder self-attention
#   "--no_apply_decoder_cross": None,   # no per-head bias in the decoder cross-attention
#   "--bias_seed": None,   # RNG seed of the embedding bias b
#   "--attn_seed": None,   # RNG seed of the per-head attention biases
# === data ===
#   "--dataset": None,   # where the text comes from
#   "--objective": None,   # translation needs parallel text; denoising trains the enco
#   "--data_dir": None,   # 
#   "--hf_repo": None,   # 
#   "--hf_endpoint": None,   # 
#   "--user_agent": None,   # the mirror answers 403 without a User-Agent header
#   "--download_timeout": None,   # 
#   "--src_field": None,   # 
#   "--tgt_field": None,   # 
#   "--max_train_samples": None,   # 0 = no limit
#   "--max_val_samples": None,   # 0 = no limit
#   "--local_src_col": None,   # 
#   "--local_tgt_col": None,   # 
#   "--synthetic_task": None,   # 
#   "--synthetic_train_size": None,   # 
#   "--synthetic_val_size": None,   # 
#   "--synthetic_test_size": None,   # 
#   "--synthetic_vocab": None,   # 
#   "--synthetic_len": None,   # 
#   "--synthetic_seed": None,   # 
#   "--fineweb_dir": None,   # directory holding the downloaded parquet shards
#   "--text_column": None,   # the text column inside the shards
#   "--max_documents": None,   # cap on documents read from the corpus (0 = all)
#   "--val_every": None,   # every N-th document is held out for validation
#   "--noise_density": None,   # fraction of tokens the denoising objective removes
#   "--mean_span_length": None,   # average length of a removed span
#   "--bpe_vocab_size": None,   # vocabulary size of the trained BPE tokenizer
#   "--tokenizer_train_documents": None,   # documents used to fit the BPE tokenizer
#   "--tokenizer": None,   # 
#   "--min_freq": None,   # 
#   "--max_vocab": None,   # 
#   "--no_lowercase": None,   # 
#   "--max_src_len": None,   # 0 = context length
#   "--max_tgt_len": None,   # 0 = context length
#   "--no_fallback_to_synthetic": None,   # fail loudly when a HuggingFace download fails
# === optimizer ===
#   "--egd_lr": 1.0,   # learning rate, rescaled internally by 1/sqrt(eta). Tuned t
#   "--egd_eta": 100.0,   # 
#   "--egd_F0": -1.0,   # loss offset; must stay BELOW the smallest reachable loss o
#   "--egd_auto_F0": False,   # resolve F0 as initial_loss - auto_F0_margin instead. WARNI
#   "--egd_auto_F0_margin": 1.0,   # margin used when --egd_auto_F0 is set
#   "--egd_nu": 0.0,   # 
#   "--egd_eps1": 1e-10,   # 
#   "--egd_eps2": 1e-40,   # 
#   "--egd_wd": 0.0,   # 
#   "--egd_consEn": True,   # 
#   "--no_egd_consEn": True,   # 
#   "--egd_seed": None,   # 
#   "--adam_lr": 0.0001,   # 
#   "--adam_wd": 0.01,   # 
#   "--adam_beta1": 0.9,   # 
#   "--adam_beta2": 0.95,   # 
#   "--sgdm_lr": 0.03,   # 
#   "--sgdm_momentum": 0.95,   # 
# === training ===
#   "--batch_size": 32,   # 
#   "--max_steps": 0,   # 0 = epochs x steps/epoch
#   "--grad_clip": 0.0,   # 0 disables clipping
#   "--label_smoothing": 0.0,   # 
#   "--save_every_updates": 0,   # 0 = only best/final
#   "--bleu_every": 0,   # 0 disables periodic BLEU
#   "--bleu_samples": 200,   # 0 = whole split
#   "--seed": 42,   # 
#   "--device": 'auto',   # 
#   "--num_workers": 0,   # 
#   "--use_bf16": False,   # autocast in bfloat16 where CUDA is available (no-op on CPU
#   "--max_seconds": 0.0,   # hard wall-clock stop; 0 = no limit
#   "--stall_warn_seconds": 300.0,   # 
#   "--max_eval_batches": 0,   # 0 = whole loader
# === run ===
#   "--resume": None,   # checkpoint (.pt) to continue from; the step counter picks 
#   "--wandb": None,   # log to Weights & Biases
#   "--no_plot": None,   # skip training_curve.png at the end
