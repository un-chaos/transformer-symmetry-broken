#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傻瓜式入口 —— 不用敲命令，输入数字就行；想调参数也不用改代码。

怎么用（Windows）：
    双击仓库里的  run.bat        ← 最简单
    或者在 VS Code 里打开这个文件，按右上角的运行按钮
    或者在终端里输入：python run.py

它做什么：
    1. 问你两三个问题（跑多久、用哪种 bias），都是输入数字；
    2. 自动调用训练、自动算 BLEU、自动出对比图和中文报告；
    3. 结果（日志 + 模型权重 + 损失曲线）都保存在 runs\\ 下面，并帮你打开文件夹。

想自由调参数（模型结构、初始值、超参……）：
    菜单 4) 打开 my_config.py。那个文件里的 SETTINGS 每一项都是一个真实的命令行
    参数，train.py 支持多少参数就能写多少 —— 没有写死的参数。改完保存，
    再用菜单 3) 跑。文件最下面还把 train.py 的全部参数列出来供参考。

进阶用户想直接用命令行：看 README.md，用 main.py。
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import runpy
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbreak_transformer.config import DATASET_PRESETS, PRESETS, BiasPresets  # noqa: E402
from symbreak_transformer.utils import configure_console_encoding  # noqa: E402

MY_CONFIG = ROOT / "my_config.py"
DEFAULT_LOG_DIR = "runs"


# ===================================================================== #
#  可选方案 —— 想加/改菜单选项改这里就够了
# ===================================================================== #
#: 「跑多久」。
#:
#: ``log_every`` / ``valid_every`` 决定曲线有多细，必须设：训练脚本默认值是给
#: 大模型的长训练调过的，用在几十步的短训练上只会画出一个孤零零的点。
SPEEDS: dict = {
    "1": {
        "key": "toy",
        "label": "玩具任务 —— 不用下载数据，每次约 15 秒",
        "minutes": "约 15 秒",
        "model": "smoke",
        "dataset_preset": "synthetic-copy",
        "epochs": 1,
        "max_steps": 60,
        "log_every": 5,
        "valid_every": 10,
        "extra": [],
    },
    "2": {
        "key": "quick",
        "label": "快速看看 —— 2000 句真实翻译数据，每次约 1 分钟",
        "minutes": "约 1 分钟",
        "model": "tiny",
        "dataset_preset": "multi30k-quick",
        "epochs": 2,
        "max_steps": 0,
        "log_every": 4,
        "valid_every": 8,
        "extra": [],
    },
    "3": {
        "key": "long",
        "label": "认真跑一次 —— 8000 句，每次约 8 分钟",
        "minutes": "约 8 分钟",
        "model": "small",
        "dataset_preset": "multi30k-tiny",
        "epochs": 2,
        "max_steps": 0,
        "log_every": 20,
        "valid_every": 40,
        "extra": [],
    },
    "4": {
        "key": "corpus",
        "label": "大规模语料 —— FineWeb-Edu 10B，需要先下载（很慢，CPU 上只适合试跑）",
        "minutes": "取决于 --max_steps，建议先用 --max_steps 限制",
        "model": "tiny",
        "dataset_preset": "fineweb-quick",
        "epochs": 1,
        "max_steps": 200,
        "log_every": 10,
        "valid_every": 20,
        "extra": [],
    },
}

#: bias 选项：编号 -> (bias 预设名, 一句话解释)
BIASES: dict = {
    "1": ("symmetric", "b = 0：对照组。完全不破缺对称性，用来跟其它几个比"),
    "2": ("b-gaussian", "高斯随机 b：给 embedding 加一个随机的固定偏置（推荐先看这个）"),
    "3": ("b-const", "常数 b：每个维度都加同一个常数"),
    "4": ("attn-bQbV", "注意力里的 bQ+bV：参考项目那种「每个头」的偏置"),
    "5": ("attn-full", "注意力里的 bQ+bK+bV：三个方向全都加上"),
    "6": ("b-learnable", "可学习的 b：b 不是固定的，交给优化器去学"),
}

#: 「跑三种做对比」用的三种
COMPARE_THREE = ["symmetric", "b-gaussian", "b-const"]


# ===================================================================== #
#  小工具
# ===================================================================== #
def hr(char: str = "=") -> None:
    print(char * 62)


def title(text: str) -> None:
    print()
    hr()
    print(f"  {text}")
    hr()


def ask(prompt: str, options: dict, default: str | None = None) -> str:
    """
    打印带编号的选项，读一个数字，返回对应的 key。

    Args:
        prompt: 问题本身。
        options: ``{key: 显示文字}`` 或 ``{key: (内部值, 说明)}``。
        default: 直接回车时用的 key。
    """
    if not options:
        return ""
    keys = list(options)
    print()
    print(prompt)
    for key, value in options.items():
        text = value[1] if isinstance(value, tuple) else value
        print(f"  {key}) {text}")
    if default is not None:
        print(f"  （直接按回车 = {default}）")
    while True:
        raw = input("请输入数字后回车：").strip()
        if not raw and default is not None:
            return default
        if raw in keys:
            return raw
        print(f"  看不懂 {raw!r}，请输入 {'/'.join(keys)} 中的一个")


def yes_no(prompt: str, default: bool = True) -> bool:
    """问一个是/否问题，回车取默认值。"""
    hint = "回车=是" if default else "回车=否"
    while True:
        raw = input(f"{prompt} (y/n，{hint}）：").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes", "是"):
            return True
        if raw in ("n", "no", "否"):
            return False
        print("  请输入 y 或 n")


def speed_by_key(key: str) -> dict:
    """按 ``SPEEDS`` 里的 ``key``（toy/quick/long/corpus）取配置。"""
    for spec in SPEEDS.values():
        if spec["key"] == key:
            return spec
    raise KeyError(f"unknown speed {key!r}; known: {[s['key'] for s in SPEEDS.values()]}")


def run_name_for(speed_key: str, bias_preset: str, optimizer: str = "egd") -> str:
    """给这次运行起一个看得懂的名字，例如 ``quick-bgaussian-egd``。"""
    short = bias_preset.replace("b-", "b").replace("-", "")
    name = f"{speed_key}-{short}"
    if optimizer != "egd":
        name += f"-{optimizer}"
    return name


# ===================================================================== #
#  参数表 / 命令行拼装
# ===================================================================== #
_train_module_cache = None


def train_parser() -> argparse.ArgumentParser:
    """``scripts/train.py`` 的 argparse 对象（缓存），用来知道有哪些参数。"""
    global _train_module_cache
    if _train_module_cache is None:
        path = ROOT / "scripts" / "train.py"
        spec = importlib.util.spec_from_file_location("_symbreak_train_script", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _train_module_cache = module
    return _train_module_cache.build_parser()


def _action_index(parser=None) -> dict:
    """``{"--flag": argparse action}``。"""
    parser = parser or train_parser()
    index: dict = {}
    for action in parser._actions:  # noqa: SLF001 - argparse has no public API
        for option in action.option_strings:
            index[option] = action
    return index


def option_catalog() -> list:
    """
    把 train.py 的全部参数按分组列出来（给控制面板当参考用）。

    Returns:
        ``[(组名, [(flag, default, help)])]``。
    """
    parser = train_parser()
    groups: dict = {}
    for action in parser._actions:  # noqa: SLF001
        if not action.option_strings:
            continue
        flag = action.option_strings[0]
        title_ = "其他"
        for group in parser._action_groups:
            if action in group._group_actions:  # noqa: SLF001
                title_ = group.title or "其他"
                break
        default = None if action.default is argparse.SUPPRESS else action.default
        groups.setdefault(title_, []).append((flag, default, (action.help or "").strip()))
    return list(groups.items())


def flags_to_argv(flags: dict, parser=None) -> list:
    """
    把 ``{"--flag": 值}`` 变成命令行参数表。

    约定：
    * 带值的参数：``"--n_embd": 256`` -> ``--n_embd 256``；值为 ``None`` 表示
      不传这个参数（用默认值）。
    * 开关参数：值就是"要不要加这个参数"。``"--norm_first": True`` 会加上
      ``--norm_first``，``False`` 就不加；``--no_*`` 这类反向开关同理
      （写 True 表示"要关掉那个东西"）。
    * 不认识的 flag 原样传下去，让 train.py 明确报错，而不是被悄悄吞掉。
    """
    index = _action_index(parser)
    argv: list = []
    for flag, value in flags.items():
        action = index.get(flag)
        if action is None:
            if value is True:
                argv.append(flag)
            elif value is not False and value is not None:
                argv += [flag, str(value)]
            continue
        is_switch = isinstance(
            action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
        ) or action.nargs == 0
        if is_switch:
            if bool(value):
                argv.append(flag)
        else:
            if value is None:
                continue
            argv += [flag, str(value)]
    return argv


def speed_flags(
    speed: dict,
    bias_preset: str,
    epochs: int | None = None,
    optimizer: str = "egd",
    run_name: str | None = None,
    log_dir: str = DEFAULT_LOG_DIR,
) -> dict:
    """菜单驱动的一次运行对应的参数表（返回值可直接喂给 :func:`flags_to_argv`）。"""
    flags: dict = {
        "--model": speed["model"],
        "--dataset_preset": speed["dataset_preset"],
        "--bias_preset": bias_preset,
        "--optimizer": optimizer,
        "--epochs": int(epochs if epochs is not None else speed["epochs"]),
        "--log_dir": log_dir,
        "--name": run_name or run_name_for(speed["key"], bias_preset, optimizer),
    }
    if speed.get("max_steps"):
        flags["--max_steps"] = int(speed["max_steps"])
    if speed.get("log_every"):
        flags["--log_every"] = int(speed["log_every"])
    if speed.get("valid_every"):
        flags["--valid_every_updates"] = int(speed["valid_every"])
    for item in speed.get("extra") or []:
        if item.startswith("--") and "=" in item:
            flag, _, value = item.partition("=")
            flags[flag] = value
    return flags


def build_train_command(
    speed: dict,
    bias_preset: str,
    run_name: str,
    log_dir: str = DEFAULT_LOG_DIR,
    epochs: int | None = None,
    optimizer: str = "egd",
    extra_flags: dict | None = None,
    extra_args: list | None = None,
) -> list:
    """
    拼出 ``python main.py train ...`` 的参数表。

    注意**不加** ``--no_plot``：损失曲线必须落盘（每个 run 目录里一份
    ``training_curve.png``），这是这个项目的硬要求。
    """
    flags = speed_flags(
        speed, bias_preset, epochs=epochs, optimizer=optimizer,
        run_name=run_name, log_dir=log_dir,
    )
    if extra_flags:
        flags.update(extra_flags)
    return ["train"] + flags_to_argv(flags) + [str(a) for a in (extra_args or [])]


def build_evaluate_command(ckpt: Path, dataset_preset: str, max_samples: int = 100) -> list:
    """拼出 ``python main.py evaluate ...`` 的参数表（跑完算一下 BLEU）。"""
    return [
        "evaluate",
        "--ckpt", str(ckpt),
        "--split", "test",
        "--dataset_preset", dataset_preset,
        "--max_samples", str(max_samples),
    ]


def build_report_command(log_dir: str = DEFAULT_LOG_DIR) -> list:
    """拼出 ``python main.py report ...`` 的参数表。"""
    return ["report", "--log_dir", log_dir]


def run_command(args: list, dry_run: bool = False) -> int:
    """执行 ``python main.py <args>``，输出直接显示在屏幕上。"""
    argv = [sys.executable, str(ROOT / "main.py")] + [str(a) for a in args]
    print()
    print("  正在执行：" + " ".join(argv[2:]))
    print()
    if dry_run:
        print("  （dry-run：只显示不执行）")
        return 0
    return subprocess.call(argv, cwd=str(ROOT))


def open_folder(path: Path) -> None:
    """在文件管理器里打开结果文件夹（失败也不报错）。"""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.call(["open", str(path)])
        else:
            subprocess.call(["xdg-open", str(path)])
        print(f"  已打开：{path}")
    except Exception as exc:  # pragma: no cover - depends on desktop
        print(f"  打不开文件夹（{exc}），你可以自己去看：{path}")


def open_in_editor(path: Path) -> None:
    """用系统默认程序打开文件（.py 一般就是 VS Code 或记事本）。"""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.call(["open", str(path)])
        else:
            subprocess.call(["xdg-open", str(path)])
        print(f"  已打开：{path}")
        print("  改完记得保存，然后回菜单选 3) 就能用新参数跑。")
    except Exception as exc:  # pragma: no cover
        print(f"  打不开（{exc}）。你可以自己用记事本/VS Code 打开：{path}")


# ===================================================================== #
#  my_config.py —— 自由配置的控制面板
# ===================================================================== #
#  设计要点：面板里的每一项都是一个**真实的命令行参数**（``--flag``），参数清单由
#  scripts/train.py 的 argparse 自动生成，所以
#    * 没有参数是写死的：train.py 有多少个参数，这里就能配多少个；
#    * 不会过期：以后 train.py 加了新参数，重新生成一次就会出现在"全部参数"里。
#  用户只需要改 SETTINGS 里那几行；想调更细的就把"全部参数"里的某行搬到 SETTINGS。

#: 面板格式版本；读到别的格式就忽略并重新生成，避免解析旧文件出错。
PANEL_FORMAT = 2

#: SETTINGS 默认放哪些参数（常用），顺序即显示顺序。值是中文说明。
COMMON_FLAGS: dict = {
    "--dataset_preset": "数据集：multi30k-quick(2000句) / multi30k-tiny(8000句) / "
                        "multi30k(全部) / fineweb-quick / fineweb-10b(10B词) / synthetic-*(玩具)",
    "--objective": "任务类型：translation=翻译（要平行语料）/ denoising=去噪（原始文本）",
    "--model": "模型大小：smoke / tiny / small / base / large（越大越强、越慢）",
    "--bias_preset": "对称性破缺设置：symmetric / b-gaussian / b-const / attn-bQbV / attn-full / ...",
    "--optimizer": "优化器：egd（本项目的能量守恒下降法）/ adamw / sgdm",
    "--egd_lr": "EGD 学习率（和 --egd_F0 配套，改一个通常要一起调）",
    "--egd_F0": "EGD 的 loss 偏移，必须低于能达到的最小 loss；默认 -1 对交叉熵永远安全",
    "--batch_size": "每批多少条数据（内存不够就调小）",
    "--epochs": "训练几轮（越大越慢、一般也越好）",
    "--max_steps": "最多训练多少步（0 = 由 epochs 决定；快速试跑就写个小数字）",
    "--ctx": "上下文长度（一句话最多多少个 token）",
    "--n_embd": "隐藏维度（不写就用模型预设的值）",
    "--n_head": "注意力头数（必须能整除 --n_embd）",
    "--log_dir": "结果保存到哪个文件夹",
    "--name": "这次实验的名字（空字符串 = 自动起名）",
}


def _panel_reference_block(current: dict) -> str:
    """文件末尾的"全部参数"参考块：把 train.py 的每个参数以注释形式列出来。"""
    lines = [
        "# =====================================================================",
        "#  全部参数（参考用，默认不用动）。",
        "#",
        "#  想调哪个：把那一行前面的 # 去掉、改成你要的值，然后把它搬到上面的",
        "#  SETTINGS 里（或者直接在 SETTINGS 里照着写一行）。等号右边是 train.py",
        "#  的默认值。参数清单是从 train.py 自动生成的，不会过期。",
        "# =====================================================================",
    ]
    for group_title, entries in option_catalog():
        shown = [item for item in entries if item[0] not in current]
        if not shown:
            continue
        lines.append(f"# === {group_title} ===")
        for flag, default, help_text in shown:
            note = help_text.replace("\n", " ")[:58]
            rendered = "None" if default is None else repr(default)
            lines.append(f'#   "{flag}": {rendered},   # {note}')
    return "\n".join(lines)


def render_my_config(flags: dict, extra: list | None = None) -> str:
    """生成 ``my_config.py`` 的内容（中文说明 + 完整参数参考）。"""
    bias = flags.get("--bias_preset", "(默认)")
    rows = []
    for flag, note in COMMON_FLAGS.items():
        value = flags.get(flag)
        rendered = "None" if value is None else repr(value)
        rows.append(f'    "{flag}": {rendered},   # {note}')
    known = set(COMMON_FLAGS)
    other = {k: v for k, v in flags.items() if k not in known}
    if other:
        rows.append("")
        rows.append("    # 这次用到的其它参数（也都可以改）：")
        for flag, value in other.items():
            rows.append(f'    "{flag}": {value!r},')
    body = "\n".join(rows)
    return f'''\
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

FORMAT = {PANEL_FORMAT}

# ============ 常用设置：改这里就够了 ============
# 当前 bias 设置：{bias}
SETTINGS = {{
{body}
}}

# ============ 万能兜底：想加什么参数就写在这里 ============
# 例：EXTRA_ARGS = ["--seed", "123", "--grad_clip", "1.0"]
EXTRA_ARGS = {list(extra or [])!r}

{_panel_reference_block(flags)}
'''


def write_my_config(flags: dict, path=MY_CONFIG, extra: list | None = None):
    """把设置写进 ``my_config.py``（返回写入的路径）。"""
    path = Path(path)
    path.write_text(render_my_config(flags, extra), encoding="utf-8")
    return path


def read_my_config(path=MY_CONFIG):
    """
    读回 ``my_config.py``。

    Returns:
        ``{"flags": {...}, "extra": [...]}``；文件不存在、格式不对、或被手改坏了
        （比如少了引号）都返回 ``None`` —— 菜单当成"没有上次的设置"，绝不崩。
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        namespace = runpy.run_path(str(path))
    except Exception as exc:  # noqa: BLE001 - a hand-edited file must not break us
        print(f"（my_config.py 读不了：{exc}")
        print("  没关系，忽略它，用菜单选就行；选一次之后我可以重新生成一份）")
        return None
    if int(namespace.get("FORMAT", 0)) != PANEL_FORMAT:
        print("（my_config.py 是旧格式，已忽略；菜单 4) 可以重新生成一份）")
        return None
    settings = namespace.get("SETTINGS")
    if not isinstance(settings, dict) or not settings:
        print("（my_config.py 里没有 SETTINGS，已忽略）")
        return None
    extra = namespace.get("EXTRA_ARGS") or []
    if not isinstance(extra, list):
        print("（my_config.py 的 EXTRA_ARGS 不是列表，已忽略它）")
        extra = []
    return {"flags": {str(k): v for k, v in settings.items()}, "extra": [str(a) for a in extra]}


def maybe_write_panel(flags: dict) -> None:
    """
    第一次跑完时生成一份 ``my_config.py`` 模板。

    已经有这个文件就**绝不覆盖** —— 用户可能已经仔细调过里面的参数了。
    """
    if MY_CONFIG.exists():
        print(f"  （{MY_CONFIG.name} 已存在，没有覆盖它。想按刚才的设置重写，")
        print(f"    就先删掉这个文件，然后选菜单 4) 重新生成）")
        return
    try:
        write_my_config(flags)
        print(f"  已生成 {MY_CONFIG.name}（想自由调参数就选菜单 4)）")
    except Exception as exc:  # noqa: BLE001
        print(f"  （生成 {MY_CONFIG.name} 失败：{exc}）")


# ===================================================================== #
#  各个菜单动作
# ===================================================================== #
_BIAS_TEXT = {
    "symmetric": "对照组：b = 0（不破缺对称性）",
    "b-gaussian": "高斯随机 b（随机方向破缺）",
    "b-const": "常数 b（每个维度加同一个常数）",
    "attn-bQbV": "注意力里的 bQ + bV",
    "attn-full": "注意力里的 bQ + bK + bV",
    "b-learnable": "可学习的 b（交给优化器学）",
}


def describe(speed: dict, bias_preset: str, epochs: int, optimizer: str, log_dir: str) -> None:
    """用大白话把「接下来要干什么」说一遍，让人确认。"""
    bias_text = _BIAS_TEXT.get(bias_preset, bias_preset)
    print()
    print("  接下来会这样做：")
    data_note = speed["label"].split("——")[-1].strip()
    print(f"    数据      ：{speed['dataset_preset']}（{data_note}）")
    print(f"    模型大小  ：{speed['model']}")
    print(f"    对称性破缺：{bias_text}")
    print(f"    训练轮数  ：{epochs}")
    print(f"    优化器    ：{optimizer}")
    print(f"    预计耗时  ：{speed['minutes']}")
    print(f"    结果放在  ：{Path(log_dir).resolve()}")
    print(f"    （损失曲线会存成 <结果目录>/training_curve.png）")


def do_training(
    speed: dict,
    bias_preset: str,
    epochs: int,
    optimizer: str,
    log_dir: str,
    dry_run: bool = False,
    want_evaluate: bool = True,
    extra_flags: dict | None = None,
    extra_args: list | None = None,
    run_name: str | None = None,
) -> int:
    """
    跑一次训练（+ 尽量算一下测试集 BLEU），返回退出码。

    训练失败只会打印提示，不会让整个菜单崩掉。
    """
    run_name = run_name or run_name_for(speed["key"], bias_preset, optimizer)
    command = build_train_command(
        speed, bias_preset, run_name, log_dir=log_dir, epochs=epochs,
        optimizer=optimizer, extra_flags=extra_flags, extra_args=extra_args,
    )
    code = run_command(command, dry_run=dry_run)
    if code != 0:
        print()
        print(f"  训练出错了（返回码 {code}）。常见原因：")
        print("    - 没连上网，下载不了数据（可以先选「玩具任务」，不需要网络）")
        print("    - 大语料还没下载完（先跑 python main.py download-data --status 看看）")
        print("    - 磁盘空间不够")
        print("  详细报错信息在上面的输出里。")
        return code

    run_dir = Path(log_dir) / run_name
    if not dry_run:
        print()
        print(f"  结果已保存到：{run_dir.resolve()}")
        for name, what in (
            ("training_curve.png", "损失曲线图（训练 + 验证）"),
            ("model_summary.txt", "模型说明书：结构 / 维度 / 参数量"),
            ("training_log.csv", "每步的 loss / 学习率等，Excel 可打开"),
            ("losses.csv", "每个训练步的 loss"),
            ("config.json", "这次的模型设置"),
            ("bias.json", "这次的 bias 设置"),
            ("args.json", "这次用的全部参数"),
            ("model_best.pt", "验证损失最好的模型权重"),
            ("model_final.pt", "训练结束时的模型权重"),
            ("summary.json", "这次运行的汇总"),
        ):
            if (run_dir / name).exists():
                print(f"    {name:<20} {what}")

    ckpt = run_dir / "model_best.pt"
    if want_evaluate and ckpt.exists() and not dry_run:
        print()
        print("  再算一下测试集上的 loss 和 BLEU（越高越好）……")
        try:
            run_command(build_evaluate_command(ckpt, speed["dataset_preset"]))
        except Exception as exc:  # noqa: BLE001
            print(f"  （没算成：{exc}；不影响训练结果）")
    return 0


def do_report(log_dir: str, dry_run: bool = False, open_when_done: bool = False) -> int:
    """出对比图和对比表格。"""
    print()
    print("  正在汇总所有结果，生成对比图……")
    code = run_command(build_report_command(log_dir), dry_run=dry_run)
    if code != 0:
        print("  （汇总失败，可能还没跑过任何实验。先去跑一个吧）")
    elif open_when_done and not dry_run:
        if yes_no("  要现在打开结果文件夹吗？", default=True):
            open_folder(Path(log_dir).resolve())
    return code


def action_compare_three(dry_run: bool = False) -> None:
    """跑三种 bias 各一次，然后出对比图 —— 第一次用推荐这个。"""
    key = ask("先选「跑多久」：", {k: v["label"] for k, v in SPEEDS.items()}, default="2")
    speed = SPEEDS[key]
    print()
    print("  会依次跑这三种，其它设置完全相同：")
    print("    1. b = 0（对照组，不破缺对称性）")
    print("    2. 高斯随机 b（随机方向破缺）")
    print("    3. 常数 b（每个维度加同一个常数）")
    print(f"  每一次{speed['minutes']}，三次总共大概 {_triple(speed['minutes'])}。")
    if not yes_no("  可以开始吗？", default=True):
        return
    last_flags: dict = {}
    for index, bias_preset in enumerate(COMPARE_THREE, start=1):
        title(f"第 {index}/3 次，正在跑：{bias_preset}")
        do_training(speed, bias_preset, speed["epochs"], "egd", DEFAULT_LOG_DIR, dry_run)
        last_flags = speed_flags(
            speed, bias_preset, epochs=speed["epochs"], optimizer="egd"
        )
    maybe_write_panel(last_flags)
    do_report(DEFAULT_LOG_DIR, dry_run, open_when_done=True)


def _triple(duration: str) -> str:
    """
    把「约 1 分钟」/「约 15 秒」换算成三倍时长的说法。

    必须区分秒和分钟：无脑当成分钟的话，「约 15 秒」会变成「约 45 分钟」。
    """
    digits = "".join(ch for ch in duration if ch.isdigit())
    if not digits:
        return duration
    total = int(digits) * 3
    if "秒" in duration:
        return f"约 {total // 60} 分钟" if total >= 60 else f"约 {total} 秒"
    return f"约 {total} 分钟"


def action_single(dry_run: bool = False) -> None:
    """只跑一种 bias。"""
    key = ask("先选「跑多久」：", {k: v["label"] for k, v in SPEEDS.items()}, default="2")
    speed = SPEEDS[key]
    bias_key = ask("用哪种 bias？", {k: v[1] for k, v in BIASES.items()}, default="2")
    bias_preset = BIASES[bias_key][0]
    epochs_raw = input(f"  训练几轮？（直接回车 = {speed['epochs']}）：").strip()
    epochs = int(epochs_raw) if epochs_raw.isdigit() and int(epochs_raw) > 0 else speed["epochs"]
    describe(speed, bias_preset, epochs, "egd", DEFAULT_LOG_DIR)
    if not yes_no("  可以开始吗？", default=True):
        return
    title(f"正在跑：{bias_preset}")
    do_training(speed, bias_preset, epochs, "egd", DEFAULT_LOG_DIR, dry_run)
    maybe_write_panel(
        speed_flags(speed, bias_preset, epochs=epochs, optimizer="egd")
    )
    do_report(DEFAULT_LOG_DIR, dry_run, open_when_done=True)


def action_from_config(dry_run: bool = False) -> None:
    """按 ``my_config.py`` 里的设置跑（参数完全由那个文件决定）。"""
    panel = read_my_config()
    if not panel:
        print()
        print(f"  还没有可用的 {MY_CONFIG.name}。选菜单 4) 我可以立刻生成一份带中文说明的。")
        return
    flags = dict(panel["flags"])
    extra = list(panel["extra"])

    print()
    print(f"  将按 {MY_CONFIG.name} 运行，参数如下（共 {len(flags)} 项）：")
    for flag, value in flags.items():
        print(f"    {flag:<26} {value}")
    if extra:
        print(f"    EXTRA_ARGS                 {' '.join(extra)}")
    print()
    print("  注意：--log_dir / --name 决定结果存哪里、叫什么。")
    if not yes_no("  可以开始吗？", default=True):
        return

    command = ["train"] + flags_to_argv(flags) + extra
    code = run_command(command, dry_run=dry_run)
    if code != 0:
        return
    log_dir = str(flags.get("--log_dir", DEFAULT_LOG_DIR))
    do_report(log_dir, dry_run, open_when_done=True)


def action_edit_config() -> None:
    """打开 ``my_config.py`` 让用户自由调参数（没有就先生成一份）。"""
    if not MY_CONFIG.exists():
        panel = read_my_config()
        write_my_config(panel["flags"] if panel else speed_flags(
            SPEEDS["2"], "b-gaussian", epochs=2
        ))
        print(f"  已生成一份 {MY_CONFIG.name}（里面每一项都有中文说明）。")
    open_in_editor(MY_CONFIG)


def action_help() -> None:
    """讲清楚每一项是什么意思、能改什么。"""
    title("我能改什么？")
    print(f"""
  1) 最省事：直接运行 run.py，用数字选。不用改任何文件。

  2) 想自由调参数：选菜单 4) 打开 {MY_CONFIG.name}。
     里面 SETTINGS 的每一项都是一个真实的命令行参数，比如
         "--n_embd": 256,        # 隐藏维度
         "--dropout": 0.1,       # dropout
         "--egd_lr": 1.0,        # 学习率
         "--bias_mode": "gaussian",
     改完保存，回菜单选 3) 就跑。**没有写死的参数**：train.py 支持多少参数，
     那个文件里就能写多少；文件最下面还把所有参数按分组列出来了。

     也支持 {MY_CONFIG.name} 里的 EXTRA_ARGS 直接追加命令行参数。

  3) 想看全部参数：菜单里有「查看全部参数」，或者命令行
         python main.py train --help

  几个关键概念：

    bias（偏置）—— 就是这个项目要研究的东西。
      b = 0        不破缺对称性（对照组）
      高斯随机 b    给 embedding 加一个随机的固定偏置 → 破缺对称性
      常数 b        每个维度加同一个常数 → 也破缺
      注意力 bQ/bV  在 attention 内部加偏置（参考项目那种做法）
      加了这个偏置以后，attention 原来的「旋转对称性」就被打破了。

    dataset / objective —— 数据从哪来、训练什么任务。
      multi30k-*   翻译（translation）：平行语料，主指标是 BLEU
      fineweb-*    去噪（denoising）：10B 词英文网页语料，不需要翻译标注
                   先跑 python main.py download-data 下载（约 28.5 GB，可断点续传）

    egd_lr / egd_F0 —— EGD 的两个关键超参，是配套的：lr 乘在动量上，
    (loss - F0) 除在动量上。把 F0 改小就要把 lr 相应调大。

  模型信息：
      每次训练一开始都会打印一份「模型说明书」，并同时存成
      runs\\<名字>\\model_summary.txt —— 里面写明结构（几层/多宽/几个头）、
      每一部分各有多少参数、参数主要花在哪、以及哪些张量是「不训练」的
      缓冲区（本项目的偏置 b 就在里面）。想知道模型到底多大，看这个文件。

  结果怎么看：
      每次训练都会存 runs\\<名字>\\：损失曲线图 training_curve.png、模型说明书
      model_summary.txt、每步日志 training_log.csv、这次的设置 config.json /
      bias.json / args.json、以及模型权重。跑完自动生成
      runs\\_compare\\compare.png 把所有实验叠在一张图上对比，还有
      compare.txt（中文表格 + 结论）。验证损失越低越好，BLEU 越高越好。
      如果几条曲线几乎重合，说明差别是噪声 —— 试着训练更久、
      换更大模型或更多数据。
""")


def action_list_options() -> None:
    """把 train.py 的全部参数按分组打印出来。"""
    title("全部参数（train.py 支持的所有设置）")
    for group_title, entries in option_catalog():
        print(f"\n  --- {group_title} ---")
        for flag, default, help_text in entries:
            rendered = "None" if default is None else repr(default)
            note = f"  # {help_text}" if help_text else ""
            print(f"    {flag:<28} 默认 {rendered}{note}")
    print(f"\n  这些都可以写进 {MY_CONFIG.name} 的 SETTINGS 里。")


# ===================================================================== #
#  主菜单
# ===================================================================== #
MENU = {
    "1": "跑三种 bias 做对比（b=0 / 高斯 / 常数）   ← 第一次用选这个",
    "2": "只跑一种 bias",
    "3": f"按 {MY_CONFIG.name} 里的设置跑（可自由调参数）",
    "4": f"打开 {MY_CONFIG.name} 调参数（自由配置）",
    "5": "看已有结果（出对比图）",
    "6": "打开结果文件夹",
    "7": "查看全部参数 / 说明",
    "8": "下载大规模语料（FineWeb-Edu 10B，可断点续传）",
    "0": "退出",
}


def welcome() -> None:
    print()
    hr("#")
    print("#  对称性破缺 Transformer —— 一键运行")
    print("#  不用敲命令：输入数字、按回车就行")
    hr("#")


def action_download(dry_run: bool = False) -> None:
    """下载大规模语料（在新窗口里跑，方便看到进度）。"""
    print()
    print("  将下载 FineWeb-Edu sample/10BT（约 10B 词、14 个分片、约 28.5 GB）。")
    print("  这是**断点续传**的：中途断了再跑一次就会接着下，不会重复下载。")
    status = ["download-data", "--status"]
    run_command(status)
    if not yes_no("  现在开始下载吗？（会另开一个窗口显示进度）", default=True):
        return
    if dry_run:
        print("  （dry-run：不启动）")
        return
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(
                ["cmd.exe", "/k", f'cd /d "{ROOT}" && python main.py download-data'],
                cwd=str(ROOT),
            )
            print("  已在新窗口里开始下载。那个窗口会实时显示进度，")
            print("  下完它自己会停住（按任意键关闭）。")
        else:
            run_command(["download-data"])
    except Exception as exc:  # noqa: BLE001
        print(f"  开新窗口失败（{exc}），改为在当前窗口下载。")
        run_command(["download-data"])


def main(argv=None) -> int:
    """菜单主循环。``argv`` 预留给测试（``--from-config`` / ``--speed`` 等）。"""
    configure_console_encoding()
    argv = list(sys.argv[1:] if argv is None else argv)

    if "--from-config" in argv:
        return _run_from_config_non_interactive(argv)
    if "--speed" in argv or "--bias" in argv:
        return _non_interactive(argv)

    welcome()
    while True:
        print()
        for key, text in MENU.items():
            print(f"  {key}) {text}")
        choice = input("\n请输入数字后回车：").strip()

        if choice == "1":
            action_compare_three()
        elif choice == "2":
            action_single()
        elif choice == "3":
            action_from_config()
        elif choice == "4":
            action_edit_config()
        elif choice == "5":
            do_report(DEFAULT_LOG_DIR, open_when_done=True)
        elif choice == "6":
            folder = Path(DEFAULT_LOG_DIR).resolve()
            folder.mkdir(parents=True, exist_ok=True)
            open_folder(folder)
        elif choice == "7":
            sub = ask(
                "看哪个？",
                {"1": "我能改什么（说明）", "2": "train.py 的全部参数"},
                default="1",
            )
            if sub == "2":
                action_list_options()
            else:
                action_help()
        elif choice == "8":
            action_download()
        elif choice in ("0", "q", "quit", "exit"):
            print("\n  再见。结果都在 runs\\ 里面。\n")
            return 0
        else:
            print(f"  没有 {choice!r} 这个选项，请输入 0-8")


def _run_from_config_non_interactive(argv: list) -> int:
    """``run.py --from-config [--dry-run]``：完全按 my_config.py 跑，不提问。"""
    dry_run = "--dry-run" in argv
    panel = read_my_config()
    if not panel:
        print("no usable my_config.py")
        return 2
    command = ["train"] + flags_to_argv(panel["flags"]) + list(panel["extra"])
    if dry_run:
        print("dry-run 命令：python main.py " + " ".join(command))
        return 0
    code = run_command(command)
    if code != 0:
        return code
    do_report(str(panel["flags"].get("--log_dir", DEFAULT_LOG_DIR)))
    return 0


def _non_interactive(argv: list) -> int:
    """给测试和脚本用的非交互模式：按给定设置跑，不做任何提问。"""
    speed_key = "quick"
    bias_preset = "b-gaussian"
    epochs = None
    dry_run = False
    log_dir = DEFAULT_LOG_DIR
    for index, item in enumerate(argv):
        if item == "--speed" and index + 1 < len(argv):
            speed_key = argv[index + 1]
        elif item == "--bias" and index + 1 < len(argv):
            bias_preset = argv[index + 1]
        elif item == "--epochs" and index + 1 < len(argv):
            epochs = int(argv[index + 1])
        elif item == "--log_dir" and index + 1 < len(argv):
            log_dir = argv[index + 1]
        elif item == "--dry-run":
            dry_run = True

    try:
        speed = speed_by_key(speed_key)
    except KeyError:
        print(f"unknown speed {speed_key!r}; choose from "
              f"{[s['key'] for s in SPEEDS.values()]}")
        return 2
    if bias_preset not in BiasPresets:
        print(f"unknown bias {bias_preset!r}; choose from {sorted(BiasPresets)}")
        return 2
    describe(speed, bias_preset, epochs or speed["epochs"], "egd", log_dir)
    if dry_run:
        print()
        print("  dry-run 命令：python main.py " + " ".join(
            build_train_command(
                speed, bias_preset, run_name_for(speed["key"], bias_preset),
                log_dir=log_dir, epochs=epochs,
            )
        ))
        return 0
    code = do_training(speed, bias_preset, epochs or speed["epochs"], "egd", log_dir)
    if code != 0:
        return code
    do_report(log_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
