#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
傻瓜式入口 —— 不用敲命令、不用改代码，输入数字就行。

怎么用（Windows）：
    双击仓库里的  run.bat        ← 最简单
    或者在 VS Code 里打开这个文件，按右上角的运行按钮
    或者在终端里输入：python run.py

它做什么：
    1. 问你两三个问题（跑多久、用哪种 bias），都是输入数字；
    2. 自动调用训练、自动出对比图和对比表格；
    3. 结果放在 runs\\ 下面，可以直接打开文件夹看。

想自己调设置（比如换模型大小、换数据量）：
    菜单里选「按 my_config.py 里的设置跑」，或者直接编辑 my_config.py
    （那个文件里每一项都有中文说明，照着改一个引号里的词就行）。

进阶用户想直接用命令行：看 README.md，用 main.py。
"""

from __future__ import annotations

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
#  可选方案 —— 想加/改选项改这里就够了
# ===================================================================== #
#: 「跑多久」。数字越靠后越慢、效果一般越好。
#:
#: ``log_every`` / ``valid_every`` 决定曲线有多细。它们一定要设：默认值是按
#: 大模型的长训练调过的，用在几十步的短训练上只会画出一个孤零零的点，
#: 对比图就没法看了。
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
}

#: bias 选项：显示名字 -> (bias 预设名, 一句话解释)
BIASES: dict = {
    "1": ("symmetric", "b = 0：对照组。完全不破缺对称性，用来跟其它两个比"),
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

    Returns:
        选中的 key。``options`` 为空时返回 ""。
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
    """按 ``SPEEDS`` 里的 ``key``（toy/quick/long）取配置。"""
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


def build_train_command(
    speed: dict,
    bias_preset: str,
    run_name: str,
    log_dir: str = DEFAULT_LOG_DIR,
    epochs: int | None = None,
    optimizer: str = "egd",
) -> list:
    """
    拼出 ``python main.py train ...`` 的参数表。

    Args:
        speed: :data:`SPEEDS` 里的一项。
        bias_preset: ``BiasPresets`` 里的名字。
        run_name: 结果文件夹的名字。
        log_dir: 结果根目录。
        epochs: 覆盖 ``speed`` 里的轮数；``None`` 表示用它自带的。
        optimizer: ``egd`` / ``adamw`` / ``sgdm``。

    Returns:
        可直接交给 ``subprocess`` 的参数列表（不含 python 和 main.py）。
    """
    command = [
        "train",
        "--model", speed["model"],
        "--dataset_preset", speed["dataset_preset"],
        "--bias_preset", bias_preset,
        "--optimizer", optimizer,
        "--epochs", str(epochs if epochs is not None else speed["epochs"]),
        "--log_dir", log_dir,
        "--name", run_name,
        "--no_plot",
    ]
    if speed.get("max_steps"):
        command += ["--max_steps", str(speed["max_steps"])]
    # Always set the cadences explicitly: the defaults are tuned for long runs on
    # big models and would log a single point here, leaving the comparison figure
    # with nothing to draw.
    if speed.get("log_every"):
        command += ["--log_every", str(speed["log_every"])]
    if speed.get("valid_every"):
        command += ["--valid_every_updates", str(speed["valid_every"])]
    command += list(speed.get("extra") or [])
    return command


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


# ===================================================================== #
#  my_config.py —— 说明书式的设置文件
# ===================================================================== #
def render_my_config(settings: dict) -> str:
    """生成 ``my_config.py`` 的内容（带中文说明）。"""
    bias = settings.get("bias", "b-gaussian")
    explanation = {
        "symmetric": "b = 0，对照组",
        "b-gaussian": "高斯随机 b",
        "b-const": "常数 b",
        "attn-bQbV": "注意力里的 bQ+bV",
        "attn-full": "注意力里的 bQ+bK+bV",
        "b-learnable": "可学习的 b",
    }.get(bias, bias)
    return f'''\
# -*- coding: utf-8 -*-
# =====================================================================
#  这是你的实验设置。改完保存，然后运行 run.py（或双击 run.bat）。
#
#  不想改也没关系：直接运行 run.py，用菜单里的数字选就行了。
#  这个文件只是把「上次的选择」记下来，方便你下次一键重跑。
# =====================================================================

# 跑多快？（toy = 玩具任务 30 秒 / quick = 快速 3 分钟 / long = 认真 20 分钟）
SPEED = "{settings.get('speed', 'quick')}"

# 用哪种 bias？
#   symmetric    b = 0，对照组（不破缺对称性）
#   b-gaussian   高斯随机 b                        ← 推荐先看这个
#   b-const      常数 b
#   attn-bQbV    参考项目那种"每个头"的 bQ+bV
#   attn-full    注意力里的 bQ+bK+bV
#   b-learnable  可学习的 b
BIAS = "{bias}"          # 当前：{explanation}

# 训练轮数。越大越慢、一般也越好。快速模式 2 轮、认真模式 2-3 轮比较合适。
EPOCHS = {settings.get('epochs', 2)}

# 优化器：egd（本项目的能量守恒下降法）/ adamw（常规对照）/ sgdm
OPTIMIZER = "{settings.get('optimizer', 'egd')}"

# 所有结果都放在这个文件夹里（想分开保存就改成别的名字）
LOG_DIR = "{settings.get('log_dir', DEFAULT_LOG_DIR)}"

# 跑完自动出对比图吗？
AUTO_REPORT = {settings.get('auto_report', True)}
'''


def write_my_config(settings: dict, path: Path = MY_CONFIG) -> Path:
    """把设置写进 ``my_config.py``。"""
    path.write_text(render_my_config(settings), encoding="utf-8")
    return path


def read_my_config(path: Path = MY_CONFIG) -> dict | None:
    """
    读回 ``my_config.py``。

    手动改坏了（比如少了引号）不会让程序崩掉：读失败就返回 ``None``，菜单会
    当成"没有上次的设置"。
    """
    if not path.exists():
        return None
    try:
        namespace = runpy.run_path(str(path))
    except Exception as exc:
        print(f"（my_config.py 读不了：{exc}")
        print("  没关系，忽略它，用菜单选就行）")
        return None
    keys = ("SPEED", "BIAS", "EPOCHS", "OPTIMIZER", "LOG_DIR", "AUTO_REPORT")
    return {k.lower(): namespace[k] for k in keys if k in namespace}


# ===================================================================== #
#  各个菜单动作
# ===================================================================== #
def describe(speed: dict, bias_preset: str, epochs: int, optimizer: str, log_dir: str) -> None:
    """用大白话把「接下来要干什么」说一遍，让人确认。"""
    bias_text = {
        "symmetric": "对照组：b = 0（不破缺对称性）",
        "b-gaussian": "高斯随机 b（随机方向破缺）",
        "b-const": "常数 b（每个维度加同一个常数）",
        "attn-bQbV": "注意力里的 bQ + bV",
        "attn-full": "注意力里的 bQ + bK + bV",
        "b-learnable": "可学习的 b（交给优化器学）",
    }.get(bias_preset, bias_preset)
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


def do_training(
    speed: dict,
    bias_preset: str,
    epochs: int,
    optimizer: str,
    log_dir: str,
    dry_run: bool = False,
    want_evaluate: bool = True,
) -> int:
    """
    跑一次训练，然后尽量算一下测试集 BLEU。返回最后一次命令的退出码。

    训练失败（例如没网、显存不够）只会打印提示，不会让整个菜单崩掉。
    """
    run_name = run_name_for(speed["key"], bias_preset, optimizer)
    command = build_train_command(
        speed, bias_preset, run_name, log_dir=log_dir, epochs=epochs, optimizer=optimizer
    )
    code = run_command(command, dry_run=dry_run)
    if code != 0:
        print()
        print(f"  训练出错了（返回码 {code}）。常见原因：")
        print("    - 没连上网，下载不了数据（可以先选「玩具任务」试试，不需要网络）")
        print("    - 磁盘空间不够")
        print("  详细报错信息在上面的输出里。")
        return code

    ckpt = Path(log_dir) / run_name / "model_best.pt"
    if want_evaluate and ckpt.exists() and not dry_run:
        print()
        print("  再算一下翻译质量（BLEU，越高越好）……")
        try:
            run_command(build_evaluate_command(ckpt, speed["dataset_preset"]))
        except Exception as exc:
            print(f"  （BLEU 没算成：{exc}；不影响训练结果）")
    return 0


def do_report(log_dir: str, dry_run: bool = False) -> int:
    """出对比图和对比表格。"""
    print()
    print("  正在汇总所有结果，生成对比图……")
    code = run_command(build_report_command(log_dir), dry_run=dry_run)
    if code != 0:
        print("  （汇总失败，可能还没跑过任何实验。先去跑一个吧）")
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
    for index, bias_preset in enumerate(COMPARE_THREE, start=1):
        title(f"第 {index}/3 次，正在跑：{bias_preset}")
        do_training(speed, bias_preset, speed["epochs"], "egd", DEFAULT_LOG_DIR, dry_run)
        write_my_config(
            {
                "speed": speed["key"],
                "bias": bias_preset,
                "epochs": speed["epochs"],
                "optimizer": "egd",
                "log_dir": DEFAULT_LOG_DIR,
                "auto_report": True,
            }
        )
    do_report(DEFAULT_LOG_DIR, dry_run)


def _triple(duration: str) -> str:
    """
    把「约 1 分钟」/「约 15 秒」换算成三倍时长的说法。

    必须区分秒和分钟：如果无脑当成分钟，「约 15 秒」会变成「约 45 分钟」。
    """
    digits = "".join(ch for ch in duration if ch.isdigit())
    if not digits:
        return duration
    total = int(digits) * 3
    if "秒" in duration:
        if total >= 60:
            return f"约 {total // 60} 分钟"
        return f"约 {total} 秒"
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
    write_my_config(
        {
            "speed": speed["key"],
            "bias": bias_preset,
            "epochs": epochs,
            "optimizer": "egd",
            "log_dir": DEFAULT_LOG_DIR,
            "auto_report": True,
        }
    )
    do_report(DEFAULT_LOG_DIR, dry_run)


def action_from_config(dry_run: bool = False) -> None:
    """按 ``my_config.py`` 里的设置跑。"""
    settings = read_my_config()
    if not settings:
        print()
        print("  还没有 my_config.py，或者它读不了。先用菜单里的 1) 或 2) 跑一次，")
        print("  我就会帮你生成一个带中文说明的 my_config.py。")
        return
    try:
        speed = speed_by_key(str(settings.get("speed", "quick")))
    except KeyError:
        print(f"  my_config.py 里的 SPEED 写错了，只能是 toy / quick / long")
        return
    bias_preset = str(settings.get("bias", "b-gaussian"))
    if bias_preset not in BiasPresets:
        print(f"  my_config.py 里的 BIAS 写错了，可选：{', '.join(BiasPresets)}")
        return
    epochs = int(settings.get("epochs", speed["epochs"]))
    optimizer = str(settings.get("optimizer", "egd"))
    log_dir = str(settings.get("log_dir", DEFAULT_LOG_DIR))

    describe(speed, bias_preset, epochs, optimizer, log_dir)
    if not yes_no("  可以开始吗？", default=True):
        return
    title(f"正在跑：{bias_preset}")
    do_training(speed, bias_preset, epochs, optimizer, log_dir, dry_run)
    if settings.get("auto_report", True):
        do_report(log_dir, dry_run)


def action_help() -> None:
    """讲清楚每一项是什么意思、能改什么。"""
    title("我能改什么？")
    print(f"""
  1) 最省事：直接运行 run.py，用数字选。不用改任何文件。

  2) 想固定一套设置：编辑 {MY_CONFIG.name}（跟 run.py 在同一个文件夹），
     里面每一项都有中文说明。改完保存，然后在菜单里选 3)。

  3) 想改更细的东西（模型多大、多少数据、学习率……）：
     这些在 symbreak_transformer/config.py 里，或者用命令行：
         python main.py train --help        （会列出全部 100 多个选项）

  几个关键概念：

    bias（偏置）—— 就是这个项目要研究的东西。
      b = 0        不破缺对称性（对照组）
      高斯随机 b    给 embedding 加一个随机的固定偏置 → 破缺对称性
      常数 b        每个维度加同一个常数 → 也破缺
      注意力 bQ/bV  在 attention 内部加偏置（参考项目那种做法）
      加了这个偏置以后，attention 原来的「旋转对称性」就被打破了。
      项目就是想知道：打破它，训练会不会更好。

    「跑多久」—— 数据越多、模型越大、轮数越多，结论越可靠，但越慢。
      第一次建议用「快速」+ 「跑三种做对比」，几分钟就能看出个大概。

  结果怎么看：
      run.py 会自动生成 runs\\_compare\\compare.png（对比曲线）
      和 compare.txt（中文表格 + 结论）。验证损失(val loss)越低越好，
      BLEU 越高越好。如果三种 bias 的曲线几乎重合，说明在这个规模下
      差别不明显 —— 那就把「跑多久」换大一号再试。
""")


# ===================================================================== #
#  主菜单
# ===================================================================== #
MENU = {
    "1": "跑三种 bias 做对比（b=0 / 高斯 / 常数）   ← 第一次用选这个",
    "2": "只跑一种 bias",
    "3": f"按 {MY_CONFIG.name} 里的设置跑",
    "4": "看已有结果（出对比图，不重新训练）",
    "5": "打开结果文件夹",
    "6": "我该改什么？（说明）",
    "0": "退出",
}


def welcome() -> None:
    print()
    hr("#")
    print("#  对称性破缺 Transformer —— 一键运行")
    print("#  不用敲命令：输入数字、按回车就行")
    hr("#")


def main(argv=None) -> int:
    """菜单主循环。``argv`` 预留给测试（``--speed`` / ``--bias`` 可跳过提问）。"""
    configure_console_encoding()
    argv = list(sys.argv[1:] if argv is None else argv)

    # 非交互用法：run.py --speed quick --bias b-gaussian --yes
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
            do_report(DEFAULT_LOG_DIR)
            if yes_no("  要现在打开结果文件夹吗？", default=True):
                open_folder(Path(DEFAULT_LOG_DIR).resolve())
        elif choice == "5":
            folder = Path(DEFAULT_LOG_DIR).resolve()
            folder.mkdir(parents=True, exist_ok=True)
            open_folder(folder)
        elif choice == "6":
            action_help()
        elif choice in ("0", "q", "quit", "exit"):
            print("\n  再见。结果都在 runs\\ 里面。\n")
            return 0
        else:
            print(f"  没有 {choice!r} 这个选项，请输入 0-6")


def _non_interactive(argv: list) -> int:
    """给测试和脚本用的非交互模式：直接按给定设置跑，不做任何提问。"""
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

    speed = speed_by_key(speed_key)
    if bias_preset not in BiasPresets:
        print(f"unknown bias {bias_preset!r}; choose from {sorted(BiasPresets)}")
        return 2
    describe(speed, bias_preset, epochs or speed["epochs"], "egd", log_dir)
    if not dry_run:
        code = do_training(speed, bias_preset, epochs or speed["epochs"], "egd", log_dir)
        if code != 0:
            return code
        do_report(log_dir)
    else:
        print()
        print("  dry-run 命令：python main.py " + " ".join(
            build_train_command(
                speed, bias_preset, run_name_for(speed["key"], bias_preset),
                log_dir=log_dir, epochs=epochs,
            )
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
