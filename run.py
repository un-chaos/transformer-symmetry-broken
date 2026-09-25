#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
唯一入口 —— 双击 ``run.bat``（或运行 ``python run.py``），然后输入数字。

不需要记命令、不需要改任何文件、不需要看懂代码：

    双击 run.bat  →  输入 1（三种 bias 做对比）  →  回车

跑完自动生成损失曲线、对比图和中文报告，并帮你打开结果文件夹。

想换设置再跑（模型大小、学习率、数据、bias、任意超参……）：

    菜单 3) 里直接按数字改。``train.py`` 支持多少参数，那里就能改多少 ——
    没有任何参数是写死的；改过的项会被程序自动记住，下次打开还在。

本文件是**唯一**的入口；训练、评估、汇总、下载、画图都只是它内部调用的实现，
不需要（也不应该）单独去用。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from symbreak_transformer.utils import configure_console_encoding  # noqa: E402

SCRIPTS = ROOT / "scripts"
TRAIN_SCRIPT = SCRIPTS / "train.py"
EVALUATE_SCRIPT = SCRIPTS / "evaluate.py"
REPORT_SCRIPT = SCRIPTS / "report.py"
DOWNLOAD_SCRIPT = SCRIPTS / "download_data.py"
ANALYZE_SCRIPT = SCRIPTS / "analyze_bias.py"

DEFAULT_LOG_DIR = "runs"

#: 自动记住用户设置的文件（程序自己读写，**不需要**手工编辑）。
SETTINGS_FILE = ROOT / "run_settings.json"


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
        "label": "玩具任务 —— 不用下载数据、不用联网，每次约 15 秒  ← 第一次用选这个",
        "minutes": "约 15 秒",
        "model": "smoke",
        "dataset_preset": "synthetic-copy",
        "epochs": 1,
        "max_steps": 60,
        "log_every": 5,
        "valid_every": 10,
    },
    "2": {
        "key": "quick",
        "label": "快速看看 —— 2000 句真实翻译数据，每次约 1 分钟（要联网）",
        "minutes": "约 1 分钟",
        "model": "tiny",
        "dataset_preset": "multi30k-quick",
        "epochs": 2,
        "max_steps": 0,
        "log_every": 4,
        "valid_every": 8,
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
    },
    "4": {
        "key": "corpus",
        "label": "大规模语料 —— FineWeb-Edu 10B（菜单 7 先下载；CPU 上很慢）",
        "minutes": "取决于步数，建议先用菜单 3) 把「最多训练多少步」改小",
        "model": "tiny",
        "dataset_preset": "fineweb-quick",
        "epochs": 1,
        "max_steps": 200,
        "log_every": 10,
        "valid_every": 20,
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

_BIAS_TEXT = {
    "symmetric": "对照组：b = 0（不破缺对称性）",
    "b-gaussian": "高斯随机 b（随机方向破缺）",
    "b-const": "常数 b（每个维度加同一个常数）",
    "attn-bQbV": "注意力里的 bQ + bV",
    "attn-full": "注意力里的 bQ + bK + bV",
    "b-learnable": "可学习的 b（交给优化器学）",
}

#: 菜单 3) 的「常用设置」：按这个顺序显示（值是中文说明）。
COMMON_FLAGS: dict = {
    "--dataset_preset": "数据集：synthetic-*（玩具，不用联网）/ multi30k-*（翻译）/ fineweb-*（10B 语料）",
    "--objective": "任务类型：translation=翻译 / denoising=去噪（原始文本，不用翻译对照）",
    "--model": "模型大小：smoke / tiny / small / base / large（越大越强、越慢）",
    "--bias_preset": "对称性破缺设置：symmetric / b-gaussian / b-const / attn-bQbV / attn-full / ...",
    "--optimizer": "优化器：egd（本项目的能量守恒下降法）/ adamw / sgdm",
    "--egd_lr": "EGD 学习率（和 F0 配套，改一个通常要一起调）",
    "--egd_F0": "EGD 的 loss 偏移，必须低于能达到的最小 loss（默认 -1 对交叉熵永远安全）",
    "--batch_size": "每批多少条数据（内存不够就调小）",
    "--epochs": "训练几轮（越大越慢、一般也越好）",
    "--max_steps": "最多训练多少步（0 = 由轮数决定；快速试跑就写个小数字）",
    "--ctx": "上下文长度（一句话最多多少个 token）",
    "--n_embd": "隐藏维度（不写就用模型预设的值）",
    "--n_head": "注意力头数（必须能整除隐藏维度）",
    "--seed": "随机种子（固定它，两次实验才有可比性）",
    "--log_dir": "结果保存到哪个文件夹",
    "--name": "这次实验的名字（留空 = 自动起名）",
}

#: 不放进「改参数」界面的开关：纯信息性的，或者会破坏本项目的硬性要求。
#: * ``--no_plot``：损失曲线必须落盘，不允许通过菜单关掉它；
#: * ``-h/--help``、``--list_models``：只是打印信息，不是"设置"。
HIDDEN_FLAGS = {"-h", "--help", "--no_plot", "--list_models"}


# ===================================================================== #
#  当前设置（由「跑多久 / 哪种 bias」的问题 + 用户手改的部分组成）
# ===================================================================== #
#: 菜单问题的答案（「跑多久」/「哪种 bias」/「哪个优化器」）。
STATE: dict = {"speed": "1", "bias": "b-gaussian", "optimizer": "egd"}

#: 用户在菜单 3) 里手改过的参数（``--flag`` -> 值）。手改的优先级最高。
CUSTOM: dict = {}


def load_settings() -> None:
    """读回上次的设置（文件不存在或坏了都当没有，绝不因此崩掉）。"""
    try:
        data = json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 自动生成的文件，坏了就忽略
        return
    if not isinstance(data, dict):
        return
    for key in ("speed", "bias", "optimizer"):
        value = data.get(key)
        if isinstance(value, str) and value:
            STATE[key] = value
    custom = data.get("custom")
    if isinstance(custom, dict):
        CUSTOM.clear()
        CUSTOM.update({str(k): v for k, v in custom.items()})
    # 兼容：设置里对应的编号可能已经不存在了（比如换了版本）。
    if STATE["speed"] not in SPEEDS:
        STATE["speed"] = "1"
    if STATE["bias"] not in {name for name, _ in BIASES.values()}:
        STATE["bias"] = "b-gaussian"
    if STATE["optimizer"] not in ("egd", "adamw", "sgdm"):
        STATE["optimizer"] = "egd"


def save_settings() -> None:
    """把当前设置写到 ``run_settings.json``（失败也不影响使用）。"""
    payload = {"speed": STATE["speed"], "bias": STATE["bias"],
               "optimizer": STATE["optimizer"], "custom": CUSTOM}
    try:
        SETTINGS_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:  # noqa: BLE001 - 记不住设置不算错误
        pass


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
        options: ``{key: 显示文字}``。
        default: 直接回车时用的 key（``None`` 表示必须明确输入）。
    """
    if not options:
        return ""
    keys = list(options)
    print()
    print(prompt)
    for key, text in options.items():
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


def run_name_for(speed_key: str, bias_preset: str, optimizer: str = "egd") -> str:
    """给这次运行起一个看得懂的名字，例如 ``quick-bgaussian``。"""
    short = bias_preset.replace("b-", "b").replace("-", "")
    name = f"{speed_key}-{short}"
    if optimizer != "egd":
        name += f"-{optimizer}"
    return name


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


# ===================================================================== #
#  参数表 / 命令行拼装（唯一事实来源：scripts/train.py 的 argparse）
# ===================================================================== #
_train_module_cache = None


def train_parser() -> argparse.ArgumentParser:
    """``scripts/train.py`` 的 argparse 对象（缓存），用来知道有哪些参数。"""
    global _train_module_cache
    if _train_module_cache is None:
        spec = importlib.util.spec_from_file_location("_symbreak_train_script", TRAIN_SCRIPT)
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


def _is_switch(action) -> bool:
    """这个参数是不是「开关」（不带值，出现即生效）？"""
    return isinstance(
        action, (argparse._StoreTrueAction, argparse._StoreFalseAction)
    ) or action.nargs == 0


def default_of(action):
    """参数的默认值（``SUPPRESS`` 表示"看 train.py 自己的默认"，记为 None）。"""
    return None if action.default is argparse.SUPPRESS else action.default


def option_groups(include_hidden: bool = False) -> list:
    """
    把 ``train.py`` 的全部参数按分组列出来。

    Returns:
        ``[(组名, [(flag, action)])]``。
    """
    parser = train_parser()
    groups: list = []
    for group in parser._action_groups:
        rows = []
        for action in group._group_actions:
            if not action.option_strings:
                continue
            flag = action.option_strings[0]
            if not include_hidden and flag in HIDDEN_FLAGS:
                continue
            rows.append((flag, action))
        if rows:
            groups.append((group.title or "其他", rows))
    return groups


def all_flags(include_hidden: bool = True) -> list:
    """``train.py`` 认识的全部 ``--flag``（默认含隐藏项）。"""
    flags: list = []
    for _, rows in option_groups(include_hidden=True):
        for flag, _ in rows:
            flags.append(flag)
    return flags


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
        if _is_switch(action):
            if bool(value):
                argv.append(flag)
        else:
            if value is None:
                continue
            argv += [flag, str(value)]
    return argv


def speed_flags(speed: dict, bias_preset: str, epochs: int | None = None,
                optimizer: str = "egd", run_name: str | None = None,
                log_dir: str = DEFAULT_LOG_DIR) -> dict:
    """「跑多久」那一组参数（用户手改的部分还没叠加上去）。"""
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
    return flags


def current_speed() -> dict:
    """当前选中的「跑多久」（``STATE["speed"]`` 存的是菜单编号 1-4）。"""
    return SPEEDS[STATE["speed"]]


def effective_flags() -> dict:
    """
    这次运行真正会用到的参数 = 「跑多久 / 哪种 bias」那一组 + 用户手改的部分。

    手改的部分**最后**叠加，所以它永远优先 —— 这正是"我改过的设置说话算数"。
    """
    flags = speed_flags(current_speed(), STATE["bias"], optimizer=STATE["optimizer"])
    flags.update(CUSTOM)
    return flags


def build_train_command(flags: dict) -> list:
    """
    拼出训练命令的参数表。

    注意**永远不加** ``--no_plot``：损失曲线必须落盘（每个 run 目录里一份
    ``training_curve.png``），这是这个项目的硬要求。
    """
    return flags_to_argv(flags)


def build_evaluate_command(ckpt: Path, dataset_preset: str, max_samples: int = 100) -> list:
    """拼出评估命令的参数表（跑完算一下测试集 loss 和 BLEU）。"""
    return [
        "--ckpt", str(ckpt),
        "--split", "test",
        "--dataset_preset", dataset_preset,
        "--max_samples", str(max_samples),
    ]


def build_report_command(log_dir: str = DEFAULT_LOG_DIR) -> list:
    """拼出汇总（对比图 + 报告）命令的参数表。"""
    return ["--log_dir", log_dir]


def build_analyze_command(run_dir: Path, dataset_preset: str,
                          max_samples: int = 100) -> list:
    """拼出「bias 到底改变了什么」分析命令的参数表。"""
    return [
        "--ckpt", str(Path(run_dir) / "model_best.pt"),
        "--split", "test",
        "--dataset_preset", str(dataset_preset),
        "--max_samples", str(max_samples),
        "--out", str(Path(run_dir) / "bias_analysis.json"),
    ]


def run_command(script: Path, args: list, dry_run: bool = False) -> int:
    """执行一个内部脚本，输出直接显示在屏幕上。"""
    argv = [sys.executable, str(script)] + [str(a) for a in args]
    print()
    print("  正在执行：" + " ".join(argv[1:]))
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
#  菜单 3)：改参数（train.py 支持多少，这里就能改多少）
# ===================================================================== #
def _render(value) -> str:
    """把参数值渲染成人看的文字。"""
    if value is None:
        return "（默认）"
    if value is True:
        return "开"
    if value is False:
        return "关"
    return str(value)


def _row_text(flag: str, action) -> str:
    """菜单里一行：``--n_embd = 128  ★``。"""
    value = effective_flags().get(flag)
    if value is None:
        value = default_of(action)
    star = "  ★" if flag in CUSTOM else ""
    return f"{flag:<24} = {_render(value)}{star}"


def _short_help(action) -> str:
    text = (action.help or "").strip().replace("\n", " ")
    return text[:70]


def _coerce(raw: str, action):
    """把用户敲的文字变成参数值，顺便检查类型。"""
    kind = action.type
    if kind is int:
        return int(raw)
    if kind is float:
        return float(raw)
    return raw


def ask_value(flag: str, action) -> None:
    """
    改一个参数。回车 = 恢复默认（也就是"不手改这一项"）。

    有固定选项的参数（数据集、模型大小、优化器……）直接给编号选；
    开关类参数问 y/n；其余的自己输入值。
    """
    print()
    print(f"  {flag}   {_short_help(action)}")
    print(f"  现在：{_render(effective_flags().get(flag))}   "
          f"（train.py 的默认值：{_render(default_of(action))}）")

    if action.choices:
        choices = list(action.choices)
        labels = {str(i + 1): str(c) for i, c in enumerate(choices)}
        key = ask("  选一个（回车 = 保持现在的）：", labels, default="")
        if not key:
            return
        value = choices[int(key) - 1]
    elif _is_switch(action):
        want = yes_no("  要加上这个开关吗？", default=bool(CUSTOM.get(flag)))
        value = True if want else None
    else:
        hint = "（直接回车 = 恢复默认）"
        raw = input(f"  输入新值 {hint}：").strip()
        if not raw:
            value = None
        else:
            try:
                value = _coerce(raw, action)
            except ValueError:
                print(f"  {raw!r} 不是合法的值（需要 {getattr(action.type, '__name__', '文本')}），没改。")
                return

    if value is None:
        CUSTOM.pop(flag, None)
        print(f"  已把 {flag} 恢复成默认。")
    else:
        CUSTOM[flag] = value
        print(f"  已把 {flag} 改成 {_render(value)}。")
    save_settings()


def _common_rows() -> list:
    """「常用设置」界面的行（只保留 train.py 真的有的参数）。"""
    index = _action_index()
    return [(flag, index[flag]) for flag in COMMON_FLAGS if flag in index]


def _edit_all() -> None:
    """「全部参数」界面：把 train.py 的每一个参数都列出来，按编号改。"""
    rows: list = []
    for group_title, entries in option_groups():
        print(f"\n  --- {group_title} ---")
        for flag, action in entries:
            rows.append((flag, action))
            print(f"  {len(rows):>3}) {_row_text(flag, action)}")
    if not rows:
        return
    raw = input("\n改哪一项？（回车=返回）：").strip()
    if raw == "0":
        return
    if raw.isdigit() and 1 <= int(raw) <= len(rows):
        flag, action = rows[int(raw) - 1]
        ask_value(flag, action)


def action_settings() -> None:
    """菜单 3)：修改训练设置。改过的项自动记住，不需要编辑任何文件。"""
    while True:
        title("修改训练设置")
        rows = _common_rows()
        print("  带 ★ 的是你手动改过的项，它们的优先级最高（回车进去可以恢复默认）。")
        print("  输入编号就能改；直接回车返回。")
        print()
        for index, (flag, action) in enumerate(rows, start=1):
            print(f"  {index:>2}) {_row_text(flag, action)}")
        print()
        print("   a) 全部参数（train.py 支持的所有设置）")
        print("   d) 全部恢复默认")
        raw = input("\n改哪一项？（回车=返回）：").strip().lower()
        if raw in ("", "0", "b", "q"):
            return
        if raw == "a":
            _edit_all()
            continue
        if raw == "d":
            CUSTOM.clear()
            save_settings()
            print("  已全部恢复默认。")
            continue
        if raw.isdigit() and 1 <= int(raw) <= len(rows):
            flag, action = rows[int(raw) - 1]
            ask_value(flag, action)
        else:
            print(f"  没有 {raw!r} 这个选项。")


def action_show_settings() -> None:
    """菜单 4)：把这次会用到的设置原原本本列出来。"""
    title("当前设置")
    flags = effective_flags()
    speed = current_speed()
    print(f"  数据      ：{speed['dataset_preset']}")
    print(f"  模型大小  ：{speed['model']}")
    print(f"  对称性破缺：{_BIAS_TEXT.get(STATE['bias'], STATE['bias'])}")
    print(f"  优化器    ：{STATE['optimizer']}")
    print(f"  预计耗时  ：{speed['minutes']}")
    print()
    print(f"  实际传给训练的 {len(flags)} 个参数：")
    for flag, value in flags.items():
        star = "  ★" if flag in CUSTOM else ""
        print(f"    {flag:<26} {_render(value)}{star}")
    print()
    print("  想改就回菜单 3)。")


# ===================================================================== #
#  各个菜单动作
# ===================================================================== #
def describe(flags: dict) -> None:
    """用大白话把「接下来要干什么」说一遍，让人确认。"""
    print()
    print("  接下来会这样做：")
    speed = current_speed()
    print(f"    数据      ：{flags.get('--dataset_preset')}（{speed['label'].split('——')[-1].strip()}）")
    print(f"    模型大小  ：{flags.get('--model')}")
    print(f"    对称性破缺：{_BIAS_TEXT.get(STATE['bias'], STATE['bias'])}")
    print(f"    训练轮数  ：{flags.get('--epochs')}")
    print(f"    优化器    ：{flags.get('--optimizer')}")
    if CUSTOM:
        print(f"    你手改过的：{'、'.join(sorted(CUSTOM))}")
    print(f"    预计耗时  ：{speed['minutes']}")
    print(f"    结果放在  ：{Path(str(flags.get('--log_dir', DEFAULT_LOG_DIR))).resolve()}")
    print("    （损失曲线会存成 <结果目录>/training_curve.png）")


def do_training(flags: dict, dry_run: bool = False, want_evaluate: bool = True) -> int:
    """
    跑一次训练（+ 尽量算一下测试集 BLEU），返回退出码。

    训练失败只会打印提示，不会让整个菜单崩掉。
    """
    log_dir = str(flags.get("--log_dir", DEFAULT_LOG_DIR))
    run_name = str(flags.get("--name") or "run")
    code = run_command(TRAIN_SCRIPT, build_train_command(flags), dry_run=dry_run)
    if code != 0:
        print()
        print(f"  训练出错了（返回码 {code}）。常见原因：")
        print("    - 没连上网，下载不了数据（菜单 1) 的「玩具任务」不需要网络）")
        print("    - 大语料还没下载完（回菜单 7) 可以下载 / 看进度）")
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
            run_command(
                EVALUATE_SCRIPT,
                build_evaluate_command(ckpt, str(flags.get("--dataset_preset", ""))),
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  （没算成：{exc}；不影响训练结果）")
    return 0


def do_report(log_dir: str, dry_run: bool = False, open_when_done: bool = False) -> int:
    """出对比图和对比表格。"""
    print()
    print("  正在汇总所有结果，生成对比图……")
    code = run_command(REPORT_SCRIPT, build_report_command(log_dir), dry_run=dry_run)
    if code != 0:
        print("  （汇总失败，可能还没跑过任何实验。先去跑一个吧）")
    elif open_when_done and not dry_run:
        if yes_no("  要现在打开结果文件夹吗？", default=True):
            open_folder(Path(log_dir).resolve())
    return code


def action_compare_three(dry_run: bool = False) -> None:
    """跑三种 bias 各一次，然后出对比图 —— 第一次用推荐这个。"""
    key = ask("先选「跑多久」：", {k: v["label"] for k, v in SPEEDS.items()}, default=STATE["speed"])
    STATE["speed"] = key
    save_settings()
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
        STATE["bias"] = bias_preset
        do_training(effective_flags(), dry_run)
    STATE["bias"] = COMPARE_THREE[-1]
    save_settings()
    do_report(DEFAULT_LOG_DIR, dry_run, open_when_done=True)


def action_single(dry_run: bool = False) -> None:
    """只跑一种 bias。"""
    key = ask("先选「跑多久」：", {k: v["label"] for k, v in SPEEDS.items()}, default=STATE["speed"])
    STATE["speed"] = key
    speed = SPEEDS[key]
    bias_key = ask("用哪种 bias？", {k: v[1] for k, v in BIASES.items()},
                   default=next((k for k, v in BIASES.items() if v[0] == STATE["bias"]), "2"))
    STATE["bias"] = BIASES[bias_key][0]
    save_settings()
    flags = effective_flags()
    describe(flags)
    if not yes_no("  可以开始吗？", default=True):
        return
    title(f"正在跑：{STATE['bias']}")
    do_training(flags, dry_run)
    do_report(DEFAULT_LOG_DIR, dry_run, open_when_done=True)


def finished_runs(log_dir: str = DEFAULT_LOG_DIR) -> list:
    """已经训练完的运行目录（里面有 model_best.pt 的），按名字排序。"""
    root = Path(log_dir)
    if not root.is_dir():
        return []
    return sorted(
        (path for path in root.iterdir() if (path / "model_best.pt").is_file()),
        key=lambda path: path.name,
    )


def dataset_preset_of(run_dir: Path) -> str:
    """从那次运行的 ``args.json`` 里读回数据集；读不到就用当前的设置。"""
    fallback = str(effective_flags().get("--dataset_preset") or "multi30k-quick")
    try:
        payload = json.loads((Path(run_dir) / "args.json").read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 缺文件不算错误
        return fallback
    if not isinstance(payload, dict):
        return fallback
    return str(payload.get("dataset_preset") or fallback)


def action_analyze_bias(dry_run: bool = False) -> None:
    """菜单 6)：量一下这个 bias 到底把模型算的东西改变了多少。"""
    runs = finished_runs()
    if not runs:
        print()
        print("  还没有训练好的模型。先回菜单 1) 或 2) 跑一次")
        print("  （选「玩具任务」的话，不用联网、十几秒就跑完）。")
        return
    if len(runs) == 1:
        run_dir = runs[0]
    else:
        options = {str(index): path.name for index, path in enumerate(runs, 1)}
        run_dir = runs[int(ask("  分析哪一个？", options, default="1")) - 1]

    print()
    print(f"  将分析「{run_dir.name}」：把这个模型里的 bias 去掉，")
    print("  看它的输出改变了多少 —— 改变越大，这个 bias 对模型越重要。")
    if not yes_no("  可以开始吗？", default=True):
        return
    title(f"正在分析：{run_dir.name}")
    code = run_command(
        ANALYZE_SCRIPT, build_analyze_command(run_dir, dataset_preset_of(run_dir)),
        dry_run=dry_run,
    )
    if code == 0 and not dry_run:
        out = run_dir / "bias_analysis.json"
        if out.exists():
            print()
            print(f"  分析结果已保存到：{out.resolve()}")


def action_download(dry_run: bool = False) -> None:
    """下载大规模语料（在新窗口里跑，方便看到进度）。"""
    print()
    print("  将下载 FineWeb-Edu sample/10BT（约 10B 词、14 个分片、约 28.5 GB）。")
    print("  这是**断点续传**的：中途断了再跑一次就会接着下，不会重复下载。")
    run_command(DOWNLOAD_SCRIPT, ["--status"])
    if not yes_no("  现在开始下载吗？（会另开一个窗口显示进度）", default=True):
        return
    if dry_run:
        print("  （dry-run：不启动）")
        return
    try:
        if sys.platform.startswith("win"):
            subprocess.Popen(
                ["cmd.exe", "/k", f'cd /d "{ROOT}" && "{sys.executable}" "{DOWNLOAD_SCRIPT}"'],
                cwd=str(ROOT),
            )
            print("  已在新窗口里开始下载。那个窗口会实时显示进度，")
            print("  下完它自己会停住（按任意键关闭）。")
        else:
            run_command(DOWNLOAD_SCRIPT, [])
    except Exception as exc:  # noqa: BLE001
        print(f"  开新窗口失败（{exc}），改为在当前窗口下载。")
        run_command(DOWNLOAD_SCRIPT, [])


def action_help() -> None:
    """讲清楚每一项是什么意思、能改什么。"""
    title("帮助：每个选项都是什么意思")
    print("""
  怎么用：只运行这个程序（双击 run.bat），然后输入数字。
  没有别的入口，也不需要改任何文件。

  菜单每一项做什么：
    1) 三种 bias 做对比 —— 一次跑三次（b=0 / 高斯 / 常数），最后出对比图。
       第一次用就选这个，选「玩具任务」完全不用联网。
    2) 只跑一种 bias —— 想单独看某一种的时候用。
    3) 修改训练设置 —— 模型大小、学习率、数据、任意超参都在这里改。
       带 ★ 的是你改过的项；回车进去可以恢复默认。改过就自动记住。
    4) 查看当前设置 —— 把这次真正会用到的参数原原本本列出来。
    5) 看已有结果 —— 不重新训练，只把 runs\\ 里的结果重新画成对比图。
    6) 分析 bias —— 拿一个训练好的模型，把里面的 bias 去掉，看它的输出改变了
       多少。改变越大，说明这个 bias 对模型越重要；这是「对称性破缺到底有没有
       用」的量化答案。
    7) 打开结果文件夹。
    8) 下载大规模语料（FineWeb-Edu 10B，约 28.5 GB，可断点续传）。
    9) 这个帮助。
    0) 退出。

  几个关键概念：

    bias（偏置）—— 就是这个项目要研究的东西。
      b = 0        不破缺对称性（对照组）
      高斯随机 b    给 embedding 加一个随机的固定偏置 → 破缺对称性
      常数 b        每个维度加同一个常数 → 也破缺
      注意力 bQ/bV  在 attention 内部加偏置（参考项目那种做法）
      加了这个偏置以后，attention 原来的「旋转对称性」就被打破了。

    数据集 / 任务类型 —— 数据从哪来、训练什么。
      synthetic-*  程序自己生成的玩具任务：不用下载、不用联网，先跑通流程
      multi30k-*   翻译（translation）：平行语料，主指标是 BLEU
      fineweb-*    去噪（denoising）：10B 词英文网页语料，不需要翻译标注
                   先回菜单 7) 下载

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


MENU = {
    "1": "开始训练：三种 bias 做对比（b=0 / 高斯 / 常数）   ← 第一次用选这个",
    "2": "开始训练：只跑一种 bias",
    "3": "修改训练设置（模型大小 / 学习率 / 数据 / 全部参数）",
    "4": "查看当前设置",
    "5": "看已有结果（重新出对比图和报告）",
    "6": "分析 bias 到底改变了什么（挑一个训练好的模型）",
    "7": "打开结果文件夹",
    "8": "下载大规模语料（FineWeb-Edu 10B，约 28.5 GB，可断点续传）",
    "9": "帮助：每个选项是什么意思",
    "0": "退出",
}


def welcome() -> None:
    print()
    hr("#")
    print("#  对称性破缺 Transformer —— 一键运行")
    print("#  不用敲命令：输入数字、按回车就行")
    hr("#")
    loaded = [f"{k}={v}" for k, v in (("跑多久", STATE["speed"]), ("bias", STATE["bias"]))] if CUSTOM else []
    if loaded:
        print(f"  （已记住你上次的设置：{'、'.join(loaded)}"
              f"{'，还有你手改过的参数' if CUSTOM else ''}）")


def main() -> int:
    """菜单主循环 —— 这是本程序**唯一**的用法。"""
    configure_console_encoding()
    load_settings()
    if len(sys.argv) > 1:
        print()
        print("  这个程序不需要参数：直接回车、按数字选就行。")
    welcome()
    while True:
        try:
            print()
            for key, text in MENU.items():
                print(f"  {key}) {text}")
            choice = input("\n请输入数字后回车：").strip()

            if choice == "1":
                action_compare_three()
            elif choice == "2":
                action_single()
            elif choice == "3":
                action_settings()
            elif choice == "4":
                action_show_settings()
            elif choice == "5":
                do_report(DEFAULT_LOG_DIR, open_when_done=True)
            elif choice == "6":
                action_analyze_bias()
            elif choice == "7":
                folder = Path(DEFAULT_LOG_DIR).resolve()
                folder.mkdir(parents=True, exist_ok=True)
                open_folder(folder)
            elif choice == "8":
                action_download()
            elif choice == "9":
                action_help()
            elif choice in ("0", "q", "quit", "exit"):
                print("\n  再见。结果都在 runs\\ 里面。\n")
                return 0
            else:
                print(f"  没有 {choice!r} 这个选项，请输入 0-9")
        except KeyboardInterrupt:
            print("\n  （已取消，回到菜单）")
        except EOFError:
            # 输入流没了（终端被关掉、或用管道喂输入）：优雅退出，不要抛栈。
            print("\n  输入结束了，退出。结果都在 runs\\ 里面。\n")
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
