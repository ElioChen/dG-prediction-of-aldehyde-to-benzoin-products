#!/usr/bin/env python3
"""
Conceptual schematic explaining the Bemis–Murcko SCAFFOLD SPLIT used in screen-v6 /
homo-benzoin honest-generalization tests. Emitted as TWO separate single-language
figures (English and 中文) — run with no args to write both.

It contrasts:
  • random split   — molecules of the same core land in BOTH train and test → leakage
  • scaffold split  — whole scaffolds go entirely to train OR test → novel-core test
Numbers (aromatic n≈146,741) from exp_scaffold_split.py / REPORT_screen_v6_models.

One figure per file; fresh filenames so prior figures are never overwritten.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Circle

# CJK-capable font so the Chinese figure renders (DejaVu has no Han glyphs).
_ZH = "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"
if Path(_ZH).exists():
    font_manager.fontManager.addfont(_ZH)
    _zh_name = font_manager.FontProperties(fname=_ZH).get_name()
    plt.rcParams["font.family"] = ["DejaVu Sans", _zh_name]
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parent.parent
OUT = {"en": REPO / "figs" / "scaffold_split_explainer_en_20260629.png",
       "zh": REPO / "figs" / "scaffold_split_explainer_zh_20260629.png"}

SCAF = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
NAMES = ["A", "B", "C", "D", "E", "F"]

# all display strings per language
TXT = {
    "en": dict(
        suptitle="Scaffold split — honest generalization to novel cores",
        legend="Bemis–Murcko scaffolds (ring core + linkers):",
        rand_title="Random split", rand_sub="shuffle rows",
        scaf_title="Scaffold split", scaf_sub="whole Bemis–Murcko cores",
        train="TRAIN", test="TEST", arrow="group by core",
        leak="✗ same core in train & test (leakage)",
        ok="✓ test cores never seen in training",
        result=("Result (aromatic n≈146,741):  random MAE 2.13 / R² 0.68   →   "
                 "scaffold MAE 2.18 / R² 0.61   →   rare-core MAE 2.58 / R² 0.54   "
                 "= generalization gap"),
    ),
    "zh": dict(
        suptitle="骨架划分 —— 对新骨架的诚实泛化检验",
        legend="Bemis–Murcko 骨架（环核 + 连接子）：",
        rand_title="随机划分", rand_sub="行随机打乱",
        scaf_title="骨架划分", scaf_sub="整骨架成组",
        train="训练", test="测试", arrow="按骨架成组",
        leak="✗ 同骨架同时在训练与测试（泄漏）",
        ok="✓ 测试骨架训练时从未见过",
        result=("结果（芳香 n≈146,741）：  随机 MAE 2.13 / R² 0.68   →   "
                 "骨架 MAE 2.18 / R² 0.61   →   罕见骨架 MAE 2.58 / R² 0.54   "
                 "= 泛化差距"),
    ),
}


def draw_panel(ax, x0, title, sub, members_train, members_test, leaky, t):
    ax.add_patch(FancyBboxPatch((x0, 1.55), 3.6, 1.7, boxstyle="round,pad=0.04",
                 fc="#eef3f8", ec="#5b7fa6", lw=1.4))
    ax.add_patch(FancyBboxPatch((x0, 0.25), 3.6, 1.05, boxstyle="round,pad=0.04",
                 fc="#fdeeee" if leaky else "#eef8ee",
                 ec="#c0392b" if leaky else "#2e7d32", lw=1.6))
    ax.text(x0 + 1.8, 3.34, title, ha="center", fontsize=13, fontweight="bold")
    ax.text(x0 + 1.8, 3.12, sub, ha="center", fontsize=9, color="#555")
    ax.text(x0 + 0.12, 3.02, t["train"], fontsize=9, color="#34618e", fontweight="bold")
    ax.text(x0 + 0.12, 1.12, t["test"], fontsize=9,
            color="#c0392b" if leaky else "#2e7d32", fontweight="bold")

    def grid(members, bx, by, ncol, dx=0.32, dy=0.32):
        for k, s in enumerate(members):
            cx, cy = bx + (k % ncol) * dx, by - (k // ncol) * dy
            ax.add_patch(Circle((cx, cy), 0.12, fc=SCAF[s], ec="white", lw=1.0, zorder=3))
    grid(members_train, x0 + 0.35, 2.82, 9)
    grid(members_test, x0 + 0.35, 0.95, 9)

    if leaky:
        train_set = set(members_train)
        for k, s in enumerate(members_test):
            if s in train_set:
                cx = x0 + 0.35 + (k % 9) * 0.32
                ax.text(cx, 0.62, "✗", ha="center", fontsize=12, color="#c0392b",
                        fontweight="bold")
        ax.text(x0 + 1.8, 0.04, t["leak"], ha="center", fontsize=9,
                color="#c0392b", fontweight="bold")
    else:
        ax.text(x0 + 1.8, 0.04, t["ok"], ha="center", fontsize=9,
                color="#2e7d32", fontweight="bold")


def render(lang: str) -> Path:
    t = TXT[lang]
    fig, ax = plt.subplots(figsize=(12.5, 6.7))
    ax.set_xlim(0, 13); ax.set_ylim(-0.7, 4.6); ax.axis("off")
    fig.suptitle(t["suptitle"], fontsize=15, fontweight="bold", y=0.98)

    tr_rand = [0,1,2,3,4,5,0,1,2,3,4,5,0,1,2,3]
    te_rand = [0,2,4,1]
    draw_panel(ax, 0.4, t["rand_title"], t["rand_sub"], tr_rand, te_rand, True, t)

    tr_scaf = [0,0,0,1,1,1,2,2,2,3,3,3]
    te_scaf = [4,4,4,5,5,5]
    draw_panel(ax, 7.0, t["scaf_title"], t["scaf_sub"], tr_scaf, te_scaf, False, t)

    ax.annotate("", xy=(6.9, 1.9), xytext=(4.1, 1.9),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=2.0))
    ax.text(5.5, 2.08, t["arrow"], ha="center", fontsize=9.5,
            color="#333", fontweight="bold")

    for i, (c, nm) in enumerate(zip(SCAF, NAMES)):
        ax.add_patch(Circle((0.7 + i * 0.62, 4.16), 0.11, fc=c, ec="white", lw=1.0))
        ax.text(0.7 + i * 0.62 + 0.17, 4.12, nm, fontsize=9, va="center")
    ax.text(0.45, 4.42, t["legend"], fontsize=9.5, color="#333")

    ax.text(6.5, -0.5, t["result"], ha="center", fontsize=10, color="#1a1a1a",
            bbox=dict(boxstyle="round", fc="#f4f4f4", ec="#aaa"))

    out = OUT[lang]
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "zh", "both"], default="both")
    args = ap.parse_args()
    langs = ["en", "zh"] if args.lang == "both" else [args.lang]
    for lang in langs:
        print(f"Wrote {render(lang)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
