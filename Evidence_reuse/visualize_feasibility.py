import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.plot_style import BAR_WIDTH, PALETTE, figsize, set_academic_style


COLOR_PRIMARY = PALETTE["line_blue_high"]
COLOR_SECONDARY = PALETTE["line_green_high"]
COLOR_BAR0 = PALETTE["bar_blue_low"]
COLOR_BAR1 = PALETTE["bar_green_low"]
COLOR_GRAY_LIGHT = PALETTE["bar_neutral"]


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_evidence_reuse_distribution(data: dict, out_dir: Path):
    reuse_data = data["visualization_data"]["full_evidence_reuse"]
    reuse_counts = np.array(reuse_data["reuse_counts"], dtype=float)
    zero_ratio = float(reuse_data.get("zero_reuse_ratio", 0.0))

    fig, ax = plt.subplots(figsize=figsize(8.6))
    ax.hist(
        reuse_counts,
        bins=30,
        color=COLOR_BAR0,
        alpha=0.45,
        edgecolor="#4a4a4a",
        linewidth=0.7,
        log=True,
    )

    handles = [
        Line2D([0], [0], color=COLOR_BAR0, lw=8, alpha=0.45, label="证据复用次数分布")
    ]

    ax.set_xlabel("每条证据命中的声明数")
    ax.set_ylabel("证据数量（对数尺度）")
    ax.set_title("证据复用分布（全证据视角）")
    ax.legend(handles=handles, loc="upper right", frameon=False)

    save_fig(fig, out_dir / "fig_feasibility_reuse_distribution.png")


def plot_similarity_by_reuse_group(data: dict, out_dir: Path):
    sim_groups = data["visualization_data"]["similarity_by_reuse_group"]["similarity_groups_iqr_filtered"]
    ordered_keys = ["0-3", "3-10", "10-20", ">=20"]
    labels = ["0-2次", "3-9次", "10-19次", "≥20次"]

    values = [sim_groups.get(k, []) for k in ordered_keys]

    fig, ax = plt.subplots(figsize=figsize(8.6))
    bp = ax.boxplot(
        values,
        tick_labels=labels,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#000000", "linewidth": 1.4},
        boxprops={"linewidth": 1.0, "edgecolor": "#000000"},
        whiskerprops={"linewidth": 1.0, "color": "#000000"},
        capprops={"linewidth": 1.0, "color": "#000000"},
    )

    for patch in bp["boxes"]:
        patch.set_facecolor(COLOR_BAR0)
        patch.set_alpha(0.35)

    ax.set_xlabel("证据复用分组")
    ax.set_ylabel("平均相似度")
    ax.set_title("不同复用强度下的相似度分布")

    save_fig(fig, out_dir / "fig_feasibility_similarity_by_reuse_group.png")


def main():
    parser = argparse.ArgumentParser(description="可行性分析结果可视化（两张图）")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/feasibility/feasibility_metrics.json"),
        help="可行性JSON输入路径",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/feasibility/figures"),
        help="图片输出目录",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"未找到输入文件: {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_academic_style()

    payload = load_json(args.input)
    plot_evidence_reuse_distribution(payload, args.out_dir)
    plot_similarity_by_reuse_group(payload, args.out_dir)

    print(f"[Export] Feasibility figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
