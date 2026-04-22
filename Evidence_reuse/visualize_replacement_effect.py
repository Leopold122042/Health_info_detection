import argparse
import json
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.plot_style import BAR_WIDTH, PALETTE, figsize, set_academic_style


COLOR_BASELINE = PALETTE["line_blue_high"]
COLOR_AFTER = PALETTE["line_green_high"]
COLOR_DELTA = PALETTE["line_neutral"]
COLOR_POSITIVE = PALETTE["bar_blue_low"]
COLOR_NEGATIVE = PALETTE["bar_green_low"]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_fig(fig, out_path: str):
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def mean_ci95(mean_value, std_value, n):
    if mean_value is None or std_value is None or n is None or n <= 1:
        return None, None
    half = 1.96 * (std_value / math.sqrt(n))
    return mean_value - half, mean_value + half


def wilson_ci95(p_hat, n):
    if p_hat is None or n is None or n <= 0:
        return None, None
    z = 1.96
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def relative_lift(after_value, before_value):
    if after_value is None or before_value is None or before_value <= 0:
        return None
    return (after_value / before_value) - 1.0


def plot_grouped_delta_with_ci(data: dict, out_dir: str):
    grouped = data["similarity_improvement"]["grouped_by_prev_valid_evidence_count"]
    groups = sorted(grouped.keys(), key=lambda x: int(x))

    mean_delta = []
    mean_low = []
    mean_high = []
    max_delta = []
    max_low = []
    max_high = []
    counts = []

    for g in groups:
        block = grouped[g]
        n = block["delta_mean_sim"]["count"]
        counts.append(n)

        pm = block["prev_mean_sim"]["mean"]
        rm = block["retrieved_mean_sim"]["mean"]
        pmax = block["prev_max_sim"]["mean"]
        rmax = block["retrieved_max_sim"]["mean"]

        dm = block["delta_mean_sim"]["mean"]
        ds = block["delta_mean_sim"]["std"]
        mm = block["delta_max_sim"]["mean"]
        ms = block["delta_max_sim"]["std"]

        ci_dm = mean_ci95(dm, ds, n)
        ci_mm = mean_ci95(mm, ms, n)

        dm_rel = relative_lift(rm, pm)
        mm_rel = relative_lift(rmax, pmax)

        if dm_rel is None or ci_dm[0] is None or pm is None or pm <= 0:
            mean_delta.append(np.nan)
            mean_low.append(np.nan)
            mean_high.append(np.nan)
        else:
            ci_dm_low_rel = ci_dm[0] / pm
            ci_dm_high_rel = ci_dm[1] / pm
            mean_delta.append(dm_rel)
            mean_low.append(dm_rel - ci_dm_low_rel)
            mean_high.append(ci_dm_high_rel - dm_rel)

        if mm_rel is None or ci_mm[0] is None or pmax is None or pmax <= 0:
            max_delta.append(np.nan)
            max_low.append(np.nan)
            max_high.append(np.nan)
        else:
            ci_mm_low_rel = ci_mm[0] / pmax
            ci_mm_high_rel = ci_mm[1] / pmax
            max_delta.append(mm_rel)
            max_low.append(mm_rel - ci_mm_low_rel)
            max_high.append(ci_mm_high_rel - mm_rel)

    fig, axes = plt.subplots(1, 2, figsize=figsize(13))
    x = np.arange(len(groups))

    ax = axes[0]
    ax.errorbar(
        x,
        mean_delta,
        yerr=[mean_low, mean_high],
        fmt="o-",
        color=COLOR_DELTA,
        ecolor=COLOR_DELTA,
        elinewidth=1.3,
        capsize=3,
        linewidth=1.8,
    )
    ax.axhline(0, color="#666666", linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_xlabel("置换前有效证据数")
    ax.set_ylabel("mean_sim 提升率（均值 ±95%CI）")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.set_title("分组平均相似度百分比提升")

    for xi, yi, n in zip(x, mean_delta, counts):
        if not np.isnan(yi):
            yoff = 0.0015 if yi >= 0 else -0.0015
            va = "bottom" if yi >= 0 else "top"
            ax.text(xi, yi + yoff, f"n={n}", ha="center", va=va, fontsize=8)

    ax2 = axes[1]
    ax2.errorbar(
        x,
        max_delta,
        yerr=[max_low, max_high],
        fmt="o-",
        color=COLOR_BASELINE,
        ecolor=COLOR_BASELINE,
        elinewidth=1.3,
        capsize=3,
        linewidth=1.8,
    )
    ax2.axhline(0, color="#666666", linewidth=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups)
    ax2.set_xlabel("置换前有效证据数")
    ax2.set_ylabel("max_sim 提升率（均值 ±95%CI）")
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax2.set_title("分组最大相似度百分比提升")

    for xi, yi, n in zip(x, max_delta, counts):
        if not np.isnan(yi):
            yoff = 0.0015 if yi >= 0 else -0.0015
            va = "bottom" if yi >= 0 else "top"
            ax2.text(xi, yi + yoff, f"n={n}", ha="center", va=va, fontsize=8)

    fig.suptitle("按证据数量分组的百分比提升一致性检验", y=1.04)
    save_fig(fig, os.path.join(out_dir, "fig3_grouped_delta_ci.png"))


def plot_grouped_improved_ratio(data: dict, out_dir: str):
    grouped = data["similarity_improvement"]["grouped_by_prev_valid_evidence_count"]
    overall = data["similarity_improvement"]["overall"]

    groups = [g for g in sorted(grouped.keys(), key=lambda x: int(x)) if g != "0"]
    labels = [f"g={g}" for g in groups] + ["overall"]

    mean_imp = []
    mean_imp_err_l = []
    mean_imp_err_u = []
    max_imp = []
    max_imp_err_l = []
    max_imp_err_u = []

    for g in groups:
        block = grouped[g]
        n = block["delta_mean_sim"]["count"]

        p_mean = block["delta_mean_sim"]["improved_ratio"]
        p_max = block["delta_max_sim"]["improved_ratio"]

        ci_mean = wilson_ci95(p_mean, n)
        ci_max = wilson_ci95(p_max, n)

        mean_imp.append(p_mean if p_mean is not None else np.nan)
        mean_imp_err_l.append((p_mean - ci_mean[0]) if ci_mean[0] is not None else np.nan)
        mean_imp_err_u.append((ci_mean[1] - p_mean) if ci_mean[1] is not None else np.nan)

        max_imp.append(p_max if p_max is not None else np.nan)
        max_imp_err_l.append((p_max - ci_max[0]) if ci_max[0] is not None else np.nan)
        max_imp_err_u.append((ci_max[1] - p_max) if ci_max[1] is not None else np.nan)

    n_overall = overall["delta_mean_sim"]["count"]
    p_mean_overall = overall["delta_mean_sim"]["improved_ratio"]
    p_max_overall = overall["delta_max_sim"]["improved_ratio"]
    ci_mean_overall = wilson_ci95(p_mean_overall, n_overall)
    ci_max_overall = wilson_ci95(p_max_overall, n_overall)

    mean_imp.append(p_mean_overall)
    mean_imp_err_l.append(p_mean_overall - ci_mean_overall[0])
    mean_imp_err_u.append(ci_mean_overall[1] - p_mean_overall)
    max_imp.append(p_max_overall)
    max_imp_err_l.append(p_max_overall - ci_max_overall[0])
    max_imp_err_u.append(ci_max_overall[1] - p_max_overall)

    fig, ax = plt.subplots(figsize=figsize(8.6))
    x = np.arange(len(labels)) * 0.86
    w = BAR_WIDTH

    ax.bar(
        x - w / 2,
        mean_imp,
        width=w,
        color=COLOR_POSITIVE,
        alpha=0.72,
        yerr=[mean_imp_err_l, mean_imp_err_u],
        capsize=3,
        label="P(Δmean_sim>0)",
    )
    ax.bar(
        x + w / 2,
        max_imp,
        width=w,
        color=COLOR_NEGATIVE,
        alpha=0.72,
        yerr=[max_imp_err_l, max_imp_err_u],
        capsize=3,
        label="P(Δmax_sim>0)",
    )

    ax.axhline(0.5, color="#666666", linewidth=1.0, linestyle="--")
    ax.text(0.01, 0.51, "0.5 基准线", transform=ax.get_yaxis_transform(), fontsize=9, va="bottom")

    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("分组（按置换前有效证据数）")
    ax.set_ylabel("改进概率（Wilson 95%CI）")
    ax.set_title("置换后“相似度提升概率”分组证据")
    ax.legend(frameon=False, loc="upper right")

    save_fig(fig, os.path.join(out_dir, "fig4_grouped_improvement_probability.png"))


def print_key_conclusions(data: dict):
    overall = data["similarity_improvement"]["overall"]
    d_mean = overall["delta_mean_sim"]["mean"]
    d_max = overall["delta_max_sim"]["mean"]
    p_mean = overall["delta_mean_sim"]["improved_ratio"]
    p_max = overall["delta_max_sim"]["improved_ratio"]

    print("[Summary] Overall similarity change")
    print(f"  Δmean_sim mean = {d_mean:+.6f}, improved_ratio = {p_mean:.2%}")
    print(f"  Δmax_sim  mean = {d_max:+.6f}, improved_ratio = {p_max:.2%}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.abspath(
        os.path.join(current_dir, "..", "outputs", "feasibility", "replacement_effect_summary.json")
    )
    default_output = os.path.abspath(
        os.path.join(current_dir, "..", "outputs", "feasibility", "figures")
    )

    parser = argparse.ArgumentParser(
        description="replacement_effect_summary.json 可视化脚本（中文学术风格）"
    )
    parser.add_argument("--input", default=default_input, help="输入 JSON 路径")
    parser.add_argument("--output-dir", default=default_output, help="输出图片目录")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for stale_name in [
        "fig1_replacement_transition_structure.png",
        "fig2_similarity_overall_improvement.png",
    ]:
        stale_path = os.path.join(args.output_dir, stale_name)
        if os.path.exists(stale_path):
            os.remove(stale_path)

    set_academic_style()
    data = load_json(args.input)

    plot_grouped_delta_with_ci(data, args.output_dir)
    plot_grouped_improved_ratio(data, args.output_dir)
    print_key_conclusions(data)

    print("可视化已完成。")
    print(f"输入 JSON：{os.path.abspath(args.input)}")
    print(f"图片输出目录：{os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
