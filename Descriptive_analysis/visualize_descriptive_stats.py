import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import PchipInterpolator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.plot_style import BAR_WIDTH, PALETTE, figsize, set_academic_style

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None


COLOR_LABEL0 = PALETTE["line_blue_high"]
COLOR_LABEL1 = PALETTE["line_green_high"]
COLOR_BAR0 = PALETTE["bar_blue_low"]
COLOR_BAR1 = PALETTE["bar_green_low"]
COLOR_GRAY_LIGHT = PALETTE["bar_neutral"]
COLOR_CMP_BLUE = PALETTE["compare_blue_high"]
COLOR_CMP_ORANGE = PALETTE["compare_orange_high"]
COLOR_CMP_BLUE_FILL = PALETTE["compare_blue_fill"]
COLOR_CMP_ORANGE_FILL = PALETTE["compare_orange_fill"]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def p_to_stars(p_value):
    if p_value is None:
        return "n.s."
    try:
        p = float(p_value)
    except Exception:
        return "n.s."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def sci_p(p_value):
    if p_value is None:
        return "NA"
    try:
        p = float(p_value)
    except Exception:
        return "NA"
    if p < 1e-3:
        return f"{p:.2e}"
    return f"{p:.4f}"


def get_chinese_font_path() -> str:
    candidates = [
        r"C:\\Windows\\Fonts\\msyh.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf",
        r"C:\\Windows\\Fonts\\msyhbd.ttc",
        r"C:\\Windows\\Fonts\\simsun.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def save_fig(fig, out_path: str):
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_time_distribution(data: dict, out_dir: str):
    monthly = data["time_and_length"]["claim_time_distribution_monthly"]
    months = list(monthly.keys())
    counts = [monthly[m] for m in months]

    fig, ax = plt.subplots(figsize=figsize(11))
    x = np.arange(len(months))
    ax.plot(x, counts, color=COLOR_LABEL0, linewidth=1.6)
    ax.set_xlabel("月份（YYYY-MM）")
    ax.set_ylabel("声明数量（条）")
    ax.set_title("声明时间分布")

    if len(months) > 24:
        step = max(1, len(months) // 12)
        tick_idx = np.arange(0, len(months), step)
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([months[i] for i in tick_idx], rotation=30, ha="right")
    else:
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=30, ha="right")

    ax.margins(x=0.01)
    save_fig(fig, os.path.join(out_dir, "fig1_time_distribution.png"))


def plot_source_distribution(data: dict, out_dir: str, top_n: int = 15):
    source_data = data.get("source_distribution", {})
    rows = source_data.get("distribution", []) if isinstance(source_data, dict) else []
    if not rows:
        return

    top_rows = rows[:top_n]
    labels = [r.get("source", "UNKNOWN") for r in top_rows]
    counts = [r.get("count", 0) for r in top_rows]
    ratios = [r.get("ratio") for r in top_rows]

    labels = labels[::-1]
    counts = counts[::-1]
    ratios = ratios[::-1]

    fig, ax = plt.subplots(figsize=figsize(10))
    y = np.arange(len(labels))
    bars = ax.barh(y, counts, color=COLOR_CMP_BLUE_FILL, edgecolor=COLOR_CMP_BLUE, alpha=0.75)

    for bar, ratio in zip(bars, ratios):
        width = bar.get_width()
        ratio_text = f" ({ratio:.1%})" if isinstance(ratio, (int, float)) else ""
        ax.text(width, bar.get_y() + bar.get_height() / 2, f"{int(width)}{ratio_text}", va="center", ha="left", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("样本数")
    ax.set_ylabel("Source")
    ax.set_title(f"Source 信息分布（Top {min(top_n, len(rows))}）")
    ax.grid(axis="x", linestyle="--", linewidth=0.6, alpha=0.45)

    save_fig(fig, os.path.join(out_dir, "fig4_source_distribution.png"))


def _kde_curve(values: List[float], points=200):
    if not values:
        return np.array([]), np.array([])
    arr = np.array(values, dtype=float)
    if len(arr) < 2 or np.std(arr) == 0:
        xs = np.linspace(arr.min() - 1, arr.max() + 1, points)
        ys = np.zeros_like(xs)
        ys[np.argmin(np.abs(xs - arr.mean()))] = 1.0
        return xs, ys

    xs = np.linspace(arr.min(), arr.max(), points)
    std = np.std(arr, ddof=1)
    bw = 1.06 * std * (len(arr) ** (-1 / 5))
    bw = max(bw, 1e-6)
    diff = (xs[:, None] - arr[None, :]) / bw
    ys = np.exp(-0.5 * diff**2).sum(axis=1) / (len(arr) * bw * np.sqrt(2 * np.pi))
    return xs, ys


def plot_length_distributions(data: dict, out_dir: str):
    dist = data["time_and_length"]["length_distribution"]
    claim_len = dist["claim_len"]["values"]
    ev_single_len = dist["evidence_len_single"]["values"]
    ev_claim_total = dist["evidence_len_per_claim_total"]["values"]

    fig, axes = plt.subplots(1, 3, figsize=figsize(14))
    payload = [
        (claim_len, "声明文本长度", "字符数"),
        (ev_single_len, "证据长度（单槽位）", "字符数"),
        (ev_claim_total, "证据长度（单条声明总计）", "字符数"),
    ]

    fill_colors = [COLOR_BAR0, COLOR_BAR1, COLOR_GRAY_LIGHT]
    line_colors = [COLOR_LABEL0, COLOR_LABEL1, PALETTE["line_neutral"]]
    for ax, (vals, title, xlabel), f_color, l_color in zip(axes, payload, fill_colors, line_colors):
        ax.hist(vals, bins=30, density=True, color=f_color, alpha=0.35, edgecolor="none")
        xs, ys = _kde_curve(vals)
        if len(xs) > 0:
            ax.plot(xs, ys, color=l_color, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("密度")

    fig.suptitle("文本长度分布", y=1.05)
    save_fig(fig, os.path.join(out_dir, "fig2_length_distributions.png"))


def plot_keywords_wordcloud(data: dict, out_dir: str, top_k=120):
    kv = data["keywords_by_time_period"]
    periods = sorted(kv.keys())

    token_global = {}
    for period in periods:
        for row in kv[period].get("top_keywords", []):
            token = row["token"]
            token_global[token] = token_global.get(token, 0) + row["freq"]

    top_freq = dict(sorted(token_global.items(), key=lambda x: x[1], reverse=True)[:top_k])
    fig, ax = plt.subplots(figsize=figsize(11))

    if WordCloud is None:
        tokens = list(top_freq.keys())[:30][::-1]
        values = [top_freq[t] for t in tokens]
        ax.barh(tokens, values, color=COLOR_LABEL0, alpha=0.85)
        ax.set_xlabel("词频（次）")
        ax.set_ylabel("关键词")
    else:
        wc = WordCloud(
            width=1600,
            height=900,
            background_color="white",
            colormap="viridis",
            max_words=top_k,
            relative_scaling=0.5,
            prefer_horizontal=0.9,
            font_path=get_chinese_font_path() or None,
            contour_width=1,
            contour_color="#bdbdbd",
        ).generate_from_frequencies(top_freq)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")

    ax.set_title("关键词词云（全时段汇总）")
    save_fig(fig, os.path.join(out_dir, "fig3_keywords_wordcloud.png"))


def plot_evidence_slots(data: dict, out_dir: str):
    slot_data = data["evidence_slot_distribution"]
    by_label = slot_data["by_label_distribution"]
    levels = [0, 1, 2, 3, 4, 5]

    counts0 = np.array([by_label.get("0", {}).get(str(s), 0) for s in levels], dtype=float)
    counts1 = np.array([by_label.get("1", {}).get(str(s), 0) for s in levels], dtype=float)

    prop0 = counts0 / counts0.sum() if counts0.sum() > 0 else counts0
    prop1 = counts1 / counts1.sum() if counts1.sum() > 0 else counts1

    x = np.arange(len(levels))
    w = BAR_WIDTH

    fig, ax = plt.subplots(figsize=figsize(8))
    ax.bar(x - w / 2, prop0, width=w, color=COLOR_BAR0, alpha=0.55, label="label=0")
    ax.bar(x + w / 2, prop1, width=w, color=COLOR_BAR1, alpha=0.55, label="label=1")

    levels_arr = np.array(levels, dtype=float)
    xfit = np.linspace(levels_arr.min(), levels_arr.max(), 300)
    if np.count_nonzero(prop0) >= 2:
        yfit0 = PchipInterpolator(levels_arr, prop0)(xfit)
        ax.plot(xfit, yfit0, color=COLOR_LABEL0, linewidth=1.2)
    if np.count_nonzero(prop1) >= 2:
        yfit1 = PchipInterpolator(levels_arr, prop1)(xfit)
        ax.plot(xfit, yfit1, color=COLOR_LABEL1, linewidth=1.2)

    ax.scatter(levels_arr, prop0, color=COLOR_LABEL0, s=18, zorder=3)
    ax.scatter(levels_arr, prop1, color=COLOR_LABEL1, s=18, zorder=3)

    ax.set_xlabel("每条声明的证据槽位数")
    ax.set_ylabel("组内占比")
    ax.set_title("不同label的证据槽位分布")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in levels])
    legend_items = [
        Line2D([0], [0], color=COLOR_LABEL0, lw=1.2, marker="o", label="label=0（平滑趋势线）"),
        Line2D([0], [0], color=COLOR_LABEL1, lw=1.2, marker="o", label="label=1（平滑趋势线）"),
    ]
    ax.legend(handles=legend_items, loc="upper right", frameon=False)

    sig = slot_data.get("significance", {})
    chi = sig.get("chi_square") if isinstance(sig.get("chi_square"), dict) else None
    mw = sig.get("mann_whitney_u") if isinstance(sig.get("mann_whitney_u"), dict) else None
    one_sided = data.get("label_group_differences", {}).get("evidence_slots_label0_less_label1", {})

    chi_p = chi.get("p_value") if chi and "p_value" in chi else None
    mw_p = mw.get("p_value") if mw and "p_value" in mw else None
    one_sided_p = one_sided.get("p_value") if isinstance(one_sided, dict) else None

    y_max = max(np.max(prop0), np.max(prop1)) if len(prop0) else 0.0
    ax.text(
        0.02,
        min(0.98, y_max + 0.08),
        "",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
    )

    save_fig(fig, os.path.join(out_dir, "fig5_evidence_slots.png"))


def _ecdf(values: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.array(values, dtype=float))
    if arr.size == 0:
        return np.array([]), np.array([])
    y = np.arange(1, arr.size + 1) / arr.size
    return arr, y


def plot_tfidf_similarity(data: dict, out_dir: str):
    sim_data = data["claim_evidence_tfidf_similarity"]
    s0 = sim_data["by_label"].get("0", {}).get("values", [])
    s1 = sim_data["by_label"].get("1", {}).get("values", [])
    ks = sim_data.get("ks_test_label0_vs_label1", {})
    one_sided = data.get("label_group_differences", {}).get("tfidf_similarity_label0_greater_label1", {})

    _ = ks.get("p_value") if isinstance(ks, dict) else None
    _ = ks.get("statistic") if isinstance(ks, dict) else None
    _ = one_sided.get("p_value") if isinstance(one_sided, dict) else None

    fig, axes = plt.subplots(1, 2, figsize=figsize(12))

    ax = axes[0]
    ax.hist(s0, bins=30, density=True, color=COLOR_CMP_BLUE_FILL, alpha=0.48, label="label=0")
    ax.hist(s1, bins=30, density=True, color=COLOR_CMP_ORANGE_FILL, alpha=0.48, label="label=1")
    x0, y0 = _kde_curve(s0)
    x1, y1 = _kde_curve(s1)
    if len(x0) > 0:
        ax.plot(x0, y0, color=COLOR_CMP_BLUE, linewidth=1.5)
    if len(x1) > 0:
        ax.plot(x1, y1, color=COLOR_CMP_ORANGE, linewidth=1.5)
    ax.set_xlabel("TF-IDF 余弦相似度")
    ax.set_ylabel("密度")
    ax.set_title("相似度分布")
    ax.legend(loc="upper right", frameon=False)

    ax2 = axes[1]
    vio = ax2.violinplot(
        [s0, s1],
        positions=[1, 2],
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, color in zip(vio["bodies"], [COLOR_CMP_BLUE, COLOR_CMP_ORANGE]):
        body.set_facecolor(COLOR_CMP_BLUE_FILL if color == COLOR_CMP_BLUE else COLOR_CMP_ORANGE_FILL)
        body.set_edgecolor(color)
        body.set_alpha(0.35)
    if "cmedians" in vio:
        vio["cmedians"].set_color("#222222")
        vio["cmedians"].set_linewidth(1.6)

    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["label=0", "label=1"])
    ax2.set_ylabel("TF-IDF 余弦相似度")
    ax2.set_title("分组小提琴图对比")

    fig.suptitle("不同label的声明-证据 TF-IDF 相似度", y=1.05)

    save_fig(fig, os.path.join(out_dir, "fig6_tfidf_similarity.png"))


def plot_evidence_sentiment(data: dict, out_dir: str):
    sent_data = data.get("evidence_sentiment", {})
    if not sent_data.get("available"):
        return

    s0 = sent_data.get("by_label", {}).get("0", {}).get("values", [])
    s1 = sent_data.get("by_label", {}).get("1", {}).get("values", [])
    sent_diff = data.get("label_group_differences", {}).get("evidence_sentiment_label0_vs_label1", {})
    _ = sent_diff.get("p_value") if isinstance(sent_diff, dict) else None

    fig, axes = plt.subplots(1, 2, figsize=figsize(12))

    ax = axes[0]
    ax.hist(s0, bins=30, density=True, color=COLOR_CMP_BLUE_FILL, alpha=0.48, label="label=0")
    ax.hist(s1, bins=30, density=True, color=COLOR_CMP_ORANGE_FILL, alpha=0.48, label="label=1")
    x0, y0 = _kde_curve(s0)
    x1, y1 = _kde_curve(s1)
    if len(x0) > 0:
        ax.plot(x0, y0, color=COLOR_CMP_BLUE, linewidth=1.6)
    if len(x1) > 0:
        ax.plot(x1, y1, color=COLOR_CMP_ORANGE, linewidth=1.6)
    ax.set_xlabel("证据情感得分（0-1）")
    ax.set_ylabel("密度")
    ax.set_title("证据情感分布")
    ax.legend(loc="upper right", frameon=False)

    ax2 = axes[1]
    ax2.boxplot(
        [s0, s1],
        tick_labels=["label=0", "label=1"],
        patch_artist=True,
        boxprops={"facecolor": COLOR_CMP_BLUE_FILL, "alpha": 0.55},
        medianprops={"color": "#222222", "linewidth": 1.4},
        whiskerprops={"color": "#444444"},
        capprops={"color": "#444444"},
    )
    ax2.set_ylabel("证据情感得分（0-1）")
    ax2.set_title("分label情感汇总")

    fig.suptitle("不同label的证据情感差异", y=1.05)

    save_fig(fig, os.path.join(out_dir, "fig7_evidence_sentiment.png"))


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_input = os.path.join(current_dir, "descriptive_stats.json")
    default_output_dir = os.path.join(current_dir, "figures")

    parser = argparse.ArgumentParser(
        description="descriptive_stats.json 可视化脚本（中文学术风格）"
    )
    parser.add_argument("--input", default=default_input, help="输入 JSON 路径")
    parser.add_argument(
        "--output-dir",
        default=default_output_dir,
        help="输出图片目录",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_academic_style()
    data = load_json(args.input)

    plot_time_distribution(data, args.output_dir)
    plot_source_distribution(data, args.output_dir, top_n=15)
    plot_length_distributions(data, args.output_dir)
    plot_keywords_wordcloud(data, args.output_dir, top_k=120)
    plot_evidence_slots(data, args.output_dir)
    plot_tfidf_similarity(data, args.output_dir)
    plot_evidence_sentiment(data, args.output_dir)

    print("可视化已完成。")
    print(f"输入 JSON：{os.path.abspath(args.input)}")
    print(f"图片输出目录：{os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
