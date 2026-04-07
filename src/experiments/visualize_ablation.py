import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.utils.plot_style import BAR_WIDTH, PALETTE, figsize, set_academic_style


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_fig(fig, out_path: Path):
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_full_model_training_curve(ablation_dir: Path, out_dir: Path, seed: int):
    log_path = ablation_dir / "full_model" / f"seed_{seed}" / "epoch_log.jsonl"
    rows = load_jsonl(log_path)
    if not rows:
        raise ValueError(f"No epoch log found in {log_path}")

    epochs = np.array([r["epoch"] for r in rows], dtype=int)
    train_loss = np.array([r["train_loss"] for r in rows], dtype=float)
    val_macro_f1 = np.array([r["val_macro_f1"] for r in rows], dtype=float)
    val_mcc = np.array([r["val_mcc"] for r in rows], dtype=float)
    val_acc = np.array([r["val_acc"] for r in rows], dtype=float)

    best_idx = int(np.argmax(val_mcc))
    best_epoch = int(epochs[best_idx])
    best_mcc = float(val_mcc[best_idx])

    fig, axes = plt.subplots(1, 2, figsize=figsize(13))

    ax = axes[0]
    ax.plot(epochs, train_loss, color=PALETTE["line_blue_high"], linewidth=1.8, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Train Loss")
    ax.set_title("full_model 训练损失曲线")

    ax2 = axes[1]
    ax2.plot(epochs, val_macro_f1, color=PALETTE["line_green_high"], linewidth=1.8, marker="o", markersize=3, label="Val Macro-F1")
    ax2.plot(epochs, val_mcc, color=PALETTE["line_blue_high"], linewidth=1.8, marker="o", markersize=3, label="Val MCC")
    ax2.plot(epochs, val_acc, color=PALETTE["line_neutral"], linewidth=1.6, linestyle="--", label="Val ACC")
    ax2.axvline(best_epoch, color=PALETTE["axis_gray"], linestyle=":", linewidth=1.2)
    ax2.axhline(best_mcc, color=PALETTE["axis_gray"], linestyle="--", linewidth=1.0)
    ax2.scatter([best_epoch], [best_mcc], color=PALETTE["line_blue_high"], s=34, zorder=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Validation Metrics")
    ax2.set_title("full_model 验证集指标演化")
    ax2.legend(frameon=False, loc="lower right")

    fig.suptitle("Ablation: full_model 训练过程", y=1.03)
    save_fig(fig, out_dir / "fig_ablation_full_model_training_curve.png")


def plot_ablation_contribution_ranking(ablation_dir: Path, out_dir: Path, seed: int):
    experiments = ["full_model", "no_tfidf", "no_ce", "no_ee", "no_feats"]
    metrics = {}
    for exp in experiments:
        path = ablation_dir / exp / f"seed_{seed}" / "metrics.json"
        metrics[exp] = load_json(path)["best"]

    base = metrics["full_model"]
    ablations = ["no_tfidf", "no_ee", "no_ce", "no_feats"]

    delta_mcc = np.array([metrics[e]["mcc"] - base["mcc"] for e in ablations], dtype=float)
    delta_macro = np.array([metrics[e]["macro_f1"] - base["macro_f1"] for e in ablations], dtype=float)
    delta_real_f1 = np.array([metrics[e]["real_f1"] - base["real_f1"] for e in ablations], dtype=float)
    delta_fake_f1 = np.array([metrics[e]["fake_f1"] - base["fake_f1"] for e in ablations], dtype=float)

    contribution_strength = -delta_mcc
    order = np.argsort(-contribution_strength)
    ablations = [ablations[i] for i in order]
    delta_mcc = delta_mcc[order]
    delta_macro = delta_macro[order]
    delta_real_f1 = delta_real_f1[order]
    delta_fake_f1 = delta_fake_f1[order]

    x = np.arange(len(ablations))
    fig, axes = plt.subplots(1, 2, figsize=figsize(13))

    ax = axes[0]
    bars = ax.bar(x, -delta_mcc, width=BAR_WIDTH, color=PALETTE["bar_blue_low"], edgecolor="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ablations)
    ax.set_ylabel("Contribution Strength (−ΔMCC)")
    ax.set_title("消融贡献排序（按 MCC 跌幅）")
    for b, d in zip(bars, delta_mcc):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"ΔMCC={d:+.3f}", ha="center", va="bottom", fontsize=9)

    ax2 = axes[1]
    w = BAR_WIDTH * 0.78
    ax2.bar(x - w, delta_macro, width=w, color=PALETTE["bar_neutral"], edgecolor="black", linewidth=0.7, label="ΔMacro-F1")
    ax2.bar(x, delta_real_f1, width=w, color=PALETTE["bar_green_low"], edgecolor="black", linewidth=0.7, label="ΔReal-F1")
    ax2.bar(x + w, delta_fake_f1, width=w, color=PALETTE["bar_blue_low"], edgecolor="black", linewidth=0.7, label="ΔFake-F1")
    ax2.axhline(0, color=PALETTE["axis_gray"], linewidth=1.0)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ablations)
    ax2.set_ylabel("Metric Delta vs full_model")
    ax2.set_title("不同指标对消融的敏感性")
    ax2.legend(frameon=False, loc="lower right")

    fig.suptitle("Ablation 特征贡献与指标退化", y=1.03)
    save_fig(fig, out_dir / "fig_ablation_contribution_ranking.png")


def plot_fake_risk_oriented_map(ablation_dir: Path, out_dir: Path, seed: int):
    experiments = ["full_model", "no_tfidf", "no_ce", "no_ee", "no_feats"]
    fake_support = None
    rows = []
    for exp in experiments:
        path = ablation_dir / exp / f"seed_{seed}" / "metrics.json"
        m = load_json(path)["best"]
        report = m.get("classification_report", {})
        support_fake = float(report.get("Fake", {}).get("support", 0.0))
        fake_support = support_fake if fake_support is None else fake_support

        fake_p = float(m["fake_precision"])
        fake_r = float(m["fake_recall"])
        tp = fake_r * support_fake
        fn = support_fake - tp
        fp = (tp / fake_p - tp) if fake_p > 0 else 0.0
        rows.append((exp, fake_p, fake_r, fn, fp))

    fig, ax = plt.subplots(figsize=figsize(8.6))

    for exp, fake_p, fake_r, fn, fp in rows:
        marker_size = 26 + fn * 0.55
        color = PALETTE["line_blue_high"] if exp == "full_model" else PALETTE["bar_blue_low"]
        edge = "black" if exp == "full_model" else PALETTE["axis_gray"]
        ax.scatter(fake_p, fake_r, s=marker_size, color=color, edgecolors=edge, linewidths=0.9, alpha=0.9, zorder=3)
        ax.annotate(
            f"{exp}\nFN≈{fn:.0f}, FP≈{fp:.0f}",
            (fake_p, fake_r),
            xytext=(-8, 8),
            textcoords="offset points",
            fontsize=8,
            ha="right",
            va="bottom",
        )

    ax.set_xlabel("Fake Precision（越高越少误报）")
    ax.set_ylabel("Fake Recall（越高越少漏检）")
    ax.set_title("风险导向：假信息识别的 Precision-Recall 分布")
    x_vals = np.linspace(0.88, 0.95, 200)
    for fn_target in [50, 80, 110]:
        if fake_support and fake_support > 0:
            recall_line = 1.0 - fn_target / fake_support
            if 0 < recall_line < 1:
                y_vals = np.full_like(x_vals, recall_line)
                ax.plot(x_vals, y_vals, linestyle="--", linewidth=0.9, color=PALETTE["axis_gray"], alpha=0.6)
                ax.text(x_vals[-1], recall_line + 0.001, f"FN≈{fn_target}", fontsize=8, ha="right", va="bottom")

    ax.set_xlim(0.90, 0.95)
    ax.set_ylim(0.84, 0.95)
    save_fig(fig, out_dir / "fig_ablation_fake_risk_map.png")


def main():
    parser = argparse.ArgumentParser(description="Ablation results visualization")
    parser.add_argument(
        "--ablation-dir",
        type=Path,
        default=Path("outputs/graph/ablation"),
        help="ablation目录路径",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/graph/ablation/figures"),
        help="图片输出目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="读取的seed目录",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_academic_style()

    plot_full_model_training_curve(args.ablation_dir, args.out_dir, args.seed)
    plot_ablation_contribution_ranking(args.ablation_dir, args.out_dir, args.seed)
    plot_fake_risk_oriented_map(args.ablation_dir, args.out_dir, args.seed)

    print(f"[Export] Ablation figures saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
