import matplotlib.pyplot as plt


FIG_HEIGHT = 4.8
BAR_WIDTH = 0.32


PALETTE = {
    "line_blue_high": "#1f6fb2",
    "line_green_high": "#1f8f6a",
    "bar_blue_low": "#9fb9d6",
    "bar_green_low": "#a9c8b7",
    "line_neutral": "#4f5b66",
    "bar_neutral": "#c9d3dd",
    "text_dark": "#1f2933",
    "axis_gray": "#5f6368",
    "compare_blue_high": "#1f77d0",
    "compare_orange_high": "#e67e22",
    "compare_blue_fill": "#66a3d2",
    "compare_orange_fill": "#f2a65a",
}


def figsize(width: float):
    return (width, FIG_HEIGHT)


def set_academic_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "savefig.dpi": 300,
            "figure.dpi": 120,
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "PingFang SC",
                "Arial Unicode MS",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
        }
    )
