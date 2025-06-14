import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import numpy as np
from scipy.interpolate import make_interp_spline

MAIN_COLORS = sns.color_palette("husl", n_colors=10)
PAIRED_COLORS = sns.color_palette("Paired", n_colors=12)
CATEGORICAL_COLORS = sns.color_palette("Set3", n_colors=10)
DIVERGING_COLORS = sns.color_palette("RdYlBu", n_colors=10)
PIE_COLORS = sns.color_palette("husl", n_colors=12)
GREY_COLORS_DARK = sns.color_palette("Greys_r", n_colors=10)

FIG_SIZE_LARGE = (15, 8)
FIG_SIZE_MEDIUM = (12, 6)
FIG_SIZE_SMALL = (8, 6)

FONT_SIZES = {
    "title": 20,
    "axis_label": 18,
    "tick": 16,
    "legend": 18,
    "annotation": 16,
}

FONT_FAMILY = "sans-serif"
FONT_WEIGHT_NORMAL = "normal"
FONT_WEIGHT_BOLD = "bold"


def format_time_label(date, granularity):
    if granularity == "week":
        return date.strftime("%Y-W%W")
    elif granularity == "month":
        return date.strftime("%Y-%m")
    elif granularity == "quarter":
        quarter = (date.month - 1) // 3 + 1
        return f"{date.year} Q{quarter}"
    else:
        return str(date.year)


def setup_plotting_style():
    plt.style.use("seaborn-v0_8-white")

    plt.rcParams.update(
        {
            "figure.autolayout": True,
            "figure.titlesize": FONT_SIZES["title"],
            "figure.facecolor": "white",
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZES["axis_label"],
            "axes.titlesize": FONT_SIZES["title"],
            "axes.titleweight": FONT_WEIGHT_BOLD,
            "axes.labelsize": FONT_SIZES["axis_label"],
            "axes.labelweight": FONT_WEIGHT_NORMAL,
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "xtick.labelsize": FONT_SIZES["tick"],
            "ytick.labelsize": FONT_SIZES["tick"],
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "legend.fontsize": FONT_SIZES["legend"],
            "legend.title_fontsize": FONT_SIZES["legend"],
            "legend.facecolor": "white",
            "legend.edgecolor": "#CCCCCC",
            "legend.framealpha": 1.0,
            "legend.borderpad": 0.6,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#CCCCCC",
            "grid.linestyle": "--",
            "grid.linewidth": 0.8,
            "axes.titlepad": 20,
        }
    )


def setup_axis_ticks(ax, dates, granularity, n_ticks=12, rotation=45):
    n_ticks = min(n_ticks if n_ticks else 12, len(dates))
    tick_indices = np.linspace(0, len(dates) - 1, n_ticks, dtype=int)
    tick_dates = [dates[i] for i in tick_indices]

    ax.set_xticks(tick_dates)
    formatted_labels = [format_time_label(date, granularity) for date in tick_dates]
    ax.set_xticklabels(
        formatted_labels,
        rotation=rotation,
        ha="right",
        fontsize=FONT_SIZES["tick"],
        color="#333333",
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")
        spine.set_linewidth(1.0)


def setup_legend(ax, title=None, loc="upper left", ncol=1):
    legend = ax.legend(
        title=title,
        loc=loc,
        ncol=ncol,
        frameon=True,
        facecolor="white",
        edgecolor="#CCCCCC",
        framealpha=1.0,
        borderpad=0.6,
    )

    if title:
        legend.get_title().set_fontsize(FONT_SIZES["legend"])
        legend.get_title().set_color("#333333")
    plt.setp(legend.get_texts(), fontsize=FONT_SIZES["legend"], color="#333333")


def apply_grid_style(ax, major_alpha=0.3, minor_alpha=0.15):
    ax.grid(
        True,
        which="major",
        linestyle="--",
        alpha=major_alpha,
        color="#CCCCCC",
        linewidth=0.8,
    )
    ax.grid(
        True,
        which="minor",
        linestyle=":",
        alpha=minor_alpha,
        color="#CCCCCC",
        linewidth=0.6,
    )

    ax.set_axisbelow(True)


def create_pie_chart(data, labels, title, figsize=FIG_SIZE_MEDIUM, explode=None):
    fig, ax = plt.subplots(figsize=figsize)

    if explode is None:
        explode = [0.02] * len(data)

    wedges, texts, autotexts = ax.pie(
        data,
        explode=explode,
        labels=labels,
        colors=PIE_COLORS,
        autopct="%1.1f%%",
        pctdistance=0.85,
        wedgeprops=dict(width=0.7),
    )

    plt.setp(
        autotexts,
        color="black",
        weight=FONT_WEIGHT_BOLD,
        fontsize=FONT_SIZES["annotation"],
    )
    plt.setp(texts, fontsize=FONT_SIZES["annotation"], weight=FONT_WEIGHT_NORMAL)

    ax.set_title(title, pad=20, fontsize=FONT_SIZES["title"], weight=FONT_WEIGHT_BOLD)

    return fig, ax


def save_plot(fig, output_path, base_filename, dpi=300):
    fig.savefig(output_path / f"{base_filename}.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(output_path / f"{base_filename}.pdf", bbox_inches="tight", format="pdf")
    plt.close(fig)


def smooth_with_bspline(dates, values, n_points=300):
    x = np.array([(d - dates[0]).total_seconds() for d in dates])
    y = values.copy()

    x_smooth = np.linspace(x.min(), x.max(), n_points)

    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)

    dates_smooth = [dates[0] + timedelta(seconds=int(xs)) for xs in x_smooth]

    return dates_smooth, y_smooth
