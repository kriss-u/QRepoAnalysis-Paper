import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import json
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

from ..config.constants import PROCESSED_DATA_DIR
from ..utils.logger import setup_logger
from ..utils.plot_utils_ieee import (
    FONT_SIZES,
    FONT_WEIGHT_BOLD,
    setup_plotting_style,
    MAIN_COLORS,
    FIG_SIZE_SINGLE_COL,
    PLOT_LINE_WIDTH,
    setup_axis_ticks,
    setup_legend,
    save_plot,
    apply_grid_style,
)

logger = setup_logger(__name__, "rq2", "commit_sizes")


def get_pandas_frequency(granularity):
    freq_map = {"day": "D", "week": "W", "month": "ME", "quarter": "QE", "year": "YE"}
    if granularity not in freq_map:
        raise ValueError(
            f"Invalid granularity: {granularity}. Must be one of {list(freq_map.keys())}"
        )
    return freq_map[granularity]


def load_commit_data(data_dir=PROCESSED_DATA_DIR / "rq2"):
    commit_patterns_dir = data_dir / "commit_patterns"
    if not commit_patterns_dir.exists():
        raise FileNotFoundError(
            f"Commit patterns directory not found: {commit_patterns_dir}"
        )

    dfs = []
    required_columns = [
        "author_date",
        "files_changed",
        "insertions",
        "deletions",
        "commit_hash",
        "subject",
        "repo_name",
    ]

    for year_dir in sorted(commit_patterns_dir.iterdir()):
        if year_dir.is_dir():
            try:
                df = pq.read_table(year_dir, columns=required_columns).to_pandas()
                dfs.append(df)
                logger.info(f"Loaded data for year {year_dir.name}")
            except Exception as e:
                logger.error(f"Error loading data for year {year_dir.name}: {str(e)}")

    if not dfs:
        raise ValueError("No data loaded")

    df = pd.concat(dfs, ignore_index=True)
    df["author_date"] = pd.to_datetime(df["author_date"], utc=True)
    df = df.sort_values("author_date")

    mask = df["author_date"] < "2024-10-01"
    df = df[mask]

    logger.info(
        f"Loaded {len(df):,} commits spanning {df['author_date'].min():%Y-%m-%d} to {df['author_date'].max():%Y-%m-%d}"
    )
    return df


def analyze_high_impact_commits(df, threshold_percentile=95):
    df = df.copy()
    df["total_changes"] = df["insertions"] + df["deletions"]

    changes_threshold = df["total_changes"].quantile(threshold_percentile / 100)
    high_impact_df = df[df["total_changes"] >= changes_threshold]

    high_impact_commits = []
    for _, row in high_impact_df.iterrows():
        commit_info = {
            "date": row["author_date"].isoformat(),
            "repo_name": row["repo_name"],
            "additions": int(row["insertions"]),
            "deletions": int(row["deletions"]),
            "files_changed": int(row["files_changed"]),
            "commit_hash": row["commit_hash"],
            "subject": row["subject"],
            "total_changes": int(row["total_changes"]),
        }
        high_impact_commits.append(commit_info)

    stats = {
        "total_commits": len(df),
        "high_impact_count": len(high_impact_df),
        "high_impact_percentage": (len(high_impact_df) / len(df)) * 100,
        "percentiles": {
            "additions": {
                str(p): int(df["insertions"].quantile(p / 100))
                for p in [75, 90, 95, 99]
            },
            "deletions": {
                str(p): int(df["deletions"].quantile(p / 100)) for p in [75, 90, 95, 99]
            },
            "total_changes": {
                str(p): int(df["total_changes"].quantile(p / 100))
                for p in [75, 90, 95, 99]
            },
        },
        "high_impact_commits": sorted(
            high_impact_commits, key=lambda x: x["total_changes"], reverse=True
        )[
            :100
        ],  # Top 100 high-impact commits
    }

    return stats


def generate_quarterly_summary(df):
    quarterly = (
        df.resample("QE", on="author_date")
        .agg(
            {
                "insertions": "sum",
                "deletions": "sum",
                "files_changed": "sum",
                "commit_hash": "count",
            }
        )
        .round(0)
    )

    summary = {}
    for idx, row in quarterly.iterrows():
        quarter = f"{idx.year}Q{idx.quarter}"
        summary[quarter] = {
            "additions": int(row["insertions"]),
            "deletions": int(row["deletions"]),
            "files_changed": int(row["files_changed"]),
            "commit_count": int(row["commit_hash"]),
        }

    return summary


def export_analysis_results(stats, quarterly_summary, output_dir):
    results = {"quarterly_summary": quarterly_summary, "high_impact_analysis": stats}

    with open(output_dir / "commit_analysis.json", "w") as f:
        json.dump(results, f, indent=2)


def analyze_commit_sizes(args):
    logger.info(f"Analyzing commit sizes using {args.granularity} granularity")

    try:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        df = load_commit_data()
        stats = analyze_high_impact_commits(df)
        quarterly_summary = generate_quarterly_summary(df)

        freq = get_pandas_frequency(args.granularity)

        resampled_df = df.resample(freq, on="author_date").agg(
            {"insertions": "sum", "deletions": "sum", "files_changed": "sum"}
        )

        # create_volume_plot(resampled_df, args.granularity, plots_dir)
        # create_dual_axis_plot(resampled_df, args.granularity, plots_dir)

        export_analysis_results(stats, quarterly_summary, output_dir)

        logger.info("Commit size analysis completed successfully")

    except Exception as e:
        logger.error(f"Error analyzing commit sizes: {str(e)}")
        raise


"""
def create_volume_plot(df, granularity, plots_dir):
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    dates = df.index.to_pydatetime()
    volume = df["insertions"] + df["deletions"]

    dates_smooth, values_smooth = smooth_with_boundary_conditions(
        dates, volume.values, num_points=max(300, len(dates))
    )

    ax.plot(
        dates_smooth,
        values_smooth,
        label="Total Changes",
        color=MAIN_COLORS[0],
        linewidth=2.5,
        alpha=0.8,
    )

    dates_smooth, insertions_smooth = smooth_with_boundary_conditions(
        dates, df["insertions"].values
    )

    dates_smooth, deletions_smooth = smooth_with_boundary_conditions(
        dates, df["deletions"].values
    )

    ax.fill_between(
        dates_smooth,
        0,
        insertions_smooth,
        alpha=0.3,
        color=MAIN_COLORS[1],
        label="Additions",
    )

    ax.fill_between(
        dates_smooth,
        insertions_smooth,
        insertions_smooth + deletions_smooth,
        alpha=0.3,
        color=MAIN_COLORS[2],
        label="Deletions",
    )

    ax.set_title(
        "Volume of Code Changes Over Time",
        fontsize=FONT_SIZES["title"],
        fontweight=FONT_WEIGHT_BOLD,
        pad=20,
    )
    ax.set_xlabel("")
    ax.set_ylabel(
        "Number of Lines Changed",
        fontsize=FONT_SIZES["axis_label"],
    )

    ax.grid(True, which="major", linestyle="--", alpha=0.7)
    ax.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)

    # Ensure plot starts from 0
    # ax.set_ylim(bottom=0)
    # ax.set_yscale("log", base=2)
    setup_axis_ticks(ax, dates, granularity)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    setup_legend(ax, title="Changes", loc="upper left")

    save_plot(fig, plots_dir, f"commit_volume_{granularity}")
    plt.close(fig)
"""

"""
def create_dual_axis_plot(df, granularity, plots_dir):
    setup_plotting_style()
    fig, ax1 = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)
    ax2 = ax1.twinx()

    dates = df.index.to_pydatetime()

    lines1 = []
    for idx, (column, label) in enumerate(
        [("insertions", "Lines Added"), ("deletions", "Lines Deleted")]
    ):
        dates_smooth, values_smooth = smooth_with_boundary_conditions(
            dates, df[column].values
        )
        ln = ax1.plot(
            dates_smooth,
            values_smooth,
            label=label,
            color=MAIN_COLORS[idx],
            linewidth=2.5,
            alpha=0.8,
        )
        lines1.extend(ln)

    dates_smooth, values_smooth = smooth_with_boundary_conditions(
        dates, df["files_changed"].values
    )
    line2 = ax2.plot(
        dates_smooth,
        values_smooth,
        label="Files Changed",
        color=MAIN_COLORS[2],
        linewidth=2.5,
        alpha=0.8,
    )

    ax1.set_title(
        "Commit Changes Over Time",
        fontsize=FONT_SIZES["title"],
        fontweight=FONT_WEIGHT_BOLD,
        pad=20,
    )
    ax1.set_xlabel("")
    ax1.set_ylabel(
        "Number of Lines Added/Deleted",
        fontsize=FONT_SIZES["axis_label"],
    )
    ax2.set_ylabel(
        "Number of Files Changed",
        fontsize=FONT_SIZES["axis_label"],
    )

    ax1.grid(True, which="major", linestyle="--", alpha=0.7)
    ax1.grid(True, which="minor", linestyle=":", alpha=0.4)
    ax1.set_axisbelow(True)

    # Ensure plot starts from 0
    # ax1.set_ylim(bottom=0)
    # ax2.set_ylim(bottom=0)

    setup_axis_ticks(ax1, dates, granularity)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ",")))

    lines = lines1 + line2
    labels = [l.get_label() for l in lines]

    ax1.legend(
        lines,
        labels,
        loc="upper left",
        title="Metrics",
        fontsize=FONT_SIZES["legend"],
        title_fontsize=FONT_SIZES["legend"],
    )

    save_plot(fig, plots_dir, f"commit_changes_{granularity}")
    plt.close(fig)
"""


def smooth_with_boundary_conditions(dates, values, num_points=300):
    x = np.array([(d - dates[0]).total_seconds() for d in dates])
    x_smooth = np.linspace(x.min(), x.max(), num_points)

    spl = make_interp_spline(x, values, k=3, bc_type="natural")
    y_smooth = spl(x_smooth)

    y_smooth = np.maximum(y_smooth, 0)

    dates_smooth = [dates[0] + pd.Timedelta(seconds=int(xs)) for xs in x_smooth]

    return dates_smooth, y_smooth


def create_log_volume_plot(df, granularity, plots_dir):
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    dates = df.index.to_pydatetime()
    volume = df["insertions"] + df["deletions"]

    from scipy.signal import savgol_filter

    window_length = 5
    poly_order = 1

    volume_smooth = savgol_filter(volume.values, window_length, poly_order)
    insertions_smooth = savgol_filter(
        df["insertions"].values, window_length, poly_order
    )
    deletions_smooth = savgol_filter(df["deletions"].values, window_length, poly_order)

    ax.plot(
        dates,
        volume_smooth,
        label="Total Changes",
        color=MAIN_COLORS[0],
        linewidth=PLOT_LINE_WIDTH,
        alpha=0.8,
    )

    ax.fill_between(
        dates,
        1000,
        insertions_smooth,
        alpha=0.3,
        color=MAIN_COLORS[1],
        label="Additions",
    )

    ax.fill_between(
        dates,
        insertions_smooth,
        insertions_smooth + deletions_smooth,
        alpha=0.3,
        color=MAIN_COLORS[2],
        label="Deletions",
    )

    ax.set_xlabel("")
    ax.set_ylabel(
        "Number of Lines Changed",
        fontsize=FONT_SIZES["axis_label"],
    )

    ax.set_yscale("log")
    ax.set_ylim(bottom=1000)

    apply_grid_style(ax)

    def format_large_numbers(x, p):
        if x >= 1_000_000_000:
            return f"{x/1_000_000_000:.0f}B"
        elif x >= 1_000_000:
            return f"{x/1_000_000:.0f}M"
        elif x >= 1_000:
            return f"{x/1_000:.0f}K"
        else:
            return f"{x:.0f}"

    setup_axis_ticks(ax, dates, granularity, rotation=0)

    setup_legend(ax, title="Changes", loc="lower right")

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))

    save_plot(fig, plots_dir, f"commit_volume_log_{granularity}")
    plt.close(fig)


def create_log_dual_axis_plot(df, granularity, plots_dir):
    setup_plotting_style()
    fig, ax1 = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)
    ax2 = ax1.twinx()

    dates = df.index.to_pydatetime()

    from scipy.signal import savgol_filter

    window_length = 5
    poly_order = 1

    lines1 = []
    for idx, (column, label) in enumerate(
        [("insertions", "Lines Added"), ("deletions", "Lines Deleted")]
    ):
        values_smooth = savgol_filter(df[column].values, window_length, poly_order)
        ln = ax1.plot(
            dates,
            values_smooth,
            label=label,
            color=MAIN_COLORS[idx],
            linewidth=PLOT_LINE_WIDTH,
            alpha=0.8,
        )
        lines1.extend(ln)

    files_smooth = savgol_filter(df["files_changed"].values, window_length, poly_order)
    line2 = ax2.plot(
        dates,
        files_smooth,
        label="Files Changed",
        color=MAIN_COLORS[2],
        linewidth=PLOT_LINE_WIDTH,
        alpha=0.8,
    )

    ax1.set_xlabel("")
    ax1.set_ylabel(
        "Number of Lines Added/Deleted",
        fontsize=FONT_SIZES["axis_label"],
    )
    ax2.set_ylabel(
        "Number of Files Changed",
        fontsize=FONT_SIZES["axis_label"],
    )

    ax1.set_yscale("log")
    ax2.set_yscale("log")

    def format_large_numbers(x, p):
        if x >= 1_000_000_000:
            return f"{x/1_000_000_000:.0f}B"
        elif x >= 1_000_000:
            return f"{x/1_000_000:.0f}M"
        elif x >= 1_000:
            return f"{x/1_000:.0f}K"
        else:
            return f"{x:.0f}"

    apply_grid_style(ax1)

    setup_axis_ticks(ax1, dates, granularity, n_ticks=8)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))

    lines = lines1 + line2
    labels = [l.get_label() for l in lines]

    ax1.legend(
        lines,
        labels,
        loc="lower right",
        title="Metrics",
        fontsize=FONT_SIZES["legend"],
        title_fontsize=FONT_SIZES["legend"],
    )

    save_plot(fig, plots_dir, f"commit_changes_log_{granularity}")
    plt.close(fig)


def analyze_commit_sizes_with_log(args):
    logger.info(f"Analyzing commit sizes using {args.granularity} granularity")

    try:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        df = load_commit_data()
        stats = analyze_high_impact_commits(df)
        quarterly_summary = generate_quarterly_summary(df)

        freq = get_pandas_frequency(args.granularity)

        resampled_df = df.resample(freq, on="author_date").agg(
            {"insertions": "sum", "deletions": "sum", "files_changed": "sum"}
        )

        # create_volume_plot(resampled_df, args.granularity, plots_dir)
        # create_dual_axis_plot(resampled_df, args.granularity, plots_dir)
        create_log_volume_plot(resampled_df, args.granularity, plots_dir)
        create_log_dual_axis_plot(resampled_df, args.granularity, plots_dir)

        export_analysis_results(stats, quarterly_summary, output_dir)

        logger.info("Commit size analysis with log scale plots completed successfully")

    except Exception as e:
        logger.error(f"Error analyzing commit sizes: {str(e)}")
        raise
