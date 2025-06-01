import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from scripts.config.constants import PROCESSED_DATA_DIR, PROJECT_ROOT
from scripts.utils.helpers import convert_categorical_to_boolean
from scripts.utils.logger import setup_logger
from ..utils.plot_utils_ieee import (
    GREY_COLORS_DARK,
    setup_plotting_style,
    MAIN_COLORS,
    PAIRED_COLORS,
    FIG_SIZE_SINGLE_COL,
    PLOT_LINE_WIDTH,
    MARKER_SIZE,
    setup_axis_ticks,
    setup_legend,
    save_plot,
    create_pie_chart,
    apply_grid_style,
    FONT_SIZES,
    CATEGORICAL_COLORS,
)

logger = setup_logger(__name__, "rq2", "temporal_patterns")


def load_commit_data():
    try:
        commit_patterns_dir = PROCESSED_DATA_DIR / "rq2" / "commit_patterns"
        if not commit_patterns_dir.exists():
            raise FileNotFoundError(
                f"Commit patterns directory not found: {commit_patterns_dir}"
            )

        year_dirs = sorted(
            [
                d
                for d in commit_patterns_dir.iterdir()
                if d.is_dir() and d.name.isdigit()
            ]
        )

        if not year_dirs:
            raise FileNotFoundError(
                f"No year directories found in {commit_patterns_dir}"
            )

        logger.info(
            f"Found {len(year_dirs)} year directories: {[d.name for d in year_dirs]}"
        )

        dataframes = []
        for year_dir in tqdm(year_dirs, desc="Loading yearly data"):
            try:
                year_df = pd.read_parquet(year_dir)
                year_df = convert_categorical_to_boolean(year_df, "is_fork")
                dataframes.append(year_df)
            except Exception as e:
                logger.error(f"Error loading year {year_dir.name}: {str(e)}")
                continue

        if not dataframes:
            raise ValueError("No data could be loaded from any year")

        df = pd.concat(dataframes, ignore_index=True)
        df["author_date"] = pd.to_datetime(df["author_date"])

        df = df[df["author_date"] < pd.Timestamp("2024-10-01")]

        logger.info("\nDataset Statistics:")
        logger.info(
            f"Time span: {df['author_date'].min().strftime('%Y-%m-%d')} to {df['author_date'].max().strftime('%Y-%m-%d')}"
        )
        logger.info(f"Total commits: {len(df):,}")
        logger.info(f"Original commits: {len(df[~df['is_fork']]):,}")
        logger.info(f"Fork commits: {len(df[df['is_fork']]):,}")

        return df

    except Exception as e:
        logger.error(f"Error loading commit data: {str(e)}")
        raise


def create_fork_comparison_plot(time_series, granularity, output_dir):
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    dates = time_series.index.to_pydatetime()

    from scipy.signal import savgol_filter

    window_length = 5
    poly_order = 3

    original_smooth = savgol_filter(
        time_series["original"].values, window_length, poly_order
    )
    fork_smooth = savgol_filter(time_series["fork"].values, window_length, poly_order)

    ax.plot(
        dates,
        original_smooth,
        label="Original Repositories",
        color=GREY_COLORS_DARK[0],
        linewidth=PLOT_LINE_WIDTH,
        alpha=0.8,
    )

    ax.plot(
        dates,
        fork_smooth,
        label="Fork Repositories",
        color=GREY_COLORS_DARK[6],
        linewidth=PLOT_LINE_WIDTH,
        alpha=0.8,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Number of Commits", fontsize=FONT_SIZES["axis_label"])

    setup_axis_ticks(ax, dates, granularity, n_ticks=8)
    setup_legend(ax, title="Repository Type", loc="upper left", ncol=1)
    apply_grid_style(ax)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_plot(fig, plots_dir, f"original_vs_fork_commits_{granularity}")
    plt.close(fig)


def create_fork_comparison_plot_log_smooth(time_series, granularity, output_dir):
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    dates = time_series.index.to_pydatetime()

    from scipy.signal import savgol_filter

    window_length = 5
    poly_order = 3

    original_smooth = savgol_filter(
        time_series["original"].values, window_length, poly_order
    )
    fork_smooth = savgol_filter(time_series["fork"].values, window_length, poly_order)

    ax.plot(
        dates,
        original_smooth,
        label="Original Repositories",
        color=GREY_COLORS_DARK[0],
        linewidth=PLOT_LINE_WIDTH,
        alpha=0.8,
    )

    ax.plot(
        dates,
        fork_smooth,
        label="Fork Repositories",
        color=GREY_COLORS_DARK[6],
        linewidth=PLOT_LINE_WIDTH,
        alpha=0.8,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Number of Commits", fontsize=FONT_SIZES["axis_label"])

    ax.set_yscale("log")

    setup_axis_ticks(ax, dates, granularity, n_ticks=8)
    setup_legend(ax, title="Repository Type", loc="upper left", ncol=1)
    apply_grid_style(ax)

    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: "0" if x < 1 else f"{int(x):,}")
    )

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_plot(fig, plots_dir, f"original_vs_fork_commits_log_smooth_{granularity}")
    plt.close(fig)


def create_cumulative_comparison_plot(time_series, granularity, output_dir):
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    dates = time_series.index.to_pydatetime()

    from scipy.signal import savgol_filter

    window_length = 5
    poly_order = 3

    original_smooth = savgol_filter(
        time_series["original_cumulative"].values, window_length, poly_order
    )
    fork_smooth = savgol_filter(
        time_series["fork_cumulative"].values, window_length, poly_order
    )

    ax.plot(
        dates,
        original_smooth,
        label="Original Repositories",
        color=GREY_COLORS_DARK[0],
        linewidth=PLOT_LINE_WIDTH,
        alpha=0.8,
    )

    ax.plot(
        dates,
        fork_smooth,
        label="Fork Repositories",
        color=GREY_COLORS_DARK[6],
        linewidth=PLOT_LINE_WIDTH,
        alpha=0.8,
    )

    ax.set_ylabel("Cumulative Number of Commits", fontsize=FONT_SIZES["axis_label"])

    setup_axis_ticks(ax, dates, granularity, n_ticks=8)
    setup_legend(ax, title="Repository Type", loc="upper left", ncol=1)
    apply_grid_style(ax)

    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_plot(fig, plots_dir, f"cumulative_commits_{granularity}")
    plt.close(fig)


def log_comparison_stats(time_series):
    total_original = time_series["original"].sum()
    total_fork = time_series["fork"].sum()
    max_original = time_series["original"].max()
    max_fork = time_series["fork"].max()
    total_commits = time_series["total"].sum()

    logger.info("\nCommit Comparison Statistics:")
    logger.info(f"Total Commits: {total_commits:,}")
    logger.info(
        f"Original Repository Commits: {total_original:,} ({total_original/total_commits*100:.1f}%)"
    )
    logger.info(
        f"Fork Repository Commits: {total_fork:,} ({total_fork/total_commits*100:.1f}%)"
    )
    logger.info(f"Peak Original Repository Commits: {max_original:,}")
    logger.info(f"Peak Fork Repository Commits: {max_fork:,}")
    logger.info(f"Original to Fork Ratio: {total_original/total_fork:.2f}")
    logger.info(
        f"Time span: {time_series.index[0].strftime('%Y-%m-%d')} to {time_series.index[-1].strftime('%Y-%m-%d')}"
    )


def calculate_commit_frequencies(df, freq):
    logger.info(f"Calculating commit frequencies with {freq} frequency")

    time_series = (
        df.groupby([pd.Grouper(key="author_date", freq=freq), "is_fork"], observed=True)
        .size()
        .unstack(fill_value=0)
    )

    time_series.columns = time_series.columns.map({False: "original", True: "fork"})

    time_series["total"] = time_series["original"] + time_series["fork"]
    time_series["original_cumulative"] = time_series["original"].cumsum()
    time_series["fork_cumulative"] = time_series["fork"].cumsum()
    time_series["total_cumulative"] = time_series["total"].cumsum()
    return time_series


def create_total_commits_plot(time_series, granularity, output_dir):
    setup_plotting_style()

    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    dates = time_series.index.to_pydatetime()

    from scipy.interpolate import CubicSpline
    import numpy as np

    date_nums = np.array([d.timestamp() for d in dates])
    values = time_series["total"].values

    cs = CubicSpline(date_nums, values)

    smooth_date_nums = np.linspace(date_nums[0], date_nums[-1], len(date_nums) * 2)
    smooth_dates = [pd.Timestamp.fromtimestamp(ts) for ts in smooth_date_nums]
    smooth_values = cs(smooth_date_nums)

    ax.plot(
        smooth_dates,
        smooth_values,
        color=GREY_COLORS_DARK[0],
        linewidth=PLOT_LINE_WIDTH,
        alpha=0.8,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Number of Commits", fontsize=FONT_SIZES["axis_label"])

    setup_axis_ticks(ax, dates, granularity, n_ticks=8)
    apply_grid_style(ax)

    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    save_plot(fig, plots_dir, f"total_commits_{granularity}")
    plt.close(fig)


def analyze_temporal_patterns(args):
    output_dir = args.output_dir or (
        PROJECT_ROOT / "results" / "rq2" / "fork_comparison"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Analyzing commit patterns using {args.granularity} granularity")

    freq_map = {"week": "W", "month": "ME", "quarter": "QE", "year": "YE"}
    freq = freq_map[args.granularity]

    try:
        logger.info("Loading commit data...")
        df = load_commit_data()

        setup_plotting_style()

        time_series = calculate_commit_frequencies(df, freq)

        logger.info("Generating original vs fork comparison plot...")
        create_fork_comparison_plot(time_series, args.granularity, output_dir)

        logger.info("Generating cumulative comparison plot...")
        create_cumulative_comparison_plot(time_series, args.granularity, output_dir)

        logger.info("Generating total commits plot...")
        create_total_commits_plot(time_series, args.granularity, output_dir)

        logger.info("Generating log scale comparison plot...")
        create_fork_comparison_plot_log_smooth(
            time_series, args.granularity, output_dir
        )

        log_comparison_stats(time_series)
        log_cumulative_stats(time_series)

        logger.info("Commit pattern analysis completed successfully")

    except Exception as e:
        logger.error(f"Error analyzing commit patterns: {str(e)}")
        raise


def log_cumulative_stats(time_series):
    final_original = time_series["original_cumulative"].iloc[-1]
    final_fork = time_series["fork_cumulative"].iloc[-1]
    final_total = time_series["total_cumulative"].iloc[-1]

    logger.info("\nCumulative Commit Statistics:")
    logger.info(f"Total Cumulative Commits: {final_total:,}")
    logger.info(
        f"Original Repository Cumulative Commits: {final_original:,} ({final_original/final_total*100:.1f}%)"
    )
    logger.info(
        f"Fork Repository Cumulative Commits: {final_fork:,} ({final_fork/final_total*100:.1f}%)"
    )
    logger.info(f"Final Original to Fork Ratio: {final_original/final_fork:.2f}")


def create_log_volume_plot(df, granularity, plots_dir):
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    dates = df.index.to_pydatetime()
    volume = df["insertions"] + df["deletions"]

    from scipy.signal import savgol_filter

    window_length = 5
    poly_order = 3

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
        1,
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
        "Number of Lines Changed (log scale)", fontsize=FONT_SIZES["axis_label"]
    )

    ax.set_yscale("log")
    apply_grid_style(ax)

    setup_axis_ticks(ax, dates, granularity, n_ticks=8)
    setup_legend(ax, title="Changes", loc="upper left")

    save_plot(fig, plots_dir, f"commit_volume_log_{granularity}")
    plt.close(fig)


def create_log_dual_axis_plot(df, granularity, plots_dir):
    setup_plotting_style()
    fig, ax1 = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)
    ax2 = ax1.twinx()

    dates = df.index.to_pydatetime()

    from scipy.signal import savgol_filter

    window_length = 5
    poly_order = 3

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
        "Number of Lines Added/Deleted (log scale)", fontsize=FONT_SIZES["axis_label"]
    )
    ax2.set_ylabel(
        "Number of Files Changed (log scale)", fontsize=FONT_SIZES["axis_label"]
    )

    ax1.set_yscale("log")
    ax2.set_yscale("log")

    apply_grid_style(ax1)

    setup_axis_ticks(ax1, dates, granularity, n_ticks=8)

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

    save_plot(fig, plots_dir, f"commit_changes_log_{granularity}")
    plt.close(fig)
