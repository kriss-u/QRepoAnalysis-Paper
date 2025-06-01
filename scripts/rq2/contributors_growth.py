import pandas as pd
import pyarrow.parquet as pq
import json
import matplotlib.pyplot as plt

from ..config.constants import PROCESSED_DATA_DIR, RESULTS_DIR
from ..utils.logger import setup_logger
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
)

logger = setup_logger(__name__, "rq2", "contributors_growth")


def get_pandas_frequency(granularity):
    freq_map = {"week": "W", "month": "ME", "quarter": "QE", "year": "YE"}
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
    required_columns = ["author_date", "author_email", "repo_name"]

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
        f"Loaded {len(df):,} commits from {len(df['author_email'].unique()):,} unique authors"
    )
    return df


def analyze_contributors_growth(args):
    logger.info(f"Analyzing author patterns using {args.granularity} granularity")

    try:
        output_dir = RESULTS_DIR / "rq2" / "authors"
        output_dir.mkdir(parents=True, exist_ok=True)

        plots_dir = output_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        df = load_commit_data()
        results_df, stats = analyze_authors_over_time(df, args.granularity)

        create_author_trends_plot(results_df, args.granularity, plots_dir)

        export_analysis_results(results_df, stats, output_dir)

        logger.info("Author analysis completed successfully")

    except Exception as e:
        logger.error(f"Error analyzing authors: {str(e)}")
        raise


def analyze_authors_over_time(df, granularity):
    freq = get_pandas_frequency(granularity)

    first_appearances = df.groupby("author_email")["author_date"].min().reset_index()
    first_appearances = first_appearances.set_index("author_date")

    author_counts = first_appearances.resample(freq).count()

    results = []
    total_authors = 0

    for period, count in author_counts["author_email"].items():
        total_authors += count
        results.append(
            {
                "period": period,
                "new_authors": int(count),  # New authors in this period
                "total_authors": total_authors,  # Total unique authors up to this period
            }
        )

    results_df = pd.DataFrame(results)
    results_df.set_index("period", inplace=True)

    # Fill any missing periods with zeros for new_authors
    # but forward fill total_authors
    results_df["new_authors"] = results_df["new_authors"].fillna(0).astype(int)
    results_df["total_authors"] = results_df["total_authors"].ffill().astype(int)

    stats = {
        "total_unique_authors": len(df["author_email"].unique()),
        "average_new_authors_per_period": float(results_df["new_authors"].mean()),
        "max_new_authors_period": {
            "period": results_df["new_authors"].idxmax().strftime("%Y-%m"),
            "count": int(results_df["new_authors"].max()),
        },
        "periods_with_new_authors": int((results_df["new_authors"] > 0).sum()),
        "total_periods": len(results_df),
    }

    return results_df, stats


def create_author_trends_plot(df, granularity, plots_dir):
    setup_plotting_style()
    fig, ax1 = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    dates = df.index
    time_diff = (dates.max() - dates.min()).days
    n_periods = len(dates)
    width = max(time_diff / n_periods * 0.8, 15)

    bars = ax1.bar(
        dates,
        df["new_authors"],
        width=width,
        alpha=0.6,
        color=GREY_COLORS_DARK[6],
        label="New Contributors",
    )

    ax1.set_xlabel("")
    ax1.set_ylabel(f"New Contributors per {granularity.title()}")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    line = ax2.plot(
        dates,
        df["total_authors"],
        color=GREY_COLORS_DARK[0],
        linewidth=PLOT_LINE_WIDTH,
        label="Total Contributors",
    )

    ax2.set_ylabel("Total Contributors")
    ax2.tick_params(axis="y")

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

    setup_axis_ticks(ax1, dates.to_list(), granularity, n_ticks=8)
    apply_grid_style(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    # Position legend below chart with proper sizing
    legend = ax2.legend(
        lines1 + lines2,
        labels1 + labels2,
        bbox_to_anchor=(0, -0.15, 1, 0.1),
        loc="upper center",
        mode="expand",
        ncol=2,
        frameon=True,
        fontsize=FONT_SIZES["legend"],
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_alpha(0.9)
    legend.get_frame().set_edgecolor("#CCCCCC")

    plt.title(f"Contributors Growth Over Time")

    # Adjust layout to accommodate legend below
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)

    save_plot(fig, plots_dir, f"author_growth_{granularity}")
    plt.close(fig)


def export_analysis_results(results_df, stats, output_dir):
    period_data = [
        {
            "period": period.strftime("%Y-%m-%d"),
            "new_authors": int(row["new_authors"]),
            "total_authors": int(row["total_authors"]),
        }
        for period, row in results_df.iterrows()
    ]

    results = {"summary_statistics": stats, "period_data": period_data}

    with open(output_dir / "author_growth.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace(granularity="month")
    analyze_contributors_growth(args)
