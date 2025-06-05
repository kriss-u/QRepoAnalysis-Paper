import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

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

logger = setup_logger(__name__, "rq1", "popularity_updated")

# Annotation configuration - easy to adjust
ANNOTATION_CONFIG = {
    "line_color": "black",
    "line_width": 0.8,
    "line_alpha": 0.8,
    "text_color": "black",
    "text_style": "italic",
    "text_weight": "normal",
    "vertical_spacing": 0.03,
    "events": [
        {
            "date": "2017-03-07",
            "label": "Qiskit released",
            "text_x_days": 400,
            "text_y_fraction": 0.32,
            "ha": "right",
        },
        {
            "date": "2018-07-18",
            "label": "Cirq released",
            "text_x_days": 400,
            "text_y_fraction": 0.45,
            "ha": "right",
        },
        {
            "date": "2018-12-21",
            "label": "US National Quantum Act",
            "text_x_days": 1100,
            "text_y_fraction": 0.65,
            "ha": "right",
        },
    ],
}


def load_repository_data():
    try:
        dataset_path = PROCESSED_DATA_DIR / "rq1" / "repos.parquet"
        df = pd.read_parquet(dataset_path)

        df["created_at"] = pd.to_datetime(df["created_at"])
        df["updated_at"] = pd.to_datetime(df["updated_at"])

        df = df[df["created_at"] < pd.Timestamp("2024-10-01")]
        df = df[df["updated_at"] < pd.Timestamp("2024-10-01")]

        df = df.sort_values("created_at")

        logger.info(f"Loaded {len(df)} repositories")
        return df

    except Exception as e:
        logger.error(f"Error loading repository data: {str(e)}")
        raise


def resample_activity_data(df, granularity):
    freq_map = {"week": "W", "month": "ME", "quarter": "QE", "year": "YE"}
    freq = freq_map[granularity]

    new_repos = df.resample(freq, on="created_at").size().reset_index()
    new_repos.columns = ["period", "new_repos"]

    updated_repos = df.resample(freq, on="updated_at").size().reset_index()
    updated_repos.columns = ["period", "updated_repos"]

    period_activity = pd.merge(
        new_repos, updated_repos, on="period", how="outer"
    ).fillna(0)

    cumulative_metrics = period_activity.copy()
    cumulative_metrics["total_repos"] = cumulative_metrics["new_repos"].cumsum()
    cumulative_metrics["total_updates"] = cumulative_metrics["updated_repos"].cumsum()

    cumulative_metrics["period"] = period_activity["period"]

    return period_activity, cumulative_metrics


def calculate_activity_metrics(df):
    metrics = {
        "total_repositories": len(df),
        "maintained_repos": len(df[df["updated_at"] > df["created_at"]]),
        "never_updated": len(df[df["updated_at"] == df["created_at"]]),
        "avg_repositories_per_year": len(df)
        / ((df["created_at"].max() - df["created_at"].min()).days / 365.25),
        "first_repository_date": df["created_at"].min(),
        "latest_repository_date": df["created_at"].max(),
        "latest_update_date": df["updated_at"].max(),
        "years_active": (df["created_at"].max() - df["created_at"].min()).days / 365.25,
        "avg_lifetime_days": (df["updated_at"] - df["created_at"]).mean().days,
        "median_lifetime_days": (df["updated_at"] - df["created_at"]).median().days,
    }
    return metrics


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


def create_combined_activity_plot(
    period_df,
    granularity,
    plots_dir,
):
    fig, ax1 = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    time_diff = (period_df["period"].max() - period_df["period"].min()).days
    n_periods = len(period_df)
    width = max(time_diff / n_periods * 0.8, 15)

    bars = ax1.bar(
        period_df["period"],
        period_df["new_repos"],
        width=width,
        alpha=0.6,
        color=GREY_COLORS_DARK[6],
        label="New Repositories",
    )

    ax1.set_xlabel("")
    ax1.set_ylabel(f"New Repositories per {granularity.title()}")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    line = ax2.plot(
        period_df["period"],
        period_df["new_repos"].cumsum(),
        color=GREY_COLORS_DARK[0],
        linewidth=PLOT_LINE_WIDTH,
        label="Total Repositories",
    )

    ax2.set_ylabel("Total Number of Repositories")
    ax2.tick_params(axis="y")

    def format_thousands(x, p):
        if x >= 1000:
            return f"{x/1000:.0f}K"
        return f"{x:.0f}"

    ax2.yaxis.set_major_formatter(plt.FuncFormatter(format_thousands))

    y_min, y_max = ax1.get_ylim()

    for event_config in ANNOTATION_CONFIG["events"]:
        event_date = pd.Timestamp(event_config["date"])

        closest_period = period_df.iloc[
            (period_df["period"] - event_date).abs().argsort()[:1]
        ]

        if not closest_period.empty:
            period_date = closest_period["period"].iloc[0]
            bar_height = closest_period["new_repos"].iloc[0]

            text_x = period_date + pd.Timedelta(days=event_config["text_x_days"])
            text_y = y_max * event_config["text_y_fraction"]
            line_end_y = text_y - (y_max * ANNOTATION_CONFIG["vertical_spacing"])

            ax1.plot(
                [period_date, period_date],
                [bar_height, line_end_y],
                color=ANNOTATION_CONFIG["line_color"],
                linestyle="-",
                linewidth=ANNOTATION_CONFIG["line_width"],
                alpha=ANNOTATION_CONFIG["line_alpha"],
            )

            ax1.text(
                text_x,
                text_y,
                event_config["label"],
                rotation=0,
                ha=event_config["ha"],
                va="center",
                fontsize=FONT_SIZES["annotation"],
                style=ANNOTATION_CONFIG["text_style"],
                color=ANNOTATION_CONFIG["text_color"],
                weight=ANNOTATION_CONFIG["text_weight"],
            )

    dates = period_df["period"].to_list()
    setup_axis_ticks(ax1, dates, granularity, n_ticks=8)
    apply_grid_style(ax1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    # plt.title(f"Repository Growth Over Time")

    save_plot(fig, plots_dir, f"combined_activity_{granularity}")


def create_scatter_plot(df, plots_dir):
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    scatter_color = PAIRED_COLORS[1]
    ax.scatter(
        df["created_at"],
        df["updated_at"],
        alpha=0.4,
        s=MARKER_SIZE * 5,
        color=scatter_color,
    )

    min_date = min(df["created_at"].min(), df["updated_at"].min())
    max_date = max(df["created_at"].max(), df["updated_at"].max())
    ax.plot(
        [min_date, max_date],
        [min_date, max_date],
        linestyle="--",
        color=PAIRED_COLORS[5],
        linewidth=PLOT_LINE_WIDTH,
        label="Created = Last Updated",
    )

    ax.set_title("Repository Creation Date vs Last Update Date")
    ax.set_xlabel("Creation Date")
    ax.set_ylabel("Last Update Date")
    apply_grid_style(ax)
    setup_legend(ax)

    save_plot(fig, plots_dir, "creation_vs_update_scatter")


def create_lifetime_distribution(df, plots_dir):
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    lifetime_days = (df["updated_at"] - df["created_at"]).dt.total_seconds() / (
        24 * 60 * 60
    )
    sns.histplot(data=lifetime_days, bins=50, ax=ax, color=MAIN_COLORS[2])

    ax.set_title("Distribution of Repository Lifetimes")
    ax.set_xlabel("Lifetime (days)")
    ax.set_ylabel("Number of Repositories")
    apply_grid_style(ax)

    save_plot(fig, plots_dir, "lifetime_distribution")


def create_activity_timeline(period_df, df, granularity, plots_dir):
    fig, ax = plt.subplots(figsize=FIG_SIZE_SINGLE_COL)

    ax.plot(
        period_df["period"],
        period_df["new_repos"].cumsum(),
        label="Total Repositories",
        color=MAIN_COLORS[0],
        linewidth=PLOT_LINE_WIDTH,
    )

    maintenance_threshold = pd.Timedelta(days=180)
    period_df["maintained_repos"] = period_df["period"].apply(
        lambda x: len(
            df[
                (df["created_at"] <= x)
                & (df["updated_at"] >= x - maintenance_threshold)
            ]
        )
    )

    ax.plot(
        period_df["period"],
        period_df["maintained_repos"],
        label="Recently Updated Repositories",
        color=MAIN_COLORS[1],
        linewidth=PLOT_LINE_WIDTH,
    )

    ax.set_title("Repository Growth and Maintenance Over Time")
    ax.set_xlabel(f"{granularity.title()}")
    ax.set_ylabel("Number of Repositories")

    setup_axis_ticks(ax, period_df["period"].to_list(), granularity, n_ticks=6)
    apply_grid_style(ax)
    setup_legend(ax)

    save_plot(fig, plots_dir, f"activity_timeline_{granularity}")


def create_maintenance_status_chart(df, plots_dir):
    maintenance_status = pd.cut(
        (df["updated_at"] - df["created_at"]).dt.total_seconds() / (24 * 60 * 60),
        bins=[-float("inf"), 0, 30, 180, 365, float("inf")],
        labels=["Never Updated", "1 Month", "6 Months", "1 Year", "Over 1 Year"],
    ).value_counts()

    fig, ax = create_pie_chart(
        data=maintenance_status.values,
        labels=maintenance_status.index,
        title="Repository Maintenance Status Distribution",
        explode=[0.05, 0.02, 0.02, 0.02, 0.02],
    )

    save_plot(fig, plots_dir, "maintenance_status_pie")


def create_visualizations(
    period_df,
    df,
    granularity,
    metrics,
    output_dir,
):
    setup_plotting_style()

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Create all visualizations
    create_combined_activity_plot(period_df, granularity, plots_dir)
    # create_scatter_plot(df, plots_dir)
    # create_lifetime_distribution(df, plots_dir)
    # create_activity_timeline(period_df, df, granularity, plots_dir)
    # create_maintenance_status_chart(df, plots_dir)

    with open(plots_dir / "visualization_summary.txt", "w") as f:
        f.write("Repository Activity Analysis Summary\n")
        f.write("===================================\n\n")
        f.write(f"Total Repositories: {len(df):,}\n")
        f.write(
            f"Never Updated Repositories: {len(df[df['created_at'] == df['updated_at']]):,}\n"
        )
        f.write(
            f"Median Lifetime: {(df['updated_at'] - df['created_at']).median().days:,} days\n"
        )
        f.write(
            f"Average Lifetime: {(df['updated_at'] - df['created_at']).mean().days:,.1f} days\n"
        )


def save_results(
    period_df,
    cumulative_df,
    metrics,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    period_df.to_csv(output_dir / "period_activity.csv", index=False)
    cumulative_df.to_csv(output_dir / "cumulative_activity.csv", index=False)

    metrics_output = {
        k: str(v) if isinstance(v, datetime) else v for k, v in metrics.items()
    }
    pd.DataFrame([metrics_output]).to_json(
        output_dir / "activity_metrics.json", orient="records", indent=2
    )


def analyze_popularity_updated(args):
    logger.info(
        f"Analyzing popularity with updates using {args.granularity} granularity"
    )

    try:
        output_dir = RESULTS_DIR / "rq1" / "popularity_updated"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = load_repository_data()
        metrics = calculate_activity_metrics(df)

        logger.info(
            f"Found {metrics['total_repositories']} repositories with "
            f"{metrics['maintained_repos']} maintained repositories over {metrics['years_active']:.1f} years"
        )

        period_df, cumulative_df = resample_activity_data(df, args.granularity)

        create_visualizations(period_df, df, args.granularity, metrics, output_dir)

        save_results(period_df, cumulative_df, metrics, output_dir)

        logger.info("Repository activity analysis completed successfully")

    except Exception as e:
        logger.error(f"Error analyzing repository activity: {str(e)}")
        raise


if __name__ == "__main__":
    from argparse import Namespace

    analyze_popularity_updated(Namespace(granularity="quarter"))
