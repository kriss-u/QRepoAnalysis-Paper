import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import json
from datetime import datetime
import matplotlib.pyplot as plt

from ..config.constants import PROCESSED_DATA_DIR, RAW_DATA_DIR, RESULTS_DIR
from ..utils.logger import setup_logger
from ..utils.plot_utils_ieee import (
    FONT_SIZES,
    FONT_WEIGHT_BOLD,
    setup_plotting_style,
    MAIN_COLORS,
    FIG_SIZE_SINGLE_COL,
    setup_legend,
    save_plot,
    apply_grid_style,
    GREY_COLORS_DARK,
)

logger = setup_logger(__name__, "rq2", "commits_classification")


def load_classified_commit_data(data_dir=PROCESSED_DATA_DIR / "rq2"):
    commit_patterns_dir = data_dir / "commit_patterns_classified"
    if not commit_patterns_dir.exists():
        raise FileNotFoundError(
            f"Classified commit patterns directory not found: {commit_patterns_dir}"
        )

    dfs = []
    required_columns = ["repo_name", "author_date", "category", "is_fork"]

    for class_dir in commit_patterns_dir.iterdir():
        if not class_dir.is_dir():
            continue

        classification = class_dir.name
        logger.info(f"Loading commits for classification: {classification}")

        for year_dir in class_dir.iterdir():
            if not year_dir.is_dir():
                continue

            try:
                df = pq.read_table(year_dir, columns=required_columns).to_pandas()

                df["is_fork"] = (
                    df["is_fork"].astype(str).map({"true": True, "false": False})
                )

                df["classification"] = classification
                dfs.append(df)

            except Exception as e:
                logger.error(
                    f"Error loading data for {classification}/{year_dir.name}: {str(e)}"
                )
                raise

    if not dfs:
        raise ValueError("No classified commit data loaded")

    df = pd.concat(dfs, ignore_index=True)
    df["author_date"] = pd.to_datetime(df["author_date"], utc=True)

    df["is_fork"] = df["is_fork"].astype(bool)

    if df["classification"].isna().any():
        logger.warning("Found commits with missing classifications")
        df["classification"] = df["classification"].fillna("Unknown")

    logger.info("\nDataset Summary:")
    logger.info(f"Total commits: {len(df):,}")
    logger.info(f"Total repositories: {len(df['repo_name'].unique()):,}")
    logger.info("\nFork status distribution:")
    fork_stats = df["is_fork"].value_counts()
    logger.info(f"Original repositories (false): {fork_stats.get(False, 0):,}")
    logger.info(f"Forked repositories (true): {fork_stats.get(True, 0):,}")

    return df


def load_maintenance_data():
    maintenance_file = RAW_DATA_DIR / "high_low_maintenance.csv"
    if not maintenance_file.exists():
        raise FileNotFoundError(
            f"Maintenance classification file not found: {maintenance_file}"
        )

    df = pd.read_csv(maintenance_file)

    logger.info(f"Loaded maintenance data for {len(df)} repositories")
    logger.info(f"Sample repository names: {df['repo_name'].head().tolist()}")

    return df


def calculate_maintenance_statistics(df, maintenance_df):
    df["is_fork"] = df["is_fork"].astype(bool)

    maintenance_map = dict(
        zip(maintenance_df["repo_name"], maintenance_df["maintenance_category"])
    )

    df["maintenance"] = df["repo_name"].map(maintenance_map)

    stats = {
        "total": {
            "repositories": len(df["repo_name"].unique()),
            "commits": len(df),
            "high_maintenance": {
                "repositories": len(
                    df[df["maintenance"] == "High maintenance"]["repo_name"].unique()
                ),
                "commits": len(df[df["maintenance"] == "High maintenance"]),
            },
            "low_maintenance": {
                "repositories": len(
                    df[df["maintenance"] == "Low maintenance"]["repo_name"].unique()
                ),
                "commits": len(df[df["maintenance"] == "Low maintenance"]),
            },
            "unclassified": {
                "repositories": len(df[df["maintenance"].isna()]["repo_name"].unique()),
                "commits": len(df[df["maintenance"].isna()]),
            },
        },
        "by_classification": {},
    }

    for classification in sorted(df["classification"].unique()):
        class_df = df[df["classification"] == classification]

        stats["by_classification"][classification] = {
            "total": {
                "repositories": len(class_df["repo_name"].unique()),
                "commits": len(class_df),
            },
            "high_maintenance": {
                "repositories": len(
                    class_df[class_df["maintenance"] == "High maintenance"][
                        "repo_name"
                    ].unique()
                ),
                "commits": len(class_df[class_df["maintenance"] == "High maintenance"]),
            },
            "low_maintenance": {
                "repositories": len(
                    class_df[class_df["maintenance"] == "Low maintenance"][
                        "repo_name"
                    ].unique()
                ),
                "commits": len(class_df[class_df["maintenance"] == "Low maintenance"]),
            },
            "unclassified": {
                "repositories": len(
                    class_df[class_df["maintenance"].isna()]["repo_name"].unique()
                ),
                "commits": len(class_df[class_df["maintenance"].isna()]),
            },
            "percentage_in_high": (
                round(
                    len(class_df[class_df["maintenance"] == "High maintenance"])
                    / len(class_df)
                    * 100,
                    2,
                )
                if len(class_df) > 0
                else 0
            ),
            "percentage_in_low": (
                round(
                    len(class_df[class_df["maintenance"] == "Low maintenance"])
                    / len(class_df)
                    * 100,
                    2,
                )
                if len(class_df) > 0
                else 0
            ),
        }

    return stats


def create_maintenance_distribution_plot(stats, output_dir):
    setup_plotting_style()

    classifications = []
    high_maint_commits = []
    low_maint_commits = []
    total_commits = []

    for classification, data in stats["by_classification"].items():
        if classification not in ["Unknown", "nan", "Corrective Adaptive Perfective"]:
            classifications.append(classification)
            high_maint_commits.append(data["high_maintenance"]["commits"])
            low_maint_commits.append(data["low_maintenance"]["commits"])
            total_commits.append(data["total"]["commits"])

    sort_idx = np.argsort(total_commits)[::-1]
    classifications = [classifications[i] for i in sort_idx]
    high_maint_commits = [high_maint_commits[i] for i in sort_idx]
    low_maint_commits = [low_maint_commits[i] for i in sort_idx]

    fig, ax = plt.subplots(
        figsize=(FIG_SIZE_SINGLE_COL[0], FIG_SIZE_SINGLE_COL[1] * 1.5)
    )

    x = np.arange(len(classifications))
    width = 0.35

    high_bars = ax.bar(
        x - width / 2,
        high_maint_commits,
        width,
        label="High Maintenance",
        color=GREY_COLORS_DARK[0],
    )
    low_bars = ax.bar(
        x + width / 2,
        low_maint_commits,
        width,
        label="Low Maintenance",
        color=GREY_COLORS_DARK[4],
    )

    ax.set_ylabel(
        "Number of Commits",
        fontsize=FONT_SIZES["axis_label"],
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        classifications, rotation=45, ha="right", fontsize=FONT_SIZES["tick"]
    )

    ax.set_yscale("log")

    ymin, ymax = ax.get_ylim()

    # Start from 1 and go up by orders of magnitude
    ticks = [1]
    current = 10
    while current <= ymax * 100:  # Go two orders higher than max value
        ticks.append(current)
        current *= 10

    ax.set_yticks(ticks)

    def format_large_numbers(x, p):
        if x >= 1_000_000_000:
            return f"{x/1_000_000_000:.0f}B"
        elif x >= 1_000_000:
            return f"{x/1_000_000:.0f}M"
        elif x >= 1_000:
            return f"{x/1_000:.0f}K"
        else:
            return f"{x:.0f}"

    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_large_numbers))

    apply_grid_style(ax)

    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if there are commits
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height * 1.1,  # Add 10% spacing above the bar
                    f"{int(height):,}",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=FONT_SIZES["annotation"],
                    color="#333333",
                )

    add_value_labels(high_bars)
    add_value_labels(low_bars)

    setup_legend(ax, loc="upper right")

    plt.tight_layout()

    save_plot(fig, output_dir, "maintenance_distribution_absolute")
    plt.close(fig)


def calculate_repository_statistics(df):
    df["is_fork"] = df["is_fork"].astype(bool)

    stats = {
        "total": {"repositories": len(df["repo_name"].unique()), "commits": len(df)},
        "by_fork_status": {
            "originals": {
                "repositories": len(df[df["is_fork"] == False]["repo_name"].unique()),
                "commits": len(df[df["is_fork"] == False]),
            },
            "forks": {
                "repositories": len(df[df["is_fork"] == True]["repo_name"].unique()),
                "commits": len(df[df["is_fork"] == True]),
            },
        },
        "by_classification": {},
    }

    for classification in sorted(df["classification"].unique()):
        mask = df["classification"] == classification
        class_df = df[mask]

        stats["by_classification"][classification] = {
            "total": {
                "repositories": len(class_df["repo_name"].unique()),
                "commits": len(class_df),
            },
            "originals": {
                "repositories": len(
                    class_df[class_df["is_fork"] == False]["repo_name"].unique()
                ),
                "commits": len(class_df[class_df["is_fork"] == False]),
            },
            "forks": {
                "repositories": len(
                    class_df[class_df["is_fork"] == True]["repo_name"].unique()
                ),
                "commits": len(class_df[class_df["is_fork"] == True]),
            },
        }

    return stats


def export_repository_statistics(stats, output_dir):
    output_file = output_dir / "repository_statistics.json"

    stats["metadata"] = {
        "generated_at": datetime.now().isoformat(),
        "description": "Repository statistics by classification and fork status",
    }

    try:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics exported to {output_file}")
    except Exception as e:
        logger.error(f"Error exporting statistics: {str(e)}")
        raise


def analyze_commits_classification(args):
    logger.info("Starting commit classification analysis with maintenance categories")

    try:
        output_dir = RESULTS_DIR / "rq2" / "commits_classification"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = load_classified_commit_data()

        maintenance_df = load_maintenance_data()

        stats = calculate_maintenance_statistics(df, maintenance_df)

        create_maintenance_distribution_plot(stats, output_dir)

        stats["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "description": "Commit classification statistics by maintenance category",
        }

        with open(output_dir / "maintenance_statistics.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("\nMaintenance Statistics Summary:")
        logger.info(f"Total repositories: {stats['total']['repositories']:,}")
        logger.info(f"Total commits: {stats['total']['commits']:,}")
        logger.info(
            f"High maintenance repositories: {stats['total']['high_maintenance']['repositories']:,}"
        )
        logger.info(
            f"Low maintenance repositories: {stats['total']['low_maintenance']['repositories']:,}"
        )

        logger.info("\nBreakdown by classification:")
        for classification, class_stats in stats["by_classification"].items():
            logger.info(f"\n{classification}:")
            logger.info(f"  Total commits: {class_stats['total']['commits']:,}")
            logger.info(
                f"  High maintenance commits: {class_stats['high_maintenance']['commits']:,} ({class_stats['percentage_in_high']}%)"
            )
            logger.info(
                f"  Low maintenance commits: {class_stats['low_maintenance']['commits']:,} ({class_stats['percentage_in_low']}%)"
            )

        logger.info("\nAnalysis completed successfully")

    except Exception as e:
        logger.error(
            f"Error analyzing commit classifications with maintenance: {str(e)}"
        )
        raise


if __name__ == "__main__":
    from argparse import Namespace

    args = Namespace(granularity="month")
    analyze_commits_classification(args)
