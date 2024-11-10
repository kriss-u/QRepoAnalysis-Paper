import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

from ..config.constants import PROCESSED_DATA_DIR, RESULTS_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__, "rq2", "contributor_stars_analysis")


def load_repository_data():
    data_path = PROCESSED_DATA_DIR / "rq1" / "repos.parquet"
    logger.info(f"Loading data from: {data_path}")

    df = pd.read_parquet(data_path)

    df = df[["contributor_count", "stargazers_count"]]
    df = df[df["contributor_count"] > 0]
    df = df[df["stargazers_count"] >= 0]

    df["contributor_star_ratio"] = df["stargazers_count"] / df["contributor_count"]

    logger.info(f"Loaded {len(df)} valid repositories")
    return df


def create_scatter_plot(df, plots_dir):
    logger.info("Creating scatter plot")
    plt.figure(figsize=(10, 8))

    sns.scatterplot(data=df, x="contributor_count", y="stargazers_count", alpha=0.5)

    plt.xscale("log")
    plt.yscale("log")

    plt.xlabel("Number of Contributors (log scale)")
    plt.ylabel("Number of Stars (log scale)")
    plt.title("Repository Stars vs Contributors")

    corr = stats.spearmanr(df["contributor_count"], df["stargazers_count"])[0]
    plt.text(
        0.05,
        0.95,
        f"Spearman Correlation: {corr:.3f}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    save_path = plots_dir / "stars_vs_contributors.png"
    logger.info(f"Saving scatter plot to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_binned_analysis_plot(df, plots_dir):
    logger.info("Creating simplified analysis plot")

    df["contributor_bin"] = pd.cut(
        df["contributor_count"],
        bins=[0, 5, 10, 20, 50, 100, float("inf")],
        labels=["1-5", "6-10", "11-20", "21-50", "51-100", "100+"],
        include_lowest=True,
    )

    stats = (
        df.groupby("contributor_bin", observed=True)
        .agg(
            {
                "stargazers_count": [
                    "mean",
                    "sem",
                    "count",
                ]
            }
        )
        .reset_index()
    )

    stats.columns = ["contributor_bin", "mean_stars", "sem_stars", "count"]

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        range(len(stats)),
        stats["mean_stars"],
        yerr=stats["sem_stars"],
        capsize=5,
        color="skyblue",
        alpha=0.7,
    )

    plt.xticks(range(len(stats)), stats["contributor_bin"], rotation=45, ha="right")
    plt.yscale("log")

    for idx, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f'n={stats["count"][idx]:,}',
            ha="center",
            va="bottom",
            rotation=0,
        )

    plt.xlabel("Number of Contributors")
    plt.ylabel("Average Number of Stars (log scale)")
    plt.title("Average Repository Stars by Contributor Range", pad=20)

    plt.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    save_path = plots_dir / "contributor_analysis.png"
    logger.info(f"Saving analysis plot to {save_path}")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def format_large_number(num):
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)


def calculate_metrics(df):
    metrics = {
        "repository_stats": {
            "total_repositories": len(df),
            "mean_contributors": float(df["contributor_count"].mean()),
            "median_contributors": float(df["contributor_count"].median()),
            "max_contributors": int(df["contributor_count"].max()),
            "min_contributors": int(df["contributor_count"].min()),
            "total_contributors": int(df["contributor_count"].sum()),
        },
        "percentiles": {
            "25th": float(df["contributor_count"].quantile(0.25)),
            "50th": float(df["contributor_count"].quantile(0.50)),
            "75th": float(df["contributor_count"].quantile(0.75)),
            "90th": float(df["contributor_count"].quantile(0.90)),
            "95th": float(df["contributor_count"].quantile(0.95)),
        },
    }

    logger.info(
        f"Total repositories analyzed: {metrics['repository_stats']['total_repositories']:,}"
    )
    return metrics


def compare_repo_contributor(args):
    try:
        output_dir = RESULTS_DIR / "rq1" / "repo_contributors"
        plots_dir = output_dir / "plots"

        output_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Plots directory: {plots_dir}")

        df = load_repository_data()
        metrics = calculate_metrics(df)

        metrics_path = output_dir / "contributor_metrics.json"
        logger.info(f"Saving metrics to {metrics_path}")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info("Analysis completed successfully")

    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        raise
