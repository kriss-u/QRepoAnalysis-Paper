import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from ..config.constants import PROCESSED_DATA_DIR, RESULTS_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__, "rq1", "popularity")


def load_repository_data():
    try:
        dataset_path = PROCESSED_DATA_DIR / "rq1" / "repos.parquet"
        df = pd.read_parquet(dataset_path)

        df["created_at"] = pd.to_datetime(df["created_at"])
        df = df.sort_values("created_at")

        logger.info(f"Loaded {len(df)} repositories")
        return df

    except Exception as e:
        logger.error(f"Error loading repository data: {str(e)}")
        raise


def resample_by_granularity(df, granularity):
    freq_map = {"week": "W", "month": "ME", "quarter": "QE", "year": "YE"}
    freq = freq_map[granularity]

    period_counts = df.resample(freq, on="created_at").size().reset_index()
    period_counts.columns = ["period", "new_repos"]

    cumulative_counts = period_counts.copy()
    cumulative_counts["total_repos"] = cumulative_counts["new_repos"].cumsum()
    cumulative_counts = cumulative_counts.drop("new_repos", axis=1)

    return period_counts, cumulative_counts


def calculate_growth_metrics(df):
    metrics = {
        "total_repositories": len(df),
        "avg_repositories_per_year": len(df)
        / ((df["created_at"].max() - df["created_at"].min()).days / 365.25),
        "first_repository_date": df["created_at"].min(),
        "latest_repository_date": df["created_at"].max(),
        "years_active": (df["created_at"].max() - df["created_at"].min()).days / 365.25,
    }
    return metrics


def create_visualizations(period_df, cumulative_df, granularity, metrics, output_dir):
    plt.style.use("seaborn-v0_8")
    # sns.set_style("whitegrid")

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: New repositories per period
    plt.figure(figsize=(12, 6))
    plt.bar(period_df["period"], period_df["new_repos"], alpha=0.7, color="royalblue")
    plt.title(f"New Quantum Computing Repositories per {granularity.title()}")
    plt.xlabel(f"{granularity.title()}")
    plt.ylabel("Number of New Repositories")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plots_dir / f"new_repos_per_{granularity}.png")
    plt.close()

    # Figure 2: Cumulative growth
    plt.figure(figsize=(12, 6))
    plt.plot(
        cumulative_df["period"],
        cumulative_df["total_repos"],
        color="darkblue",
        linewidth=2,
    )
    plt.title("Cumulative Growth of Quantum Computing Repositories")
    plt.xlabel(f"{granularity.title()}")
    plt.ylabel("Total Number of Repositories")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plots_dir / f"cumulative_growth_{granularity}.png")
    plt.close()


def save_results(period_df, cumulative_df, metrics, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    period_df.to_csv(output_dir / "period_counts.csv", index=False)
    cumulative_df.to_csv(output_dir / "cumulative_counts.csv", index=False)

    metrics_output = {
        k: str(v) if isinstance(v, datetime) else v for k, v in metrics.items()
    }
    pd.DataFrame([metrics_output]).to_json(
        output_dir / "growth_metrics.json", orient="records", indent=2
    )


def analyze_popularity(args):
    logger.info(f"Analyzing popularity with {args.granularity} granularity")

    try:
        output_dir = RESULTS_DIR / "rq1" / "popularity"
        output_dir.mkdir(parents=True, exist_ok=True)

        df = load_repository_data()
        metrics = calculate_growth_metrics(df)
        logger.info(
            f"Found {metrics['total_repositories']} repositories "
            f"over {metrics['years_active']:.1f} years"
        )

        period_df, cumulative_df = resample_by_granularity(df, args.granularity)
        create_visualizations(
            period_df, cumulative_df, args.granularity, metrics, output_dir
        )
        save_results(period_df, cumulative_df, metrics, output_dir)

        logger.info("Popularity analysis completed successfully")

    except Exception as e:
        logger.error(f"Error analyzing popularity: {str(e)}")
        raise
