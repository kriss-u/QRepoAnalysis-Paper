import json
from pathlib import Path
import dask.dataframe as dd
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging

# Import constants from your project
from scripts.config.constants import DATA_DIR, RESULTS_DIR

logger = logging.getLogger(__name__)


def load_repository_mapping() -> Tuple[Dict[str, str], Dict[str, List[str]]]:
    """Load fork to parent repository mapping and create reverse mapping."""
    mapping_file = DATA_DIR / "raw" / "repos_combined_final.json"

    with open(mapping_file) as f:
        repos_data = json.load(f)

    fork_to_parent = {}
    parent_to_forks = {}

    for repo in repos_data:
        if repo.get("is_fork") and "parent" in repo:
            fork_name = f"{repo['owner']}/{repo['repo']}"
            parent_name = f"{repo['parent']['owner']}/{repo['parent']['repo']}"
            fork_to_parent[fork_name] = parent_name

            if parent_name not in parent_to_forks:
                parent_to_forks[parent_name] = []
            parent_to_forks[parent_name].append(fork_name)

    return fork_to_parent, parent_to_forks


def calculate_fork_metrics(ddf: dd.DataFrame) -> pd.DataFrame:
    """Calculate only commit count for each fork."""
    logger.info("Calculating fork commit counts...")

    commit_counts = ddf.groupby("repo_name").agg({"commit_hash": "count"}).compute()

    # Rename column for clarity
    commit_counts.columns = ["commit_count"]

    return commit_counts


def calculate_global_threshold(fork_metrics: pd.DataFrame) -> float:
    """Calculate global minimum threshold using median + 2*MAD."""
    commit_counts = fork_metrics["commit_count"].values

    # Calculate median and MAD
    median = np.median(commit_counts)
    mad = np.median(np.abs(commit_counts - median))

    # Use median + 2*MAD as threshold
    global_threshold = median + 2 * mad

    logger.info(f"Global stats - Median: {median:.2f}, MAD: {mad:.2f}")
    logger.info(f"Global minimum threshold: {global_threshold:.2f}")

    return global_threshold


def detect_high_activity_forks(
    parent: str,
    group_df: pd.DataFrame,
    global_min_threshold: float,
    percentile: float = 97.5,
) -> List[Dict]:
    """Detect forks with high commit counts within a parent repository group."""
    if len(group_df) < 2:
        return []

    high_activity_forks = []
    parent_threshold = group_df["commit_count"].quantile(percentile / 100)

    # Use maximum of global and parent-specific thresholds
    final_threshold = max(global_min_threshold, parent_threshold)

    for fork_name in group_df.index:
        commit_count = group_df.loc[fork_name, "commit_count"]

        if commit_count > final_threshold:
            high_activity_forks.append(
                {
                    "fork_name": fork_name,
                    "metrics": {"commit_count": int(commit_count)},
                    "analysis": {
                        "threshold": int(final_threshold),
                        "percentile": percentile,
                    },
                }
            )

    return high_activity_forks


def organize_results(
    fork_metrics: pd.DataFrame,
    fork_to_parent: Dict[str, str],
    parent_to_forks: Dict[str, List[str]],
    percentile: float = 99,
) -> Dict[str, List[Dict]]:
    """Organize results by parent repository."""
    # Calculate global threshold
    global_min_threshold = calculate_global_threshold(fork_metrics)

    organized_results = {}

    for parent, forks in parent_to_forks.items():
        parent_forks_metrics = fork_metrics[fork_metrics.index.isin(forks)]

        if len(parent_forks_metrics) > 0:
            high_activity_forks = detect_high_activity_forks(
                parent, parent_forks_metrics, global_min_threshold, percentile
            )

            if high_activity_forks:
                organized_results[parent] = high_activity_forks

    return organized_results


def write_results(results_dir: Path, organized_results: Dict[str, List[Dict]]) -> None:
    """Write results to text and JSON files."""
    # Write text file with grouped forks
    with open(results_dir / "high_activity_forks.txt", "w") as f:
        for parent_forks in organized_results.values():
            # Write all forks for this parent together
            for fork_data in parent_forks:
                f.write(f"{fork_data['fork_name']}\n")
            # Add blank line between different parents' forks
            f.write("\n")

    # Write detailed JSON with parent organization
    with open(results_dir / "high_activity_forks_details.json", "w") as f:
        json.dump(organized_results, f, indent=2, default=str)


def main():
    """Main function to detect high-activity forks."""
    try:
        results_dir = RESULTS_DIR / "rq2"
        results_dir.mkdir(parents=True, exist_ok=True)

        # Load repository mappings
        fork_to_parent, parent_to_forks = load_repository_mapping()

        # Read parquet files
        parquet_path = str(
            DATA_DIR
            / "processed"
            / "rq2"
            / "commit_patterns"
            / "*"
            / "is_fork=true"
            / "*.parquet"
        )
        logger.info(f"Reading parquet files from: {parquet_path}")

        ddf = dd.read_parquet(
            parquet_path,
            engine="pyarrow",
            columns=["repo_name", "commit_hash"],  # Only needed columns
        )

        # Calculate metrics for all forks
        fork_metrics = calculate_fork_metrics(ddf)
        logger.info(f"Calculated commit counts for {len(fork_metrics)} forks")

        # Organize results by parent repository
        organized_results = organize_results(
            fork_metrics, fork_to_parent, parent_to_forks
        )

        # Write results
        write_results(results_dir, organized_results)

        logger.info(
            f"Analysis complete. Found high-activity forks in "
            f"{len(organized_results)} parent repositories."
        )

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
