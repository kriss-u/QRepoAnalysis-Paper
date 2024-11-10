import dask.dataframe as dd
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from scripts.config.constants import RESULTS_DIR


def json_serial(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif (
        isinstance(obj, dict)
        and "email" in str(obj.get("email", "")).lower()
        and pd.isna(obj["email"])
    ):
        obj["email"] = "null"
        return obj
    elif pd.isna(obj):
        return None
    raise TypeError(f"Type {type(obj)} not serializable")


def create_directory_structure(base_dir):
    dirs = {
        "stats": base_dir / "stats",
        "data": base_dir / "data",
        "json": base_dir / "json",
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def calculate_author_detailed_stats(author_metrics):
    return {
        "contribution_summary": {
            "total_authors": int(len(author_metrics)),
            "total_commits": int(author_metrics["num_commits"].sum()),
            "total_insertions": int(author_metrics["total_insertions"].sum()),
            "total_deletions": int(author_metrics["total_deletions"].sum()),
            "total_changes": int(author_metrics["total_changes"].sum()),
        },
        "activity_metrics": {
            "avg_commits_per_author": float(author_metrics["num_commits"].mean()),
            "median_commits_per_author": float(author_metrics["num_commits"].median()),
            "avg_repos_per_author": float(author_metrics["num_repos"].mean()),
            "median_repos_per_author": float(author_metrics["num_repos"].median()),
        },
        "time_metrics": {
            "avg_activity_days": float(author_metrics["activity_days"].mean()),
            "median_activity_days": float(author_metrics["activity_days"].median()),
            "max_activity_days": int(author_metrics["activity_days"].max()),
            "min_activity_days": int(author_metrics["activity_days"].min()),
        },
        "code_impact": {
            "avg_changes_per_commit": float(
                author_metrics["total_changes"].sum()
                / author_metrics["num_commits"].sum()
            ),
            "avg_files_per_commit": float(
                author_metrics["avg_files_per_commit"].mean()
            ),
        },
    }


def calculate_top_author_stats(top_authors, author_names):
    top_author_stats = []

    for email in top_authors.index:
        author_data = top_authors.loc[email]
        author_stat = {
            "email": email,
            "names": author_names[email].tolist() if email in author_names else [],
            "metrics": {
                "commits": int(author_data["num_commits"]),
                "repositories": int(author_data["num_repos"]),
                "total_insertions": int(author_data["total_insertions"]),
                "total_deletions": int(author_data["total_deletions"]),
                "activity_days": int(author_data["activity_days"]),
                "avg_files_per_commit": float(author_data["avg_files_per_commit"]),
            },
            "timespan": {
                "first_commit": author_data["first_commit"],
                "last_commit": author_data["last_commit"],
            },
        }
        top_author_stats.append(author_stat)

    return {
        "top_authors": top_author_stats,
        "summary": {
            "total_commits": int(top_authors["num_commits"].sum()),
            "avg_commits": float(top_authors["num_commits"].mean()),
            "total_repositories": int(top_authors["num_repos"].sum()),
            "avg_repositories": float(top_authors["num_repos"].mean()),
            "total_changes": int(top_authors["total_changes"].sum()),
        },
    }


def calculate_repo_metrics_stats(repo_metrics):
    return {
        "repository_summary": {
            "total_repos": int(
                len(repo_metrics.index.get_level_values("repo_name").unique())
            ),
            "avg_commits_per_repo": float(repo_metrics["num_commits"].mean()),
            "median_commits_per_repo": float(repo_metrics["num_commits"].median()),
            "total_commits": int(repo_metrics["num_commits"].sum()),
        },
        "activity_patterns": {
            "avg_activity_days": float(repo_metrics["activity_days"].mean()),
            "median_activity_days": float(repo_metrics["activity_days"].median()),
            "max_activity_days": int(repo_metrics["activity_days"].max()),
        },
        "code_changes": {
            "total_insertions": int(repo_metrics["total_insertions"].sum()),
            "total_deletions": int(repo_metrics["total_deletions"].sum()),
            "avg_changes_per_repo": float(repo_metrics["total_changes"].mean()),
        },
    }


def analyze_authors(commit_patterns_dir, output_dir=None, top_n=20):
    logger = logging.getLogger(__name__)

    base_dir = output_dir or RESULTS_DIR / "rq1" / "author_analysis"
    dirs = create_directory_structure(base_dir)

    logger.info(f"Reading commit patterns from {commit_patterns_dir}")
    start_time = datetime.now()

    df = dd.read_parquet(
        commit_patterns_dir,
        engine="pyarrow",
        columns=[
            "repo_name",
            "author_name",
            "author_email",
            "author_date",
            "files_changed",
            "insertions",
            "deletions",
        ],
    )

    total_repos = df["repo_name"].nunique().compute()
    logger.info(f"Found {total_repos} repositories in dataset")

    logger.info("Aggregating overall author metrics...")
    basic_metrics = (
        df.groupby("author_email")
        .agg(
            {
                "author_date": ["min", "max"],
                "files_changed": ["count", "mean"],
                "insertions": "sum",
                "deletions": "sum",
            }
        )
        .compute()
    )

    logger.info("Computing repository counts per author...")
    repo_counts = (
        df.groupby("author_email")["repo_name"]
        .apply(lambda x: x.nunique(), meta=("repo_name", "int64"))
        .compute()
    )

    logger.info("Combining metrics and calculating derived statistics...")
    author_metrics = basic_metrics.copy()
    author_metrics.columns = [
        "first_commit",
        "last_commit",
        "num_commits",
        "avg_files_per_commit",
        "total_insertions",
        "total_deletions",
    ]
    author_metrics["num_repos"] = repo_counts

    author_metrics["activity_days"] = (
        author_metrics["last_commit"] - author_metrics["first_commit"]
    ).dt.days
    author_metrics["total_changes"] = (
        author_metrics["total_insertions"] + author_metrics["total_deletions"]
    )

    logger.info(f"Identifying top {top_n} authors by commit count...")
    top_authors = author_metrics.sort_values("num_commits", ascending=False)

    logger.info("Computing repository-level metrics for top authors...")
    top_authors_df = df[df.author_email.isin(top_authors.index)]
    repo_metrics = (
        top_authors_df.groupby(["author_email", "repo_name"])
        .agg(
            {
                "author_date": ["min", "max"],
                "files_changed": "count",
                "insertions": "sum",
                "deletions": "sum",
            }
        )
        .compute()
    )

    logger.info("Processing author names...")
    author_names = (
        top_authors_df.groupby("author_email")
        .author_name.apply(lambda x: x.unique(), meta=("author_name", "object"))
        .compute()
    )

    repo_metrics.columns = [
        "first_commit",
        "last_commit",
        "num_commits",
        "total_insertions",
        "total_deletions",
    ]

    repo_metrics["activity_days"] = (
        repo_metrics["last_commit"] - repo_metrics["first_commit"]
    ).dt.days
    repo_metrics["total_changes"] = (
        repo_metrics["total_insertions"] + repo_metrics["total_deletions"]
    )

    logger.info("Calculating comprehensive statistics...")
    processing_time = datetime.now() - start_time

    stats = {
        "metadata": {
            "analysis_timestamp": datetime.now(),
            "commit_patterns_dir": str(commit_patterns_dir),
            "top_n_authors": top_n,
            "total_repos_analyzed": total_repos,
            "processing_time_seconds": processing_time.total_seconds(),
        },
        "overall_stats": calculate_author_detailed_stats(author_metrics),
        "top_author_analysis": calculate_top_author_stats(top_authors, author_names),
        "repository_metrics": calculate_repo_metrics_stats(repo_metrics),
        "analysis_parameters": {
            "repository_count": total_repos,
            "date_range": {
                "start": author_metrics["first_commit"].min(),
                "end": author_metrics["last_commit"].max(),
            },
        },
    }

    logger.info("Saving JSON statistics...")
    json_path = dirs["json"] / "author_analysis_stats.json"
    with open(json_path, "w") as f:
        json.dump(stats, f, default=json_serial, indent=2)

    logger.info(
        f"\nAnalysis complete. Processed {total_repos:,} repositories in {processing_time}"
    )
    logger.info(f"Found {len(author_metrics):,} unique authors")
    logger.info(
        f"Total commits analyzed: {stats['overall_stats']['contribution_summary']['total_commits']:,}"
    )
    logger.info(f"JSON statistics saved to: {json_path}")
