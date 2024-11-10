import pandas as pd
import pyarrow.parquet as pq
from pathlib import Path
import json
from datetime import datetime
import humanize
from collections import defaultdict
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, defaultdict):
            return dict(obj)
        return super(NumpyEncoder, self).default(obj)


def get_rq1_detailed_stats(df):
    oldest_repo_idx = df["created_at"].idxmin()
    oldest_repo_data = df.loc[oldest_repo_idx]

    stats = {
        "repository_stats": {
            "total_repositories": len(df),
            "oldest_repo_date": df["created_at"].min().strftime("%Y-%m-%d"),
            "oldest_repo_name": oldest_repo_data["repo_id"],
            "newest_repo_date": df["created_at"].max().strftime("%Y-%m-%d"),
            "repository_age_distribution": {
                "min_days": (df["created_at"].max() - df["created_at"].min()).days,
                "avg_days": (df["created_at"].max() - df["created_at"]).mean().days,
                "median_days": (df["created_at"].max() - df["created_at"])
                .median()
                .days,
            },
        },
        "popularity_metrics": {
            "stars": {
                "total": int(df["stargazers_count"].sum()),
                "average": float(df["stargazers_count"].mean()),
                "median": float(df["stargazers_count"].median()),
                "max": int(df["stargazers_count"].max()),
                "max_repo": df.loc[df["stargazers_count"].idxmax(), "repo_id"],
            },
            "forks": {
                "total": int(df["forks_count"].sum()),
                "average": float(df["forks_count"].mean()),
                "median": float(df["forks_count"].median()),
                "max": int(df["forks_count"].max()),
                "max_repo": df.loc[df["forks_count"].idxmax(), "repo_id"],
            },
        },
        "size_metrics": {
            "repository_size": {
                "average_kb": float(df["size"].mean()),
                "median_kb": float(df["size"].median()),
                "total_gb": float(df["size"].sum() / 1024 / 1024),
                "max_mb": float(df["size"].max() / 1024),
                "max_repo": df.loc[df["size"].idxmax(), "repo_id"],
            },
            "code_lines": {
                "total": int(df["code_lines"].sum()),
                "average": float(df["code_lines"].mean()),
                "median": float(df["code_lines"].median()),
                "max": int(df["code_lines"].max()),
                "max_repo": df.loc[df["code_lines"].idxmax(), "repo_id"],
            },
        },
    }

    return stats


def get_dataset_stats(dataset_path, dataset_name):
    stats = {
        "dataset_name": dataset_name,
        "size_bytes": 0,
        "total_rows": 0,
        "total_files": 0,
        "memory_usage": None,
    }

    if not dataset_path.exists():
        return {**stats, "error": "Dataset path does not exist"}

    try:
        files = list(dataset_path.rglob("*.parquet"))
        stats["total_files"] = len(files)
        stats["size_bytes"] = sum(f.stat().st_size for f in files)

        dataset = pq.ParquetDataset(dataset_path)
        df = dataset.read().to_pandas()
        stats["total_rows"] = len(df)

        stats["memory_usage"] = int(df.memory_usage(deep=True).sum())

        if dataset_name == "repos.parquet":
            stats["detailed_stats"] = get_rq1_detailed_stats(df)

        return stats

    except Exception as e:
        return {**stats, "error": str(e)}


def main():
    datasets = [
        "rq1/repos.parquet",
        "rq2/commit_patterns",
        "rq2/commit_patterns_classified",
    ]

    base_dir = Path("data/processed")
    output_dir = base_dir

    results = {"analysis_timestamp": datetime.now().isoformat(), "datasets": {}}

    for dataset_path in datasets:
        name = Path(dataset_path).name
        full_path = base_dir / dataset_path
        print(f"Analyzing dataset: {dataset_path}")
        stats = get_dataset_stats(full_path, name)

        if "size_bytes" in stats:
            stats["size_human"] = humanize.naturalsize(stats["size_bytes"])
        if "memory_usage" in stats and stats["memory_usage"]:
            stats["memory_usage_human"] = humanize.naturalsize(stats["memory_usage"])

        results["datasets"][dataset_path] = stats

    output_file = output_dir / "dataset_statistics.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nStatistics saved to: {output_file}")

    print("\nDataset Summary:")
    for path, stats in results["datasets"].items():
        print(f"\n{path}:")
        print(f"  Total rows: {stats.get('total_rows', 'N/A'):,}")
        print(f"  Total files: {stats.get('total_files', 'N/A'):,}")
        print(f"  Size: {stats.get('size_human', 'N/A')}")

        if "detailed_stats" in stats:
            ds = stats["detailed_stats"]
            print("\nDetailed Repository Statistics:")
            print(
                f"  Total Repositories: {ds['repository_stats']['total_repositories']:,}"
            )
            print(
                f"  Oldest Repository: {ds['repository_stats']['oldest_repo_date']} ({ds['repository_stats']['oldest_repo_name']})"
            )
            print(f"  Stars:")
            print(f"    Average: {ds['popularity_metrics']['stars']['average']:.1f}")
            print(f"    Median: {ds['popularity_metrics']['stars']['median']:.1f}")
            print(
                f"    Max: {ds['popularity_metrics']['stars']['max']:,} ({ds['popularity_metrics']['stars']['max_repo']})"
            )
            print(f"  Forks:")
            print(f"    Average: {ds['popularity_metrics']['forks']['average']:.1f}")
            print(f"    Median: {ds['popularity_metrics']['forks']['median']:.1f}")
            print(
                f"    Max: {ds['popularity_metrics']['forks']['max']:,} ({ds['popularity_metrics']['forks']['max_repo']})"
            )
            print(f"  Repository Size:")
            print(
                f"    Average: {ds['size_metrics']['repository_size']['average_kb']:.1f} KB"
            )
            print(
                f"    Median: {ds['size_metrics']['repository_size']['median_kb']:.1f} KB"
            )
            print(f"  Lines of Code:")
            print(f"    Average: {ds['size_metrics']['code_lines']['average']:,.0f}")
            print(f"    Median: {ds['size_metrics']['code_lines']['median']:,.0f}")
            print(f"    Total: {ds['size_metrics']['code_lines']['total']:,}")
        if "error" in stats:
            print(f"  Error: {stats['error']}")


if __name__ == "__main__":
    main()
