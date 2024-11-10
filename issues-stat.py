import json
import logging
from datetime import datetime
import pandas as pd

from scripts.config.constants import RAW_DATA_DIR


def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


def load_excluded_repos(data_dir):
    excluded_file = data_dir / "repos_removed.txt"
    if not excluded_file.exists():
        l.warning(f"Excluded repos file not found: {excluded_file}")
        return set()

    with open(excluded_file) as f:
        excluded = {line.strip() for line in f if line.strip()}

    l.info(f"Loaded {len(excluded)} repositories to exclude")
    return excluded


def process_issue_file(file_path):
    try:
        with open(file_path, "r") as f:
            issues = json.load(f)

        if not issues:
            return None

        stats = {
            "issue_count": len(issues),
            "total_comments": 0,
            "users": set(),
            "created_dates": [],
            "closed_dates": [],
            "open_issues": 0,
        }

        for issue in issues:
            # Count comments
            stats["total_comments"] += issue.get("comments", 0)

            # Track unique users
            if issue.get("user") and issue["user"].get("login"):
                stats["users"].add(issue["user"]["login"])

            # Add comment authors to unique users
            for comment in issue.get("comment_data", []):
                if comment.get("user") and comment["user"].get("login"):
                    stats["users"].add(comment["user"]["login"])

            # Track dates
            if issue.get("created_at"):
                stats["created_dates"].append(
                    datetime.strptime(issue["created_at"], "%Y-%m-%dT%H:%M:%SZ")
                )
            if issue.get("closed_at") and issue["closed_at"] is not None:
                stats["closed_dates"].append(
                    datetime.strptime(issue["closed_at"], "%Y-%m-%dT%H:%M:%SZ")
                )
            else:
                stats["open_issues"] += 1

        return stats

    except Exception as e:
        l.error(f"Error processing file {file_path}: {str(e)}")
        return None


def analyze_issues(issues_dir, excluded_repos):
    l.info("Starting issue analysis...")

    if not issues_dir.exists():
        l.error(f"Issues directory not found: {issues_dir}")
        return

    all_stats = []
    skipped_repos = 0

    for file_path in issues_dir.glob("*.json"):
        repo_name = file_path.stem.replace("+", "/")

        if repo_name in excluded_repos:
            skipped_repos += 1
            l.info(f"Skipping excluded repository: {repo_name}")
            continue

        l.info(f"Processing {file_path.name}")
        stats = process_issue_file(file_path)
        if stats:
            stats["repo"] = repo_name
            all_stats.append(stats)

    if not all_stats:
        l.warning("No valid issue data found")
        return

    l.info(f"Skipped {skipped_repos} excluded repositories")

    total_stats = {
        "total_repositories": len(all_stats),
        "total_issues": sum(s["issue_count"] for s in all_stats),
        "total_comments": sum(s["total_comments"] for s in all_stats),
        "total_unique_users": len(set().union(*[s["users"] for s in all_stats])),
        "open_issues": sum(s["open_issues"] for s in all_stats),
        "date_range": {
            "earliest": min(
                min(s["created_dates"]) for s in all_stats if s["created_dates"]
            ),
            "latest": max(
                max(s["created_dates"]) for s in all_stats if s["created_dates"]
            ),
        },
    }

    repo_summaries = []
    for stats in all_stats:
        summary = {
            "repository": stats["repo"],
            "issue_count": stats["issue_count"],
            "comment_count": stats["total_comments"],
            "unique_users": len(stats["users"]),
            "open_issues": stats["open_issues"],
            "first_issue": min(stats["created_dates"]).strftime("%Y-%m-%d"),
            "last_issue": max(stats["created_dates"]).strftime("%Y-%m-%d"),
        }
        repo_summaries.append(summary)

    output_file = RAW_DATA_DIR / "issue_statistics.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "overall_statistics": {
                    **total_stats,
                    "excluded_repositories": skipped_repos,
                    "date_range": {
                        "earliest": total_stats["date_range"]["earliest"].strftime(
                            "%Y-%m-%d"
                        ),
                        "latest": total_stats["date_range"]["latest"].strftime(
                            "%Y-%m-%d"
                        ),
                    },
                },
                "repository_statistics": repo_summaries,
            },
            f,
            indent=2,
        )

    summary_df = pd.DataFrame(repo_summaries)
    summary_df.to_csv(RAW_DATA_DIR / "issue_statistics.csv", index=False)

    l.info(
        f"Analysis complete. Found {total_stats['total_issues']} issues across {total_stats['total_repositories']} repositories"
    )
    l.info(
        f"Date range: {total_stats['date_range']['earliest'].strftime('%Y-%m-%d')} to {total_stats['date_range']['latest'].strftime('%Y-%m-%d')}"
    )
    l.info(f"Total comments: {total_stats['total_comments']}")
    l.info(f"Total unique users: {total_stats['total_unique_users']}")
    l.info(f"Currently open issues: {total_stats['open_issues']}")
    l.info(f"Excluded repositories: {skipped_repos}")
    l.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    l = setup_logger(__name__)

    ISSUES_DIR = RAW_DATA_DIR / "issues_including_replies"
    excl_repos = load_excluded_repos(RAW_DATA_DIR)

    try:
        analyze_issues(ISSUES_DIR, excl_repos)
    except Exception as e:
        l.error(f"Analysis failed: {str(e)}", exc_info=True)
