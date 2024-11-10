import os
from datetime import datetime
import shutil
import humanize
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

from scripts.config.constants import (
    COMMITS_CLASSIFIED_FILE,
    DATA_DIR,
    PROJECT_ROOT,
    PROCESSED_DATA_DIR,
)
from scripts.config.rq2_config import (
    COMMIT_PATTERNS_SCHEMA,
    BATCH_SIZE,
    MAX_WORKERS,
    COMMIT_PATTERNS_DIR,
    MIN_BATCH_SIZE,
    MIN_WORKERS,
)
from filelock import FileLock

logger = logging.getLogger(__name__)


def load_commit_classifications(data_dir):
    classification_file = COMMITS_CLASSIFIED_FILE
    if not classification_file.exists():
        logger.warning(f"Commit classification file not found: {classification_file}")
        return {}

    try:
        df = pd.read_csv(classification_file)
        classifications = dict(zip(df["message"].str.lower(), df["classification"]))
        logger.info(f"Loaded {len(classifications)} commit message classifications")
        return classifications
    except Exception as e:
        logger.error(f"Error loading commit classifications: {str(e)}")
        return {}


def classify_commit_message(message, classifications):
    message = message.lower()
    return classifications.get(message, "Unknown")


def write_classified_partition(
    classification,
    year,
    commits,
    output_dir,
    schema,
    row_group_size=100000,
):
    try:
        classification = str(classification).strip()
        year = int(year)

        if not classification:
            classification = "Unknown"

        partition_dir = (
            output_dir / "commit_patterns_classified" / classification / str(year)
        )
        lock_file = partition_dir / f"{classification}_{year}.lock"

        with FileLock(lock_file):
            try:
                partition_dir.mkdir(parents=True, exist_ok=True)

                fork_count = sum(1 for c in commits if c["is_fork"])
                orig_count = sum(1 for c in commits if not c["is_fork"])
                logger.info(
                    f"Classification {classification}, Year {year}: {orig_count} original, {fork_count} fork commits"
                )

                df = pd.DataFrame(commits)
                df["author_date"] = pd.to_datetime(df["author_date"], utc=True)
                df = df.sort_values("author_date")

                logger.info(
                    f"Classification {classification}, Year {year} fork distribution:\n{df['is_fork'].value_counts()}"
                )

                df = df[list(schema.names)]
                table = pa.Table.from_pandas(df, schema=schema)

                pq.write_to_dataset(
                    table,
                    partition_dir,
                    partition_cols=["is_fork"],
                    row_group_size=row_group_size,
                    compression="snappy",
                    use_dictionary=True,
                    write_statistics=True,
                    existing_data_behavior="overwrite_or_ignore",
                )

            except Exception as e:
                logger.error(
                    f"Error writing partition {classification}/{year}: {str(e)}"
                )
                raise
    except Exception as e:
        logger.error(f"Invalid classification or year value: {classification}, {year}")
        logger.error(f"Error details: {str(e)}")
        raise


def process_single_repository(
    repo_dir,
    is_fork,
    excluded_authors,
    classifications,
):
    try:
        log_file = repo_dir / "log.json"
        if not log_file.exists():
            logger.warning(f"No log.json found in {repo_dir}")
            return None

        repo_name = get_repo_name(repo_dir)
        commits_by_classification_year = defaultdict(lambda: defaultdict(list))
        repo_stats = {"filtered_tag_commits": 0, "classifications": defaultdict(int)}

        with open(log_file, "r") as f:
            commits = json.load(f)

            if not commits:
                logger.warning(f"Empty commits list for {repo_dir}")
                return None

            for commit in commits:
                refs = commit.get("refs", "")
                if refs and refs.startswith("tag:"):
                    repo_stats["filtered_tag_commits"] += 1
                    continue

                author_name = commit.get("author_name", "").strip()
                author_email = commit.get("author_email", "").strip()
                if author_name in excluded_authors or author_email in excluded_authors:
                    continue

                author_date = commit["author_date"]
                if author_date.startswith("@"):
                    author_date = author_date[1:]

                try:
                    commit_year = int(pd.to_datetime(author_date).year)
                except Exception as e:
                    logger.warning(f"Invalid date format in {repo_dir}: {author_date}")
                    continue

                classification = str(
                    classify_commit_message(commit.get("subject", ""), classifications)
                ).strip()

                if not classification:
                    classification = "Unknown"

                repo_stats["classifications"][classification] += 1

                files_changed, insertions, deletions = extract_file_stats(commit)

                processed_commit = {
                    "repo_name": repo_name,
                    "is_fork": is_fork,
                    "commit_hash": commit["commit"],
                    "parent_hash": commit.get("parent", ""),
                    "author_name": author_name,
                    "author_email": author_email,
                    "author_date": author_date,
                    "files_changed": files_changed,
                    "insertions": insertions,
                    "deletions": deletions,
                    "subject": commit["subject"],
                    "body": commit.get("body", ""),
                    "category": classification,
                }

                commits_by_classification_year[classification][commit_year].append(
                    processed_commit
                )

        return dict(commits_by_classification_year), repo_stats

    except Exception as e:
        logger.error(f"Error processing repository {repo_dir}: {str(e)}")
        return None


def process_repositories(
    volume_path,
    output_dir,
    batch_size,
    max_workers,
    excluded_repos,
    excluded_authors,
    classifications,
):
    start_time = datetime.now()
    stats = {
        "start_time": start_time,
        "total_repositories_before_filtering": 0,
        "excluded_by_name": 0,
        "excluded_by_author": 0,
        "repositories_after_filtering": 0,
        "total_repositories": 0,
        "processed_repositories": 0,
        "failed_repositories": 0,
        "total_commits": 0,
        "filtered_tag_commits": 0,
        "commits_by_classification": defaultdict(lambda: defaultdict(int)),
        "output_size": 0,
    }

    modified_schema = dict(COMMIT_PATTERNS_SCHEMA)
    modified_schema["category"] = "string"
    schema = pa.schema(
        [(name, pa.type_for_alias(dtype)) for name, dtype in modified_schema.items()]
    )

    try:
        orig_repos = get_valid_repo_dirs(volume_path / "originals", excluded_repos)
        fork_repos = get_valid_repo_dirs(volume_path / "forks", excluded_repos)

        stats["total_repositories_before_filtering"] = len(orig_repos) + len(fork_repos)
        stats["total_repositories"] = len(orig_repos) + len(fork_repos)

        commits_by_classification_year = defaultdict(lambda: defaultdict(list))

        for is_fork, repo_dirs in [(False, orig_repos), (True, fork_repos)]:
            with tqdm(
                total=len(repo_dirs),
                desc=f"Processing {'forks' if is_fork else 'originals'}",
            ) as pbar:

                for i in range(0, len(repo_dirs), batch_size):
                    batch = repo_dirs[i : i + batch_size]

                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(
                                process_single_repository,
                                repo_dir,
                                is_fork,
                                excluded_authors,
                                classifications,
                            ): repo_dir
                            for repo_dir in batch
                        }

                        for future in as_completed(futures):
                            repo_dir = futures[future]
                            try:
                                result = future.result()
                                if result:
                                    classification_commits, repo_stats = result

                                    stats["filtered_tag_commits"] += repo_stats[
                                        "filtered_tag_commits"
                                    ]
                                    for (
                                        classification,
                                        year_commits,
                                    ) in classification_commits.items():
                                        for year, commits in year_commits.items():
                                            commits_by_classification_year[
                                                classification
                                            ][year].extend(commits)
                                            stats["commits_by_classification"][
                                                classification
                                            ][year] += len(commits)
                                            stats["total_commits"] += len(commits)

                                    stats["processed_repositories"] += 1
                                else:
                                    stats["failed_repositories"] += 1
                            except Exception as e:
                                logger.error(
                                    f"Error processing repository {repo_dir}: {str(e)}"
                                )
                                stats["failed_repositories"] += 1
                            pbar.update(1)

                    for (
                        classification,
                        year_commits,
                    ) in commits_by_classification_year.items():
                        for year, commits in year_commits.items():
                            if commits:
                                write_classified_partition(
                                    classification, year, commits, output_dir, schema
                                )

                    commits_by_classification_year.clear()

        stats["end_time"] = datetime.now()
        stats["processing_time"] = stats["end_time"] - stats["start_time"]
        stats["output_size"] = sum(
            f.stat().st_size for f in output_dir.rglob("*.parquet")
        )

        return stats

    except Exception as e:
        logger.error(f"Error in repository processing: {str(e)}")
        raise


def create_dataset(args):
    volume_path = args.volume_path
    output_dir = args.output_dir or PROCESSED_DATA_DIR / "rq2"
    batch_size = args.batch_size

    (output_dir / "commit_patterns_classified").mkdir(parents=True, exist_ok=True)

    commit_patterns_dir = output_dir / "commit_patterns_classified"
    if commit_patterns_dir.exists():
        logger.info(f"Removing existing dataset: {commit_patterns_dir}")
        try:
            shutil.rmtree(commit_patterns_dir)
            commit_patterns_dir.mkdir(parents=True)
        except Exception as e:
            logger.error(f"Error removing existing dataset: {str(e)}")
            raise

    if not volume_path.exists():
        raise ValueError(f"Volume path does not exist: {volume_path}")

    logger.info(f"Starting classified dataset creation from {volume_path}")
    logger.info(f"Output directory: {output_dir}")

    excluded_repos = load_excluded_repos(DATA_DIR)
    excluded_authors = load_excluded_authors(DATA_DIR)
    classifications = load_commit_classifications(DATA_DIR)

    logger.info(
        f"Loaded {len(excluded_repos)} excluded repositories, "
        f"{len(excluded_authors)} excluded authors, and "
        f"{len(classifications)} commit classifications"
    )

    batch_size, max_workers = validate_processing_params(batch_size, MAX_WORKERS)

    try:
        stats = process_repositories(
            volume_path=volume_path,
            output_dir=output_dir,
            batch_size=batch_size,
            max_workers=max_workers,
            excluded_repos=excluded_repos,
            excluded_authors=excluded_authors,
            classifications=classifications,
        )

        stats["start_time"] = stats["start_time"].isoformat()
        stats["end_time"] = stats["end_time"].isoformat()
        stats["processing_time"] = str(stats["processing_time"])

        with open(output_dir / "processing_stats.json", "w") as f:
            json.dump(stats, f, indent=2)

        logger.info("\nProcessing Statistics:")
        logger.info(
            f"Total repositories before filtering: {stats['total_repositories_before_filtering']}"
        )
        logger.info(f"Successfully processed: {stats['processed_repositories']}")
        logger.info(f"Failed: {stats['failed_repositories']}")
        logger.info("\nCommits by classification:")
        for classification, year_counts in sorted(
            stats["commits_by_classification"].items()
        ):
            total = sum(year_counts.values())
            logger.info(f"\n{classification}:")
            for year, count in sorted(year_counts.items()):
                logger.info(f"  {year}: {count:,} commits")
            logger.info(f"  Total: {total:,} commits")
        logger.info(f"\nTotal processing time: {stats['processing_time']}")
        logger.info(f"Output size: {humanize.naturalsize(stats['output_size'])}")

    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise

    logger.info("Classified dataset creation completed successfully")


def get_repo_name(repo_dir):
    dir_name = repo_dir.name
    owner, repo = dir_name.split("+", 1)
    return f"{owner}/{repo}"


def extract_file_stats(commit):
    if isinstance(commit.get("stats"), dict):
        return (
            commit["stats"].get("files_changed", 0),
            commit["stats"].get("insertions", 0),
            commit["stats"].get("deletions", 0),
        )
    return (
        commit.get("files_changed", 0),
        commit.get("insertions", 0),
        commit.get("deletions", 0),
    )


def get_valid_repo_dirs(base_path, excluded_repos):
    if not base_path.exists():
        logger.warning(f"Path does not exist: {base_path}")
        return []

    valid_dirs = []
    for repo_dir in base_path.iterdir():
        repo_name = get_repo_name(repo_dir)
        if repo_name not in excluded_repos:
            valid_dirs.append(repo_dir)
        else:
            logger.debug(f"Excluding repository by name: {repo_name}")

    logger.info(f"Found {len(valid_dirs)} valid repositories after name filtering")
    return valid_dirs


def load_excluded_repos(data_dir):
    excluded_file = data_dir / "raw" / "repos_removed.txt"
    if not excluded_file.exists():
        logger.warning(f"Excluded repos file not found: {excluded_file}")
        return set()

    with open(excluded_file) as f:
        excluded = {line.strip() for line in f if line.strip()}

    logger.info(f"Loaded {len(excluded)} repositories to exclude")
    return excluded


def load_excluded_authors(data_dir):
    excluded_file = data_dir / "raw" / "authors_removed.txt"
    if not excluded_file.exists():
        logger.warning(f"Excluded authors file not found: {excluded_file}")
        return {""}

    with open(excluded_file) as f:
        excluded = {""}
        excluded.update(line.strip() for line in f)

    logger.info(f"Loaded {len(excluded)} authors to exclude")
    return excluded


def validate_processing_params(batch_size, max_workers):
    validated_batch = max(MIN_BATCH_SIZE, min(batch_size, 10000))
    validated_workers = max(MIN_WORKERS, min(max_workers, os.cpu_count() or 1))

    if validated_batch != batch_size:
        logger.warning(f"Adjusted batch size from {batch_size} to {validated_batch}")
    if validated_workers != max_workers:
        logger.warning(
            f"Adjusted worker count from {max_workers} to {validated_workers}"
        )

    return validated_batch, validated_workers
