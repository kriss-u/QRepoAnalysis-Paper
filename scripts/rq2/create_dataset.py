import logging
import os
from datetime import datetime
import shutil
import humanize
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from collections import defaultdict

from scripts.config.constants import DATA_DIR, PROCESSED_DATA_DIR
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


def write_year_partition(year, commits, output_dir, schema, row_group_size=100000):
    year_dir = output_dir / "commit_patterns" / str(year)
    lock_file = output_dir / "commit_patterns" / f"{year}.lock"

    with FileLock(lock_file):
        try:
            year_dir.mkdir(parents=True, exist_ok=True)

            fork_count = sum(1 for c in commits if c["is_fork"])
            orig_count = sum(1 for c in commits if not c["is_fork"])
            logger.info(
                f"Year {year} pre-processing: {orig_count} original, {fork_count} fork commits"
            )

            df = pd.DataFrame(commits)
            df["author_date"] = pd.to_datetime(df["author_date"], utc=True)
            df = df.sort_values("author_date")

            logger.info(
                f"Year {year} is_fork distribution:\n{df['is_fork'].value_counts()}"
            )

            df = df[list(schema.names)]
            table = pa.Table.from_pandas(df, schema=schema)

            temp_dir = year_dir / "temp"
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_file = temp_dir / f"part_{os.getpid()}.parquet"

            pq.write_table(
                table,
                temp_file,
                compression="snappy",
                use_dictionary=True,
                write_statistics=True,
            )

            existing_files = list(temp_dir.glob("*.parquet"))
            if existing_files:
                tables = [pq.read_table(f) for f in existing_files]
                merged_table = pa.concat_tables(tables)

                pq.write_to_dataset(
                    merged_table,
                    year_dir,
                    partition_cols=["is_fork"],
                    row_group_size=row_group_size,
                    compression="snappy",
                    use_dictionary=True,
                    write_statistics=True,
                    existing_data_behavior="overwrite_or_ignore",
                )

                shutil.rmtree(temp_dir)

        except Exception as e:
            logger.error(f"Error writing year partition {year}: {str(e)}")
            raise
        finally:
            if (year_dir / "temp").exists():
                shutil.rmtree(year_dir / "temp")


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


def check_repository_authors(repo_dir, excluded_authors):
    try:
        log_file = repo_dir / "log.json"
        if not log_file.exists():
            logger.warning(f"No log.json found in {repo_dir}")
            return False, repo_dir

        with open(log_file, "r") as f:
            commits = json.load(f)

        for commit in commits:
            author_name = commit.get("author_name", "").strip()
            author_email = commit.get("author_email", "").strip()

            if author_name in excluded_authors or author_email in excluded_authors:
                logger.debug(
                    f"Excluding {get_repo_name(repo_dir)} due to author: {author_name} / {author_email}"
                )
                return False, repo_dir

        return True, repo_dir

    except Exception as e:
        logger.error(f"Error checking authors for repository {repo_dir}: {str(e)}")
        return False, repo_dir


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


def process_single_repository(repo_dir, is_fork, excluded_authors):
    try:
        log_file = repo_dir / "log.json"
        if not log_file.exists():
            logger.warning(f"No log.json found in {repo_dir}")
            return None

        repo_name = get_repo_name(repo_dir)
        commits_by_year = defaultdict(list)
        repo_stats = {"filtered_tag_commits": 0}
        chunk_size = 1000

        with open(log_file, "r") as f:
            commits_list = json.load(f)

            if not commits_list:
                logger.warning(f"Empty commits list for {repo_dir}")
                return None

            for i in range(0, len(commits_list), chunk_size):
                chunk = commits_list[i : i + chunk_size]

                for commit in chunk:
                    refs = commit.get("refs", "")
                    if refs and refs.startswith("tag:"):
                        repo_stats["filtered_tag_commits"] += 1
                        continue

                    author_name = commit.get("author_name", "").strip()
                    author_email = commit.get("author_email", "").strip()

                    if (
                        author_name in excluded_authors
                        or author_email in excluded_authors
                    ):
                        continue

                    author_date = commit["author_date"]
                    if author_date.startswith("@"):
                        author_date = author_date[1:]

                    try:
                        commit_year = pd.to_datetime(author_date).year
                    except Exception as e:
                        logger.warning(
                            f"Invalid date format in {repo_dir}: {author_date}"
                        )
                        continue

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
                    }
                    commits_by_year[commit_year].append(processed_commit)

                del chunk

        return dict(commits_by_year), repo_stats

    except Exception as e:
        logger.error(f"Error processing repository {repo_dir}: {str(e)}")
        return None


def write_year_partition(year, commits, output_dir, schema, row_group_size=100000):
    try:
        year_dir = output_dir / "commit_patterns" / str(year)
        temp_dir = year_dir / "temp"
        year_dir.mkdir(parents=True, exist_ok=True)
        temp_dir.mkdir(parents=True, exist_ok=True)

        chunk_size = min(row_group_size, 10000)
        for i in range(0, len(commits), chunk_size):
            chunk = commits[i : i + chunk_size]

            df = pd.DataFrame(chunk)
            df["author_date"] = pd.to_datetime(df["author_date"], utc=True)
            df = df.sort_values("author_date")
            df = df[list(schema.names)]

            table = pa.Table.from_pandas(df, schema=schema)

            temp_file = temp_dir / f"part_{os.getpid()}_{i}.parquet"
            pq.write_table(
                table,
                temp_file,
                compression="snappy",
                use_dictionary=True,
                write_statistics=True,
            )

            del df
            del table

        temp_files = list(temp_dir.glob("*.parquet"))
        if temp_files:
            tables = []
            for f in temp_files:
                table = pq.read_table(f)
                tables.append(table)

            merged_table = pa.concat_tables(tables)

            pq.write_to_dataset(
                merged_table,
                year_dir,
                partition_cols=["is_fork"],
                row_group_size=row_group_size,
                compression="snappy",
                use_dictionary=True,
                write_statistics=True,
                existing_data_behavior="overwrite_or_ignore",
            )

            del tables
            del merged_table
            shutil.rmtree(temp_dir)

    except Exception as e:
        logger.error(f"Error writing year partition {year}: {str(e)}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def process_repositories(
    volume_path, output_dir, batch_size, max_workers, excluded_repos, excluded_authors
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
        "commits_per_year": defaultdict(int),
        "output_size": 0,
    }

    schema = pa.schema(
        [
            (name, pa.type_for_alias(dtype))
            for name, dtype in COMMIT_PATTERNS_SCHEMA.items()
        ]
    )

    try:
        orig_path = volume_path / "originals"
        fork_path = volume_path / "forks"

        orig_repos = list(orig_path.iterdir()) if orig_path.exists() else []
        fork_repos = list(fork_path.iterdir()) if fork_path.exists() else []

        stats["total_repositories_before_filtering"] = len(orig_repos) + len(fork_repos)

        orig_repos = [r for r in orig_repos if get_repo_name(r) not in excluded_repos]
        fork_repos = [r for r in fork_repos if get_repo_name(r) not in excluded_repos]

        stats["excluded_by_name"] = (
            stats["total_repositories_before_filtering"]
            - len(orig_repos)
            - len(fork_repos)
        )
        stats["repositories_after_filtering"] = len(orig_repos) + len(fork_repos)
        stats["total_repositories"] = stats["repositories_after_filtering"]

        for is_fork, repo_dirs in [(False, orig_repos), (True, fork_repos)]:
            with tqdm(
                total=len(repo_dirs),
                desc=f"Processing {'forks' if is_fork else 'originals'}",
            ) as pbar:
                for i in range(0, len(repo_dirs), batch_size):
                    batch = repo_dirs[i : i + batch_size]
                    commits_by_year = defaultdict(list)

                    with ProcessPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(
                                process_single_repository,
                                repo_dir,
                                is_fork,
                                excluded_authors,
                            ): repo_dir
                            for repo_dir in batch
                        }

                        for future in as_completed(futures):
                            repo_dir = futures[future]
                            try:
                                result = future.result()
                                if isinstance(result, tuple):
                                    year_commits, repo_stats = result
                                    stats["filtered_tag_commits"] += repo_stats.get(
                                        "filtered_tag_commits", 0
                                    )
                                    if year_commits:
                                        for year, commits in year_commits.items():
                                            commits_by_year[year].extend(commits)
                                            stats["commits_per_year"][year] += len(
                                                commits
                                            )
                                            stats["total_commits"] += len(commits)
                                        stats["processed_repositories"] += 1
                                    else:
                                        stats["failed_repositories"] += 1
                                else:
                                    logger.warning(
                                        f"Unexpected result format from {repo_dir}"
                                    )
                                    stats["failed_repositories"] += 1
                            except Exception as e:
                                logger.error(
                                    f"Error processing repository {repo_dir}: {str(e)}"
                                )
                                stats["failed_repositories"] += 1
                            pbar.update(1)

                    for year, commits in commits_by_year.items():
                        if commits:
                            write_year_partition(year, commits, output_dir, schema)

                    del commits_by_year

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

    (output_dir / "commit_patterns").mkdir(parents=True, exist_ok=True)

    commit_patterns_dir = output_dir / "commit_patterns"
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

    logger.info(f"Starting temporally optimized dataset creation from {volume_path}")
    logger.info(f"Output directory: {output_dir}")

    excluded_repos = load_excluded_repos(DATA_DIR)
    excluded_authors = load_excluded_authors(DATA_DIR)

    logger.info(
        f"Loaded {len(excluded_repos)} excluded repositories and {len(excluded_authors)} excluded authors"
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
        logger.info(f"Excluded by name: {stats['excluded_by_name']}")
        logger.info(f"Excluded by author: {stats['excluded_by_author']}")
        logger.info(
            f"Repositories after filtering: {stats['repositories_after_filtering']}"
        )
        logger.info(f"Successfully processed: {stats['processed_repositories']}")
        logger.info(f"Failed: {stats['failed_repositories']}")
        logger.info("\nCommits per year:")
        for year, count in sorted(stats["commits_per_year"].items()):
            logger.info(f"{year}: {count:,} commits")
        logger.info(f"\nTotal processing time: {stats['processing_time']}")
        logger.info(f"Output size: {humanize.naturalsize(stats['output_size'])}")

    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}")
        raise

    logger.info("Dataset creation completed successfully")
