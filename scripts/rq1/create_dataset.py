import json
from datetime import datetime
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from ..config.constants import (
    REPO_CLASSIFIED_FILE,
    REPOS_COMBINED_FILE,
    REPOS_FILE,
    REPOS_DATA_DIR,
    PROCESSED_DATA_DIR,
    REPO_LANGUAGES_FILE,
    REPO_CONTRIBUTORS_FILE,
    REPOS_TO_REMOVE,
)
from ..utils.logger import setup_logger
from ..utils.data_loader import load_language_data, parse_frameworks


logger = setup_logger(__name__, "rq1", "create_dataset")


def validate_repo_record(repo):
    if not isinstance(repo, dict):
        return False, "Record is not a dictionary"

    if "owner" not in repo or "repo" not in repo:
        return False, "Missing owner or repo field"

    if not isinstance(repo.get("is_fork"), bool):
        return False, "Invalid or missing is_fork field"

    if repo["is_fork"]:
        if not repo.get("parent"):
            return False, "Fork missing parent information"

        parent = repo["parent"]
        if not isinstance(parent, dict):
            return False, "Invalid parent record format"

        if "owner" not in parent or "repo" not in parent:
            return False, "Missing parent owner or repo field"

    return True, None


def load_contributor_data(file_path):
    try:
        with open(file_path, "r") as f:
            contributor_map = json.load(f)
        logger.info(f"Loaded contributor data for {len(contributor_map)} repositories")
        return contributor_map
    except Exception as e:
        logger.error(f"Error loading contributor data from {file_path}: {str(e)}")
        return {}


def read_repos_list(file_path, language_stats):
    try:
        # Load repos to remove
        with open(REPOS_TO_REMOVE, "r") as f:
            repos_to_remove = {line.strip() for line in f if line.strip()}
        logger.info(f"Loaded {len(repos_to_remove)} repositories to exclude")

        # Load and filter main repos list
        with open(file_path, "r") as f:
            all_repos = [line.strip() for line in f if line.strip()]

        # Filter repos that have language stats and are not in removal list
        valid_repos = [
            repo
            for repo in all_repos
            if repo in language_stats and repo not in repos_to_remove
        ]

        logger.info(f"Found {len(all_repos)} total repositories")
        logger.info(
            f"Filtered to {len(valid_repos)} repositories with language statistics "
            f"(excluded {len(repos_to_remove)} repositories)"
        )

        if len(valid_repos) == 0:
            logger.warning("No repositories found with language statistics!")

        return valid_repos
    except Exception as e:
        logger.error(f"Error reading repos list from {file_path}: {str(e)}")
        raise


def read_repo_data(repo_id):
    try:
        filename = f"{repo_id.replace('/', '+')}.json"
        direct_path = REPOS_DATA_DIR / filename

        if direct_path.exists():
            with open(direct_path, "r") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
                elif isinstance(data, (list, tuple)) and len(data) > 0:
                    if isinstance(data[0], dict):
                        logger.warning(
                            f"Repository {repo_id} data was in list format, using first entry"
                        )
                        return data[0]
                logger.error(
                    f"Invalid data format for {repo_id}: expected dict, got {type(data)}"
                )
                return None

        target_lower = filename.lower()
        for entry in REPOS_DATA_DIR.iterdir():
            if entry.is_file() and entry.name.lower() == target_lower:
                logger.debug(f"Found case-variant file for {filename}: {entry.name}")
                with open(entry, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
                    elif isinstance(data, (list, tuple)) and len(data) > 0:
                        if isinstance(data[0], dict):
                            logger.warning(
                                f"Repository {repo_id} data was in list format, using first entry"
                            )
                            return data[0]
                    logger.error(
                        f"Invalid data format for {repo_id}: expected dict, got {type(data)}"
                    )
                    return None

        logger.warning(f"Repository data not found for {repo_id}")
        return None

    except FileNotFoundError:
        logger.warning(f"Repository data not found for {repo_id}")
        return None
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON data for {repo_id}")
        return None
    except Exception as e:
        logger.error(f"Error reading data for {repo_id}: {str(e)}")
        return None


def load_fork_relationships(file_path):
    fork_map = {}
    try:
        with open(file_path, "r") as f:
            repos_data = json.load(f)

        for repo in repos_data:
            if repo.get("is_fork") and repo.get("parent"):
                fork_id = f"{repo['owner']}/{repo['repo']}"
                parent_id = f"{repo['parent']['owner']}/{repo['parent']['repo']}"
                fork_map[fork_id.lower()] = parent_id

        logger.info(f"Loaded {len(fork_map)} fork relationships")
        return fork_map
    except Exception as e:
        logger.error(f"Error loading fork relationships: {str(e)}")
        return {}


def load_classified_repos(file_path):
    try:
        df = pd.read_csv(file_path)
        category_map = {
            repo_name.lower(): (repo_name, category)
            for repo_name, category in zip(df["repo_name"], df["category"])
        }
        logger.info(f"Loaded {len(category_map)} repository classifications")
        return category_map
    except Exception as e:
        logger.error(f"Error loading classified repos: {str(e)}")
        return {}


def map_repo_category(repo_id, classified_repos, fork_map):
    repo_id_lower = repo_id.lower()

    # Check if repo is directly classified
    if repo_id_lower in classified_repos:
        return classified_repos[repo_id_lower][1], False, None

    # Check if repo is a fork
    if repo_id_lower in fork_map:
        parent_id = fork_map[repo_id_lower]
        parent_lower = parent_id.lower()

        # Check if parent is classified
        if parent_lower in classified_repos:
            return classified_repos[parent_lower][1], True, parent_id
        return None, True, parent_id

    return None, False, None


def process_repo_data(
    data,
    language_stats,
    contributor_map,
    classified_repos,
    fork_map,
):
    if not data or not isinstance(data, dict):
        return None

    try:
        repo_id = f"{data['owner']['login']}/{data['name']}"

        stats = language_stats.get(repo_id, {})
        frameworks = parse_frameworks(stats.get("frameworks", ""))
        contributors = contributor_map.get(repo_id, [])

        try:
            code_lines = int(stats.get("code", 0))
        except (ValueError, TypeError):
            code_lines = 0
            logger.warning(
                f"Invalid code_lines value for {repo_id}: {stats.get('code')}"
            )

        category, is_fork, parent_repo = map_repo_category(
            repo_id, classified_repos, fork_map
        )

        return {
            "repo_id": repo_id,
            "created_at": datetime.strptime(data["created_at"], "%Y-%m-%dT%H:%M:%SZ"),
            "updated_at": datetime.strptime(data["updated_at"], "%Y-%m-%dT%H:%M:%SZ"),
            "language": stats.get("language", None),
            "stargazers_count": int(data.get("stargazers_count", 0)),
            "forks_count": int(data.get("forks_count", 0)),
            "size": int(data.get("size", 0)),
            "owner_type": data["owner"]["type"],
            "description": data.get("description", None),
            "has_wiki": bool(data.get("has_wiki", False)),
            "has_pages": bool(data.get("has_pages", False)),
            "code_lines": code_lines,
            "packages": frameworks,
            "contributors": contributors,
            "contributor_count": len(contributors),
            "category": category,
            "is_fork": is_fork,
            "parent_repo": parent_repo,
        }

    except Exception as e:
        logger.error(f"Error processing repository data for {repo_id}: {str(e)}")
        return None


def create_dataset(args):
    logger.info("Starting enhanced dataset creation for RQ1")

    logger.info("Loading data sources...")
    language_stats = load_language_data(REPO_LANGUAGES_FILE)
    contributor_map = load_contributor_data(REPO_CONTRIBUTORS_FILE)
    classified_repos = load_classified_repos(REPO_CLASSIFIED_FILE)
    fork_map = load_fork_relationships(REPOS_COMBINED_FILE)

    output_dir = PROCESSED_DATA_DIR / "rq1"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "repos.parquet"
    if dataset_path.exists():
        logger.info(f"Removing existing dataset at {dataset_path}")
        if dataset_path.is_file():
            dataset_path.unlink()
        else:
            import shutil

            shutil.rmtree(dataset_path)

    repos = read_repos_list(REPOS_FILE, language_stats)

    processed_data = []
    category_stats = {"direct": 0, "inherited": 0, "missing": 0}

    for repo_id in tqdm(repos, desc="Processing repositories"):
        raw_data = read_repo_data(repo_id)
        if raw_data:
            processed = process_repo_data(
                raw_data, language_stats, contributor_map, classified_repos, fork_map
            )

            if processed:
                try:
                    assert isinstance(processed["code_lines"], int)
                    assert isinstance(processed["stargazers_count"], int)
                    assert isinstance(processed["forks_count"], int)
                    assert isinstance(processed["size"], int)
                    assert isinstance(processed["contributor_count"], int)
                    assert isinstance(processed["contributors"], list)
                    assert isinstance(processed["is_fork"], bool)

                    if processed["category"]:
                        if processed["is_fork"]:
                            category_stats["inherited"] += 1
                        else:
                            category_stats["direct"] += 1
                    else:
                        category_stats["missing"] += 1

                    processed_data.append(processed)

                except AssertionError:
                    logger.error(f"Invalid data types for {repo_id}")
                    continue

    logger.info(f"Successfully processed {len(processed_data)} repositories")
    logger.info("Category coverage statistics:")
    logger.info(f"  Direct classifications: {category_stats['direct']}")
    logger.info(f"  Inherited from parent: {category_stats['inherited']}")
    logger.info(f"  Missing categories: {category_stats['missing']}")

    try:
        schema = pa.schema(
            [
                ("repo_id", pa.string()),
                ("created_at", pa.timestamp("s")),
                ("updated_at", pa.timestamp("s")),
                ("language", pa.string()),
                ("stargazers_count", pa.int32()),
                ("forks_count", pa.int32()),
                ("size", pa.int64()),
                ("owner_type", pa.string()),
                ("description", pa.string()),
                ("has_wiki", pa.bool_()),
                ("has_pages", pa.bool_()),
                ("code_lines", pa.int32()),
                ("packages", pa.list_(pa.string())),
                ("contributors", pa.list_(pa.string())),
                ("contributor_count", pa.int32()),
                ("category", pa.string()),
                ("is_fork", pa.bool_()),
                ("parent_repo", pa.string()),
            ]
        )

        table = pa.Table.from_pylist(processed_data, schema=schema)

        table = table.append_column(
            "created_year", pa.array([d.year for d in table["created_at"].to_pylist()])
        )

        pq.write_to_dataset(
            table,
            root_path=str(dataset_path),
            partition_cols=["created_year"],
            existing_data_behavior="overwrite_or_ignore",
            compression="snappy",
            use_dictionary=True,
            write_statistics=True,
        )

        logger.info(f"Enhanced dataset successfully created at {dataset_path}")

    except Exception as e:
        logger.error(f"Error creating dataset: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    from argparse import Namespace

    create_dataset(Namespace())
