import json
from datetime import datetime
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

from scripts.config.constants import PROCESSED_DATA_DIR, RAW_DATA_DIR

ISSUES_SCHEMA = {
    "repo_id": pa.string(),
    "issue_number": pa.int32(),
    "title": pa.string(),
    "body": pa.string(),
    "comment_count": pa.int32(),
    "created_at": pa.timestamp("s"),
    "updated_at": pa.timestamp("s"),
    "closed_at": pa.timestamp("s"),
    "is_locked": pa.bool_(),
    "lock_reason": pa.string(),
    "reaction_count": pa.int32(),
    "user_login": pa.string(),
    "user_type": pa.string(),
    "comments": pa.list_(
        pa.struct(
            [
                ("id", pa.int64()),
                ("body", pa.string()),
                ("created_at", pa.timestamp("s")),
                ("updated_at", pa.timestamp("s")),
                ("user_login", pa.string()),
                ("user_type", pa.string()),
                ("reaction_count", pa.int32()),
            ]
        )
    ),
}


def load_excluded_repos(file_path):
    try:
        with open(file_path) as f:
            return {line.strip() for line in f if line.strip()}
    except Exception as e:
        logging.error(f"Error loading excluded repos: {str(e)}")
        return set()


def parse_datetime(dt_str):
    if not dt_str:
        return None
    try:
        return datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def process_comment(comment):
    return {
        "id": comment["id"],
        "body": comment.get("body", ""),
        "created_at": parse_datetime(comment.get("created_at")),
        "updated_at": parse_datetime(comment.get("updated_at")),
        "user_login": comment.get("user", {}).get("login", ""),
        "user_type": comment.get("user", {}).get("type", ""),
        "reaction_count": comment.get("reactions", {}).get("total_count", 0),
    }


def process_issue(issue, repo_id):
    return {
        "repo_id": repo_id,
        "issue_number": issue["number"],
        "title": issue.get("title", ""),
        "body": issue.get("body", ""),
        "comment_count": issue.get("comments", 0),
        "created_at": parse_datetime(issue.get("created_at")),
        "updated_at": parse_datetime(issue.get("updated_at")),
        "closed_at": parse_datetime(issue.get("closed_at")),
        "is_locked": bool(issue.get("active_lock_reason", False)),
        "lock_reason": issue.get("active_lock_reason", ""),
        "reaction_count": issue.get("reactions", {}).get("total_count", 0),
        "user_login": issue.get("user", {}).get("login", ""),
        "user_type": issue.get("user", {}).get("type", ""),
        "comments": [
            process_comment(comment) for comment in issue.get("comment_data", [])
        ],
    }


def read_issues_file(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            logging.warning(f"Invalid issues data format in {file_path}")
            return None
    except Exception as e:
        logging.error(f"Error reading issues file {file_path}: {str(e)}")
        return None


def process_repositories(raw_data_dir, excluded_repos, batch_size=1000):
    current_batch = []

    for file_path in raw_data_dir.glob("*.json"):
        try:
            repo_id = file_path.stem.replace("+", "/")

            if repo_id in excluded_repos:
                continue

            issues = read_issues_file(file_path)
            if not issues:
                continue

            for issue in issues:
                processed_issue = process_issue(issue, repo_id)
                current_batch.append(processed_issue)

                if len(current_batch) >= batch_size:
                    yield current_batch
                    current_batch = []

        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")
            continue

    if current_batch:
        yield current_batch


def create_issues_dataset(raw_data_dir, output_dir, excluded_repos_file):
    logging.info("Starting issues dataset creation")

    excluded_repos = load_excluded_repos(excluded_repos_file)
    logging.info(f"Loaded {len(excluded_repos)} excluded repositories")

    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if output_dir.exists():
        if output_dir.is_file():
            output_dir.unlink()
        else:
            import shutil

            shutil.rmtree(output_dir)

    schema = pa.schema([(name, dtype) for name, dtype in ISSUES_SCHEMA.items()])

    total_issues = 0
    for batch in tqdm(
        process_repositories(raw_data_dir, excluded_repos),
        desc="Processing repositories",
    ):
        try:
            table = pa.Table.from_pylist(batch, schema=schema)

            created_year = pa.array(
                [d.year if d else None for d in table["created_at"].to_pylist()]
            )
            created_month = pa.array(
                [d.month if d else None for d in table["created_at"].to_pylist()]
            )

            table = table.append_column("year", created_year)
            table = table.append_column("month", created_month)

            pq.write_to_dataset(
                table,
                root_path=str(output_dir),
                partition_cols=["year", "month"],
                existing_data_behavior="overwrite_or_ignore",
                compression="snappy",
                use_dictionary=True,
                write_statistics=True,
            )

            total_issues += len(batch)

        except Exception as e:
            logging.error(f"Error writing batch: {str(e)}")
            continue

    logging.info(f"Successfully processed {total_issues} issues")
    logging.info(f"Dataset created at {output_dir}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    raw_data_dir = RAW_DATA_DIR / "issues_including_replies"
    processed_data_dir = PROCESSED_DATA_DIR / "issues.parquet"
    excluded_repos_file = RAW_DATA_DIR / "repos_removed.txt"

    create_issues_dataset(raw_data_dir, processed_data_dir, excluded_repos_file)
