COMMIT_PATTERNS_SCHEMA = {
    "repo_name": "string",
    "is_fork": "bool",
    "commit_hash": "string",
    "parent_hash": "string",
    "author_name": "string",
    "author_email": "string",
    "author_date": "timestamp[us]",
    "files_changed": "int32",
    "insertions": "int32",
    "deletions": "int32",
    "subject": "string",
    "body": "string",
}

BATCH_SIZE = 1000
MAX_WORKERS = 8
MIN_BATCH_SIZE = 100
MIN_WORKERS = 1
COMMIT_PATTERNS_DIR = "commit_patterns"
