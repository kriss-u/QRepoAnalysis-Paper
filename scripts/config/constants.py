from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPOS_FILE = RAW_DATA_DIR / "repos_list.txt"
REPOS_TO_REMOVE = RAW_DATA_DIR / "repos_removed.txt"
REPO_CONTRIBUTORS_FILE = RAW_DATA_DIR / "repo_contributor_map.json"
REPOS_DATA_DIR = RAW_DATA_DIR / "repos_meta"
REPO_LANGUAGES_FILE = RAW_DATA_DIR / "repo_languages.csv"
REPO_CLASSIFIED_FILE = RAW_DATA_DIR / "repos_classified.csv"
REPOS_COMBINED_FILE = RAW_DATA_DIR / "repos_combined_final.json"
COMMITS_CLASSIFIED_FILE = RAW_DATA_DIR / "commits_classified.csv"
REPOS_MAINTENANCE_INFO_FILE = RAW_DATA_DIR / "high_low_maintenance.csv"

FIGURES_DIR = "figures"
TABLES_DIR = "tables"
STATS_DIR = "stats"

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"
