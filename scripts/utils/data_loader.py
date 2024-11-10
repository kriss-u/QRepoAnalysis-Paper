import logging

logger = logging.getLogger(__name__)


def load_language_data(file_path):
    import pandas as pd

    try:
        df = pd.read_csv(
            file_path,
            dtype={
                "repo": str,
                "language": str,
                "frameworks": str,
                "files": str,
                "code": str,
                "comments": str,
                "blanks": str,
            },
            keep_default_na=False,
        )

        stats = {}
        for _, row in df.iterrows():
            try:
                files = (
                    int(row.get("files", "0")) if row.get("files", "").isdigit() else 0
                )
                code = int(row.get("code", "0")) if row.get("code", "").isdigit() else 0
                comments = (
                    int(row.get("comments", "0"))
                    if row.get("comments", "").isdigit()
                    else 0
                )
                blanks = (
                    int(row.get("blanks", "0"))
                    if row.get("blanks", "").isdigit()
                    else 0
                )

                stats[row["repo"]] = {
                    "language": row.get("language"),
                    "frameworks": row.get("frameworks", ""),
                    "files": files,
                    "code": code,
                    "comments": comments,
                    "blanks": blanks,
                }
            except ValueError as e:
                logger.warning(
                    f"Skipping invalid numeric values for repo {row.get('repo')}: {str(e)}"
                )
                continue

        logger.info(f"Successfully loaded data for {len(stats)} repositories")
        return stats

    except Exception as e:
        logger.error(f"Error loading language data: {str(e)}")
        raise


def parse_frameworks(frameworks_str):
    if not frameworks_str or frameworks_str == "0":
        return []

    frameworks = [fw.strip() for fw in frameworks_str.split(";") if fw.strip()]
    return frameworks
