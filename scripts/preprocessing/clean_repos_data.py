import json
from tqdm import tqdm

from ..utils.logger import setup_logger
from ..config.constants import REPOS_DATA_DIR, RAW_DATA_DIR

logger = setup_logger(__name__, "preprocess", "clean-repos")


def find_first_json_object(content):
    bracket_count = 0
    start_index = content.find("{")

    if start_index == -1:
        return None

    for i in range(start_index, len(content)):
        if content[i] == "{":
            bracket_count += 1
        elif content[i] == "}":
            bracket_count -= 1

        if bracket_count == 0:
            return content[start_index : i + 1]

    return None


def validate_json(data):
    required_fields = [
        "id",
        "name",
        "full_name",
        "owner",
        "created_at",
        "updated_at",
        "stargazers_count",
        "language",
    ]

    return all(field in data for field in required_fields)


def clean_repo_files(args):
    logger.info("Starting repository data cleaning")

    input_dir = (
        args.input_dir
        if hasattr(args, "input_dir") and args.input_dir
        else REPOS_DATA_DIR
    )
    output_dir = (
        args.output_dir
        if hasattr(args, "output_dir") and args.output_dir
        else RAW_DATA_DIR / "repos_final"
    )

    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_files": 0,
        "cleaned_files": 0,
        "invalid_files": 0,
        "already_clean": 0,
    }

    try:
        json_files = list(input_dir.glob("*.json"))
        stats["total_files"] = len(json_files)

        logger.info(f"Found {len(json_files)} files to process")

        for file_path in tqdm(json_files, desc="Cleaning repository data"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                try:
                    data = json.loads(content)
                    if validate_json(data):
                        stats["already_clean"] += 1
                        output_path = output_dir / file_path.name
                        with open(output_path, "w", encoding="utf-8") as f:
                            json.dump(data, f, indent=2)
                        continue
                except json.JSONDecodeError:
                    pass

                first_json = find_first_json_object(content)
                if first_json:
                    try:
                        data = json.loads(first_json)
                        if validate_json(data):
                            stats["cleaned_files"] += 1
                            output_path = output_dir / file_path.name
                            with open(output_path, "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2)
                        else:
                            stats["invalid_files"] += 1
                            logger.warning(f"Invalid data structure in {file_path}")
                    except json.JSONDecodeError:
                        stats["invalid_files"] += 1
                        logger.warning(f"Invalid JSON after cleaning {file_path}")
                else:
                    stats["invalid_files"] += 1
                    logger.warning(f"Could not extract valid JSON from {file_path}")

            except Exception as e:
                stats["invalid_files"] += 1
                logger.error(f"Error processing {file_path}: {str(e)}")

        logger.info("Cleaning completed:")
        logger.info(f"Total files processed: {stats['total_files']}")
        logger.info(f"Files already clean: {stats['already_clean']}")
        logger.info(f"Files cleaned: {stats['cleaned_files']}")
        logger.info(f"Invalid files: {stats['invalid_files']}")

        stats_path = output_dir / "cleaning_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    except Exception as e:
        logger.error(f"Error during cleaning process: {str(e)}", exc_info=True)
        raise


if __name__ == "__main__":
    from argparse import Namespace

    clean_repo_files(Namespace())
