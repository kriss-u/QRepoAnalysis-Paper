import os
import subprocess
import jsonlines
import logging
import shutil
import time

INPUT_FILE = "comparison_report_with_sha_final.jsonl"
OUTPUT_DIR = "../quantum-repos"


def setup_logging():
    log_file = os.path.join(os.getcwd(), "clone_errors.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def encode_branch_name(branch):
    return branch.replace("/", "___")


def check_git_installed():
    return shutil.which("git") is not None


def clone_repo(repo_data):
    owner = repo_data["owner"]
    repo = repo_data["repo"]
    branch = repo_data["branch"]
    sha = repo_data.get("sha")

    safe_branch_name = encode_branch_name(branch)
    safe_dir_name = f"{owner}+{repo}+{safe_branch_name}"
    full_path = os.path.join(OUTPUT_DIR, safe_dir_name)

    if os.path.exists(full_path):
        print(f"Directory {full_path} already exists. Skipping clone.")
        return 0  # Success (already exists)

    print(f"Cloning {owner}/{repo} ({branch}) into {full_path}...")

    git_url = f"git@github.com:{owner}/{repo}.git"
    try:
        clone_cmd = [
            "git",
            "clone",
            "--single-branch",
            "--branch",
            branch,
            "--depth",
            "1",
            git_url,
            full_path,
        ]
        subprocess.run(clone_cmd, check=True, capture_output=True, text=True)

        if sha:
            # Checkout the specific commit
            subprocess.run(
                ["git", "checkout", sha],
                cwd=full_path,
                check=True,
                capture_output=True,
                text=True,
            )

        print(f"Successfully cloned {owner}/{repo} ({branch})")
        return 0  # Success
    except subprocess.CalledProcessError as e:
        error_message = f"Failed to clone {owner}/{repo} ({branch}): {e.stderr}"
        print(error_message)
        logging.error(error_message)
        return 1  # Failure


def process_repo_data(repo_data):
    original_result = clone_repo(
        {
            "owner": repo_data["original_owner"],
            "repo": repo_data["original_repo"],
            "branch": repo_data["base"],
            "sha": repo_data.get("base_sha"),
        }
    )

    fork_result = clone_repo(
        {
            "owner": repo_data["fork_owner"],
            "repo": repo_data["fork_repo"],
            "branch": repo_data["head"],
            "sha": repo_data.get("head_sha"),
        }
    )

    return original_result + fork_result  # Total failures for this repo pair


def main():
    if not check_git_installed():
        print(
            "Error: 'git' command not found. Please install Git and make sure it's in your PATH."
        )
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging()

    failed_clones = 0

    with jsonlines.open(INPUT_FILE) as reader:
        for repo_data in reader:
            failed_clones += process_repo_data(repo_data)
            time.sleep(1)

    print(f"Cloning complete. Failed clones: {failed_clones}")


if __name__ == "__main__":
    main()
