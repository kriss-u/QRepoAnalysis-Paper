import os
import subprocess
import logging
import shutil
import time

INPUT_FILE = "final-repo-list-2/repos_for_analysis_all.txt"
OUTPUT_DIR = "/media/researchstudent/T7/repos_shallow_clones"
LOG_FILE = "clone_errors_all_shallow.log"
CLONE_TIMEOUT = 300


def setup_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def check_git_installed():
    return shutil.which("git") is not None


def clone_repo(repo_full_name):
    owner, repo = repo_full_name.split("/")
    dir_name = f"{owner}+{repo}"
    full_path = os.path.join(OUTPUT_DIR, dir_name)

    if os.path.exists(full_path):
        print(f"Directory {full_path} already exists. Skipping clone.")
        return 0  # Success (already exists)

    print(f"Cloning {repo_full_name} into {full_path}...")
    # Use HTTPS instead of SSH (SSH hangs after 20 to 30 repos processing)
    git_url = f"https://github.com/{owner}/{repo}.git"
    # git_url = f"git@github.com:{owner}/{repo}.git"
    try:
        clone_cmd = [
            "git",
            "clone",
            "--single-branch",
            "--depth",
            "1",
            git_url,
            full_path,
        ]

        process = subprocess.run(
            clone_cmd,
            text=True,
            timeout=CLONE_TIMEOUT,
            env=dict(os.environ, GIT_TERMINAL_PROMPT="0"),
        )

        if process.returncode == 0:
            print(f"Successfully cloned {repo_full_name}")
            return 0
        else:
            error_message = f"Failed to clone {repo_full_name} with return code {process.returncode}"
            print(error_message)
            logging.error(error_message)
            return 1

    except subprocess.TimeoutExpired:
        error_message = (
            f"Timeout while cloning {repo_full_name} after {CLONE_TIMEOUT} seconds"
        )
        print(error_message)
        logging.error(error_message)
        if os.path.exists(full_path):
            shutil.rmtree(full_path, ignore_errors=True)
        return 1

    except subprocess.CalledProcessError as e:
        error_message = f"Failed to clone {repo_full_name}"
        print(error_message)
        if hasattr(e, "stderr") and e.stderr:
            logging.error(f"{error_message}: {e.stderr}")
        else:
            logging.error(error_message)
        if os.path.exists(full_path):
            shutil.rmtree(full_path, ignore_errors=True)
        return 1

    except Exception as e:
        error_message = f"Unexpected error while cloning {repo_full_name}: {str(e)}"
        print(error_message)
        logging.error(error_message)
        if os.path.exists(full_path):
            shutil.rmtree(full_path, ignore_errors=True)
        return 1


def main():
    if not check_git_installed():
        print(
            "Error: 'git' command not found. Please install Git and make sure it's in your PATH."
        )
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    setup_logging()

    failed_clones = 0
    total_repos = 0

    with open(INPUT_FILE, "r") as f:
        repo_list = f.read().splitlines()

    total_repos = len(repo_list)
    for i, repo_full_name in enumerate(repo_list, 1):
        print(f"\nProcessing {i}/{total_repos}: {repo_full_name}")
        failed_clones += clone_repo(repo_full_name.strip())
        time.sleep(1)

    print(f"\nCloning complete. Failed clones: {failed_clones}/{total_repos}")


if __name__ == "__main__":
    main()
