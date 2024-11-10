import os
import json
import asyncio
import logging
import shutil
import aiohttp
from github_simplified import GitHubAPIClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class ContributorRepoMapper:
    def __init__(self):
        self.repos_client = GitHubAPIClient(os.getenv("GH_TOKEN"))
        self.readme_client = GitHubAPIClient(os.getenv("GH_TOKEN_2"))
        self.root_dir = os.getcwd()
        self.profiles_dir = os.path.join(self.root_dir, "developer_profiles_final")
        self.source_profiles_dir = os.path.join(
            self.root_dir, "developer_profiles_combined"
        )
        self.output_dir = os.path.join(self.root_dir, "developer_profiles_final_2")
        self.processed_developers = set()

        os.makedirs(self.output_dir, exist_ok=True)

    def get_developer_list(self):
        developers = []
        if not os.path.exists(self.profiles_dir):
            logger.error(f"Profiles directory {self.profiles_dir} does not exist")
            return developers

        for filename in os.listdir(self.profiles_dir):
            if filename.endswith(".json"):
                developer = filename.replace(".json", "")
                developers.append(developer)

        logger.info(f"Found {len(developers)} developers in profiles directory")
        return developers

    def copy_existing_developer_data(self):
        existing_developers = set()

        if not os.path.exists(self.source_profiles_dir):
            logger.warning(
                f"Source directory {self.source_profiles_dir} does not exist"
            )
            return existing_developers

        developers = [
            d
            for d in os.listdir(self.source_profiles_dir)
            if os.path.isdir(os.path.join(self.source_profiles_dir, d))
        ]

        total = len(developers)
        logger.info(f"Starting to copy existing data for {total} developers")

        for i, developer in enumerate(developers, 1):
            source_dir = os.path.join(self.source_profiles_dir, developer)
            target_dir = os.path.join(self.output_dir, developer)

            if os.path.isdir(source_dir):
                try:
                    if not os.path.exists(target_dir):
                        shutil.copytree(source_dir, target_dir)
                    existing_developers.add(developer)
                    logger.info(
                        f"Copied existing data for developer: {developer} ({i}/{total})"
                    )
                except Exception as e:
                    logger.error(f"Error copying data for {developer}: {e}")

        return existing_developers

    def copy_profile_json(self, developer: str) -> bool:
        source_file = os.path.join(self.profiles_dir, f"{developer}.json")
        if not os.path.exists(source_file):
            logger.error(f"Profile file not found for developer: {developer}")
            return False

        developer_dir = os.path.join(self.output_dir, developer)
        os.makedirs(developer_dir, exist_ok=True)
        target_file = os.path.join(developer_dir, "profile.json")

        try:
            shutil.copy2(source_file, target_file)
            logger.info(f"Copied profile for developer: {developer}")
            return True
        except Exception as e:
            logger.error(f"Error copying profile for {developer}: {e}")
            return False

    def extract_repo_info(self, repo_data):
        owner = repo_data.get("owner", {})
        return {
            "name": repo_data.get("name"),
            "full_name": repo_data.get("full_name"),
            "description": repo_data.get("description"),
            "language": repo_data.get("language"),
            "stargazers_count": repo_data.get("stargazers_count"),
            "watchers_count": repo_data.get("watchers_count"),
            "archived": repo_data.get("archived"),
            "forks_count": repo_data.get("forks_count"),
            "created_at": repo_data.get("created_at"),
            "updated_at": repo_data.get("updated_at"),
            "pushed_at": repo_data.get("pushed_at"),
            "topics": repo_data.get("topics", []),
            "size": repo_data.get("size"),
            "open_issues_count": repo_data.get("open_issues_count"),
            "has_issues": repo_data.get("has_issues"),
            "has_projects": repo_data.get("has_projects"),
            "has_wiki": repo_data.get("has_wiki"),
            "has_discussions": repo_data.get("has_discussions"),
            "license": repo_data.get("license"),
            "has_pages": repo_data.get("has_pages"),
            "visibility": repo_data.get("visibility"),
            "is_template": repo_data.get("is_template"),
            "default_branch": repo_data.get("default_branch"),
            "owner.login": owner.get("login"),
            "owner.id": owner.get("id"),
        }

    async def get_repo_readme(self, owner, repo):
        endpoint = f"/repos/{owner}/{repo}/readme"
        try:
            self.readme_client.headers["Accept"] = "application/vnd.github.raw+json"
            async for response in self.readme_client.get_data(endpoint):
                if isinstance(response, str):
                    return response
            return None
        except aiohttp.ClientResponseError as e:
            if e.status == 404:
                logger.debug(f"No README found for {owner}/{repo}")
                return None
            logger.error(f"Error fetching README for {owner}/{repo}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching README for {owner}/{repo}: {e}")
            return None
        finally:
            self.readme_client.headers["Accept"] = "application/vnd.github+json"

    async def process_developer_repos(self, developer: str):
        if developer in self.processed_developers:
            logger.info(f"Developer {developer} already processed")
            return

        logger.info(f"Starting to process repositories for developer: {developer}")

        if not self.copy_profile_json(developer):
            return

        developer_dir = os.path.join(self.output_dir, developer)
        repos_dir = os.path.join(developer_dir, "repos")
        os.makedirs(repos_dir, exist_ok=True)

        endpoint = f"/users/{developer}/repos"
        page = 1
        total_repos = 0

        try:
            async for response in self.repos_client.get_data(endpoint):
                if isinstance(response, list):
                    logger.info(
                        f"Processing page {page} with {len(response)} repositories for {developer}"
                    )
                    for repo_data in response:
                        try:
                            repo_name = repo_data["name"]
                            repo_dir = os.path.join(repos_dir, repo_name)
                            repo_json_path = os.path.join(repo_dir, "repo.json")
                            readme_path = os.path.join(repo_dir, "README.md")

                            if not os.path.exists(repo_dir):
                                os.makedirs(repo_dir)

                            if not os.path.exists(repo_json_path):
                                repo_info = self.extract_repo_info(repo_data)
                                with open(repo_json_path, "w") as f:
                                    json.dump(repo_info, f, indent=2)
                                logger.info(
                                    f"Saved repo info for {developer}/{repo_name}"
                                )

                            if not os.path.exists(readme_path):
                                readme = await self.get_repo_readme(
                                    developer, repo_name
                                )
                                if readme:
                                    with open(readme_path, "w", encoding="utf-8") as f:
                                        f.write(readme)
                                    logger.info(
                                        f"Saved README for {developer}/{repo_name}"
                                    )

                            total_repos += 1
                        except Exception as e:
                            logger.error(f"Error processing repo {repo_name}: {e}")
                            continue

                    page += 1
                    logger.info(
                        f"Completed page {page-1} for {developer}, processed {total_repos} repos so far"
                    )

        except Exception as e:
            logger.error(f"Error processing repos for {developer}: {e}")
            return

        self.processed_developers.add(developer)
        logger.info(
            f"Completed processing {total_repos} repositories for developer: {developer}"
        )

    async def run(self):
        developer_list = self.get_developer_list()

        existing_developers = self.copy_existing_developer_data()
        logger.info(f"Copied data for {len(existing_developers)} existing developers")

        remaining_developers = [
            dev for dev in developer_list if dev not in existing_developers
        ]

        total = len(remaining_developers)
        logger.info(f"Processing {total} remaining developers")

        for i, developer in enumerate(remaining_developers, 1):
            logger.info(f"Processing developer {i}/{total}: {developer}")
            await self.process_developer_repos(developer)

        logger.info("Completed processing all developers")


async def main():
    mapper = ContributorRepoMapper()
    await mapper.run()


if __name__ == "__main__":
    asyncio.run(main())
