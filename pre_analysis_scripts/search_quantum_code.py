import asyncio
import logging
import os
import aiofiles
from dotenv import load_dotenv
from github_simplified import GitHubAPIClient

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def write_repo_to_file(repo_name, file_name="repos_new.txt"):
    async with aiofiles.open(file_name, "a") as f:
        await f.write(f"{repo_name}\n")


async def main():
    load_dotenv(override=True)
    client = GitHubAPIClient(os.getenv("GH_TOKEN"))
    endpoint = "/search/code"
    params = {
        "q": "quantum "
        # + "filename:Pipfile "
        # + "filename:Pyproject.toml "
        # + "filename:setup.py "
        + "filename:requirements.txt "
    }
    try:
        async for response in client.get_data(endpoint, params):
            if isinstance(response, dict):
                for item in response["items"]:
                    repo = item["repository"]
                    if not repo["fork"]:
                        repo_name = repo["full_name"]
                        await write_repo_to_file(repo_name)
            else:
                logger.error("Unexpected response type")

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
