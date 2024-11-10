import time
import requests
from datetime import datetime
import re
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GithubAPI:
    BASE_URL = "https://api.github.com"

    def __init__(self, token):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )

        self.limit = None
        self.remaining = None
        self.reset_time = None

        self.logger = logging.getLogger(f"{__name__}.GithubAPI")

        self.branches = BranchesResource(self)
        self.users = UsersResource(self)
        self.repos = ReposResource(self)
        self.commits = CommitsResource(self)
        self.issues = IssuesResource(self)

    def _make_request(self, method, endpoint, **kwargs):
        while True:
            try:
                response = self._attempt_request(method, endpoint, **kwargs)

                if response.status_code == 404:
                    self.logger.warning(f"Resource not found: {endpoint}")
                    return None

                if response.status_code == 422:
                    self.logger.warning(f"Unprocessable Entity for request: {endpoint}")
                    self.logger.warning(f"Response content: {response.text}")
                    return None

                if response.status_code == 403 and "Retry-After" in response.headers:
                    retry_after = int(response.headers["Retry-After"])
                    self.logger.info(
                        f"Rate limit exceeded. Retrying after {retry_after} seconds."
                    )
                    self._display_countdown(retry_after)
                    continue

                if (
                    response.status_code not in [403, 429]
                    or "x-ratelimit-remaining" not in response.headers
                ):
                    response.raise_for_status()
                    return response

                wait_time = self._get_wait_time(response.headers)
                if wait_time > 0:
                    self.logger.info(
                        f"Rate limit exceeded. Waiting for {wait_time} seconds before retrying."
                    )
                    self._display_countdown(wait_time)
                else:
                    return response
            except requests.exceptions.RequestException as e:
                if isinstance(
                    e, requests.exceptions.HTTPError
                ) and e.response.status_code not in [404, 422]:
                    self.logger.error(f"Request failed: {e}. Retrying in 60 seconds.")
                    time.sleep(60)
                else:
                    self.logger.exception("Unexpected error occurred")
                    raise

    def _attempt_request(self, method, endpoint, **kwargs):
        url = f"{self.BASE_URL}{endpoint}"
        self.logger.info(f"Making {method} request to: {url}")
        response = self.session.request(method, url, **kwargs)
        self._update_rate_limit(response.headers)
        time.sleep(0.51)  # Just an empirical value
        return response

    def _get_wait_time(self, headers):
        reset_time = int(headers.get("x-ratelimit-reset", 0))
        current_time = time.time()
        return max(reset_time - current_time, 0)

    def _update_rate_limit(self, headers):
        self.limit = int(headers.get("x-ratelimit-limit", 0))
        self.remaining = int(headers.get("x-ratelimit-remaining", 0))
        reset_time = headers.get("x-ratelimit-reset")
        self.reset_time = int(reset_time) if reset_time else None
        self.logger.info(f"Rate limit: {self.remaining}/{self.limit}")
        if self.reset_time:
            self.logger.info(f"Reset time: {datetime.fromtimestamp(self.reset_time)}")
        else:
            self.logger.info("Reset time: Unknown")

    def _display_countdown(self, wait_time):
        for remaining in range(int(wait_time), 0, -1):
            hours, remainder = divmod(remaining, 3600)
            minutes, seconds = divmod(remainder, 60)
            timer = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            self.logger.info(f"Time remaining: {timer}")
            time.sleep(1)
        self.logger.info("Rate limit reset. Resuming requests.")

    def _get_paginated_data(self, endpoint, **params):
        next_pattern = re.compile(r'<([^>]*)>;\s*rel="next"')
        params["per_page"] = 100
        url = endpoint
        all_data = []

        while url:
            response = self._make_request("GET", url, params=params)
            if not response or response.status_code == 204:  # No Content
                return []
            data = response.json()
            all_data.extend(self._parse_data(data))

            link_header = response.headers.get("Link", "")
            next_match = next_pattern.search(link_header)
            url = next_match.group(1) if next_match else None

            # If there's a next URL, remove the base URL to get just the endpoint
            if url and url.startswith(self.BASE_URL):
                url = url[len(self.BASE_URL) :]

            # Clear params for subsequent requests as they're included in the next URL
            params = {}

        return all_data

    def _parse_data(self, data):
        if isinstance(data, list):
            return data
        if not data:
            return []

        # Remove keys that don't include the array of items
        data.pop("incomplete_results", None)
        data.pop("repository_selection", None)
        data.pop("total_count", None)

        # Return the first (and should be only) remaining item, which is the list we want
        return list(data.values())[0]


class Resource:
    def __init__(self, api: "GithubAPI"):
        self.api = api
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")


class BranchesResource(Resource):
    def get(self, owner: str, repo: str, branch: str):
        endpoint = f"/repos/{owner}/{repo}/branches/{branch}"
        self.logger.info(f"Getting branch: {owner}/{repo}/{branch}")
        res = self.api._make_request("GET", endpoint)
        return res.json() if res else None

    def list(self, owner: str, repo: str, **params):
        endpoint = f"/repos/{owner}/{repo}/branches"
        self.logger.info(f"Listing branches for: {owner}/{repo}")
        return self.api._get_paginated_data(endpoint, **params)

    def rename(self, owner: str, repo: str, branch: str, new_name: str):
        endpoint = f"/repos/{owner}/{repo}/branches/{branch}/rename"
        self.logger.info(f"Renaming branch: {owner}/{repo}/{branch} to {new_name}")
        data = {"new_name": new_name}
        res = self.api._make_request("POST", endpoint, json=data)
        return res.json() if res else None


class UsersResource(Resource):
    def list(self, **params):
        endpoint = "/users"
        self.logger.info("Listing users")
        return self.api._get_paginated_data(endpoint, **params)

    def get(self, username: str):
        endpoint = f"/users/{username}"
        self.logger.info(f"Getting user: {username}")
        res = self.api._make_request("GET", endpoint)
        return res.json() if res else None


class ReposResource(Resource):
    def get(self, owner: str, repo: str):
        endpoint = f"/repos/{owner}/{repo}"
        self.logger.info(f"Getting repo: {owner}/{repo}")
        res = self.api._make_request("GET", endpoint)
        return res.json() if res else None

    def list(self, owner: str, **params):
        endpoint = f"/users/{owner}/repos"
        self.logger.info(f"Listing repos for: {owner}")
        return self.api._get_paginated_data(endpoint, **params)

    def list_languages(self, owner: str, repo: str):
        endpoint = f"/repos/{owner}/{repo}/languages"
        self.logger.info(f"Listing languages for: {owner}/{repo}")
        res = self.api._make_request("GET", endpoint)
        return res.json() if res else None

    def get_readme(self, owner: str, repo: str):
        endpoint = f"/repos/{owner}/{repo}/readme"
        self.logger.info(f"Getting README for: {owner}/{repo}")
        res = self.api._make_request("GET", endpoint)
        return res.json() if res else None

    def list_contributors(self, owner: str, repo: str, **params):
        endpoint = f"/repos/{owner}/{repo}/contributors"
        self.logger.info(f"Listing contributors for: {owner}/{repo}")
        return self.api._get_paginated_data(endpoint, **params)


class CommitsResource(Resource):
    def compare(self, owner: str, repo: str, base: str, head: str):
        endpoint = f"/repos/{owner}/{repo}/compare/{base}...{head}"
        self.logger.info(f"Comparing commits: {owner}/{repo} {base}...{head}")
        res = self.api._make_request("GET", endpoint)
        return res.json() if res else None


class IssuesResource(Resource):
    def list(self, owner: str, repo: str, **params):
        endpoint = f"/repos/{owner}/{repo}/issues"
        self.logger.info(f"Listing issues for: {owner}/{repo}")
        return self.api._get_paginated_data(endpoint, **params)

    def get(self, owner: str, repo: str, issue_number: int):
        endpoint = f"/repos/{owner}/{repo}/issues/{issue_number}"
        self.logger.info(f"Getting issue: {owner}/{repo}#{issue_number}")
        res = self.api._make_request("GET", endpoint)
        return res.json() if res else None

    def list_comments(self, owner: str, repo: str, issue_number: int, **params):
        endpoint = f"/repos/{owner}/{repo}/issues/{issue_number}/comments"
        self.logger.info(f"Listing comments for issue: {owner}/{repo}#{issue_number}")
        return self.api._get_paginated_data(endpoint, **params)


if __name__ == "__main__":
    logging.info("GithubAPI script started")
    logging.info("GithubAPI script completed")
