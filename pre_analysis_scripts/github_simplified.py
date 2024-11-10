import asyncio
import json
import os
import sys
import time
import aiohttp
import logging
from urllib.parse import urljoin

from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

load_dotenv(override=True)


class GitHubAPIClient:
    BASE_URL = "https://api.github.com"
    MIN_INTERVAL = 0.05  # 50ms
    INITIAL_INTERVAL = 0.1  # 100ms
    MAX_RESET_TIMES = 3  # Number of recent reset times to keep

    def __init__(self, token=os.getenv("GH_TOKEN")):
        self.token = token
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self.interval = self.INITIAL_INTERVAL
        self.session = None
        self.stop_flag = False
        self.pause_flag = asyncio.Event()
        self.pause_flag.set()  # Initially not paused
        self.limit: int = 0
        self.remaining: int = 0
        self.reset_time: int = 0
        self.last_reset_time: int = 0
        self.reset_times = []
        self.rate_limit_period: float = 3600

    async def get_data(self, endpoint, params=None):

        self.stop_flag = False
        self.pause_flag.set()
        logger.info(f"Starting data retrieval for endpoint: {endpoint}")
        async with aiohttp.ClientSession(headers=self.headers) as self.session:
            async for response in self._process_requests(endpoint, params):
                yield response

    async def _process_requests(self, endpoint, params):
        url = urljoin(self.BASE_URL, endpoint)
        params = params or {}
        if "per_page" not in params:
            params["per_page"] = 100

        logger.info(f"Processing requests starting with URL: {url}")
        while url and not self.stop_flag:
            await self.pause_flag.wait()  # Wait if paused
            try:
                logger.debug(f"Making request to URL: {url}")
                response_data, headers = await self._make_request(url, params)

                if isinstance(response_data, list):
                    logger.info(
                        f"Received list response with {len(response_data)} items"
                    )
                elif isinstance(response_data, dict):
                    item_count = len(response_data.get("items", []))
                    logger.info(f"Received dict response with {item_count} items")
                elif isinstance(response_data, str):
                    logger.info(
                        f"Received string response with {len(response_data)} characters"
                    )
                else:
                    logger.warning(f"Unexpected response type: {type(response_data)}")

                yield response_data

                url = self._get_next_url(headers)
                logger.debug(f"Next URL: {url}")
                # Clear params for subsequent requests, as they're included in the next URL
                params = {}

            except aiohttp.ClientError as e:
                logger.error(f"Request failed: {e}")
                break  # Stop processing on error

    async def _make_request(self, url, params):
        if self.session is None:
            raise RuntimeError(
                "Session is not initialized. Use 'get_data' method to start a session."
            )

        while True:
            try:
                async with self.session.get(url, params=params) as response:
                    logger.debug(f"Request made to {url}. Status: {response.status}")
                    await self._update_rate_limit(response.headers)

                    if response.status == 403:
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            wait_time = int(retry_after)
                            logger.warning(
                                f"Rate limited. Retrying after {wait_time} seconds."
                            )
                            await self._display_countdown(wait_time)
                            continue
                        elif (
                            int(response.headers.get("X-RateLimit-Remaining", "0")) == 0
                        ):
                            reset_time = int(
                                response.headers.get("X-RateLimit-Reset", "0")
                            )
                            wait_time = max(reset_time - time.time(), 0)
                            logger.warning(
                                f"Rate limit exceeded. Waiting for {wait_time:.2f} seconds."
                            )
                            await self._display_countdown(wait_time)
                            continue
                        else:
                            # 403 for reasons other than rate limiting
                            logger.error(
                                "Received 403 status code. This may be due to authentication issues or lack of permissions."
                            )
                            raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=response.status,
                                message="403 Forbidden: Not a rate limiting issue",
                                headers=response.headers,
                            )

                    response.raise_for_status()

                    try:
                        data = await response.json()
                        logger.debug("Successfully parsed response as JSON")
                    except (aiohttp.ContentTypeError, json.JSONDecodeError):
                        logger.debug(
                            "Failed to parse JSON, falling back to text content"
                        )
                        data = await response.text()

                    return data, response.headers

            except aiohttp.ClientResponseError as e:
                if e.status not in [403, 404, 422, 451]:
                    logger.warning(f"Request failed: {e}. Retrying in 60 seconds.")
                    await self._display_countdown(60)
                else:
                    logger.error(
                        f"Request failed with status {e.status}: {e}. Not retrying."
                    )
                    raise

            finally:
                await self._adaptive_sleep()

    def _get_next_url(self, headers):
        links = headers.get("Link", "")
        for link in links.split(","):
            try:
                url, rel = link.split(";")
                url = url.strip()[1:-1]  # Remove < and >
                if 'rel="next"' in rel:
                    return url
            except ValueError:
                continue
        return None

    async def _update_rate_limit(self, headers):
        current_time = int(time.time())
        new_reset_time = int(headers.get("X-RateLimit-Reset", "0"))

        # Ensure we never have zero values for core rate limiting parameters
        self.limit = max(int(headers.get("X-RateLimit-Limit", "60")), 1)
        self.remaining = int(headers.get("X-RateLimit-Remaining", "0"))

        # Update reset times list for period calculation
        if not self.reset_times or new_reset_time != self.reset_times[-1]:
            self.reset_times.append(new_reset_time)
            if len(self.reset_times) > self.MAX_RESET_TIMES:
                self.reset_times.pop(0)

        # Calculate rate limit period with safety checks
        if (
            len(self.reset_times) >= 2
        ):  # Need at least 2 reset times to calculate a period
            try:
                # Calculate time differences between consecutive reset times
                periods = [
                    t2 - t1
                    for t1, t2 in zip(self.reset_times[:-1], self.reset_times[1:])
                    if t2 > t1  # Ensure only positive periods
                ]

                # Only update rate_limit_period if we have valid periods
                if periods:
                    self.rate_limit_period = max(sum(periods) / len(periods), 1.0)
                logger.debug(
                    f"Calculated periods: {periods}, Using rate_limit_period: {self.rate_limit_period}"
                )
            except Exception as e:
                logger.warning(
                    f"Error calculating rate limit period: {e}. Using default period."
                )

        # Ensure rate_limit_period is never 0 or too small
        self.rate_limit_period = max(self.rate_limit_period, 3600)  # Minimum 1 hour

        time_to_reset = max(
            new_reset_time - current_time, 1
        )  # Ensure at least 1 second
        rate_limit_ratio = self.remaining / self.limit

        # Calculate target requests per second with improved edge case handling
        if self.remaining > 0:
            target_rps = self.remaining / time_to_reset
        elif time_to_reset <= 1:
            target_rps = self.limit / self.rate_limit_period
        else:
            target_rps = max(self.limit / self.rate_limit_period, 0.016)

        # Ensure target_rps is never zero or too small
        target_rps = max(target_rps, 0.016)  # Never go slower than 1 request per minute

        # Calculate resource per time ratio with safety check
        resource_time_ratio = (
            rate_limit_ratio / (time_to_reset / self.rate_limit_period)
            if time_to_reset > 0
            else 1
        )

        # Adjust interval based on target RPS and resource_time_ratio
        if resource_time_ratio > 2:  # We have more than twice the expected resources
            self.interval = max(
                self.MIN_INTERVAL, (1 / target_rps) * 0.5
            )  # Be very aggressive
        elif resource_time_ratio > 1:  # We have more resources than expected
            self.interval = max(
                self.MIN_INTERVAL, (1 / target_rps) * 0.8
            )  # Be aggressive
        else:
            self.interval = max(
                self.MIN_INTERVAL, (1 / target_rps) * 1.1
            )  # Normal 10% safety margin

        # Further adjust based on how close we are to reset
        time_ratio = time_to_reset / self.rate_limit_period
        if time_ratio < 0.1:  # Less than 10% of period left
            self.interval = max(
                self.MIN_INTERVAL, self.interval * 0.8
            )  # Be more aggressive
        elif (
            time_ratio > 0.9 and resource_time_ratio < 1
        ):  # More than 90% of period left and low on resources
            self.interval = min(
                self.interval * 1.2, 1.0
            )  # Be more conservative, but cap at 1 second

        # Cap the final interval
        self.interval = min(max(self.interval, self.MIN_INTERVAL), 60.0)

        logger.info(
            f"Rate limit: {self.remaining}/{self.limit}, "
            f"Time to reset: {time_to_reset:.2f}s, "
            f"Rate limit period: {self.rate_limit_period:.2f}s, "
            f"Resource/Time ratio: {resource_time_ratio:.2f}, "
            f"Target RPS: {target_rps:.2f}, "
            f"New interval: {self.interval:.2f}s"
        )

    async def _adaptive_sleep(self):
        logger.debug(f"Sleeping for {self.interval:.2f} seconds")
        await asyncio.sleep(self.interval)

    async def _display_countdown(self, wait_time):
        start_time = time.time()
        end_time = start_time + wait_time

        print()

        while time.time() < end_time:
            remaining = max(0, int(end_time - time.time()))
            hours, remainder = divmod(remaining, 3600)
            minutes, seconds = divmod(remainder, 60)

            timer = f"\rResuming in: {hours:02d}:{minutes:02d}:{seconds:02d}"
            sys.stdout.write(timer)
            sys.stdout.flush()

            await asyncio.sleep(0.1)
        sys.stdout.write("\rResuming requests...                      \n")
        sys.stdout.flush()
        logger.info("Rate limit reset. Resuming requests.")

    def stop(self):
        self.stop_flag = True

    def pause(self):
        self.pause_flag.clear()

    def resume(self):
        self.pause_flag.set()
