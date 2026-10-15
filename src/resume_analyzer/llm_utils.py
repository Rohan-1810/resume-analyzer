
import time
from groq import RateLimitError


def retry_on_rate_limit(
    operation,
    max_retries: int = 4,
    wait_seconds: int = 5,
):
    for attempt in range(max_retries + 1):
        try:
            return operation()

        except RateLimitError:
            if attempt == max_retries:
                raise

            delay = wait_seconds * (attempt + 1)

            print(
                f"Rate limit reached. "
                f"Retrying in {delay} seconds..."
            )

            time.sleep(delay)
