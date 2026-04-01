import requests
import json
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


api_key = ''

TOTAL_PROMPT_TOKENS = 0
TOTAL_COMPLETION_TOKENS = 0
TOTAL_TOKENS = 0
_token_lock = threading.Lock()

MODEL_COSTS = {
    "gemini-2.5-flash": dict(input=3/(10**6), output=15/(10**6)),
}

logger = logging.getLogger(__name__)


def call_model(
    prompt,
    model="gemini-2.5-flash",
    max_retries=5,
    retry_delay=2.0,
    **kwargs,
):
    """Call the LLM API with automatic retry on transient errors.

    Args:
        prompt:       The user prompt string.
        model:        Model name to use.
        max_retries:  Maximum number of retry attempts on failure.
        retry_delay:  Base delay (seconds) between retries; doubles each attempt.
        **kwargs:     Extra parameters forwarded to the API (e.g. max_tokens).

    Returns:
        The full JSON response dict, or None if all retries are exhausted.
    """
    global TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS, TOTAL_TOKENS

    url = ''
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {api_key}',
    }
    data = {
        "stream": False,
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        **kwargs,
    }

    delay = retry_delay
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
            if response.status_code == 200:
                result = response.json()

                # Update token counters thread-safely
                usage = result.get("usage", {})
                with _token_lock:
                    TOTAL_PROMPT_TOKENS += usage.get("prompt_tokens", 0)
                    TOTAL_COMPLETION_TOKENS += usage.get("completion_tokens", 0)
                    TOTAL_TOKENS += usage.get("total_tokens", 0)
                    cost = (
                        TOTAL_PROMPT_TOKENS * MODEL_COSTS[model]["input"]
                        + TOTAL_COMPLETION_TOKENS * MODEL_COSTS[model]["output"]
                    )

                logger.debug(
                    "LLM Cost: prompt_tokens=%d  completion_tokens=%d  "
                    "total_tokens=%d  cost=%.6f",
                    TOTAL_PROMPT_TOKENS, TOTAL_COMPLETION_TOKENS,
                    TOTAL_TOKENS, cost,
                )
                return result

            # Treat rate-limit (429) and server errors (5xx) as retryable
            retryable = response.status_code in (429, 500, 502, 503, 504)
            logger.warning(
                "Attempt %d/%d – HTTP %d%s",
                attempt, max_retries, response.status_code,
                " (retrying)" if retryable else "",
            )
            if not retryable:
                logger.error("Non-retryable error: %s", response.text)
                return None

        except requests.exceptions.RequestException as exc:
            logger.warning("Attempt %d/%d – request exception: %s", attempt, max_retries, exc)

        if attempt < max_retries:
            logger.info("Waiting %.1f s before retry…", delay)
            time.sleep(delay)
            delay = min(delay * 2, 60.0)   # exponential back-off, capped at 60 s

    logger.error("All %d attempts failed for prompt: %.80s…", max_retries, prompt)
    return None


def call_model_parallel(prompts, model="gemini-2.5-flash", max_workers=8, **kwargs):
    """Call the LLM API for multiple prompts concurrently.

    Args:
        prompts:     List of (key, prompt) tuples.  `key` is returned with each result.
        model:       Model name to use.
        max_workers: Maximum number of concurrent threads.
        **kwargs:    Extra parameters forwarded to each API call.

    Returns:
        Dict mapping key -> result (None on failure).
    """
    results = {}

    def _call(key, prompt):
        return key, call_model(prompt, model=model, **kwargs)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_call, key, prompt): key for key, prompt in prompts}
        for future in as_completed(futures):
            key, result = future.result()
            results[key] = result

    return results
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Example usage:
    prompts = [
        ("task1", "What is the capital of France?"),
        ("task2", "Summarize the plot of 'To Kill a Mockingbird'."),
    ]
    results = call_model_parallel(prompts, model="gemini-2.5-flash", max_workers=4)
    print(json.dumps(results, indent=2))