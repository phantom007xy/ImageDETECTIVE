"""Meta Model API for the app: one HTTP client, a daily spending cap and a result cache.

Every paid call goes through `paid_call`: it refuses when today's spending would pass the cap,
returns a cached result for the same request (so the same image is never paid for twice), and
records what was spent. If anything fails, the caller gets None and the app shows everything
else: the verdict never waits on, or depends on, an API call (plan Section 0).

Settings (environment variables): MODEL_API_KEY (or a line in .env), API_DAILY_CAP_USD (default 3).
Spending is kept per UTC day in .app_state/api_usage.json (ignored by git).
"""

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

BASE_URL = "https://api.meta.ai/v1"
STATE = Path(".app_state/api_usage.json")
_lock = threading.Lock()
_cache = {}
_client = None


def api_key():
    if os.environ.get("MODEL_API_KEY"):
        return os.environ["MODEL_API_KEY"]
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "MODEL_API_KEY":
                return value.strip().strip('"').strip("'")
    return None


def daily_cap():
    return float(os.environ.get("API_DAILY_CAP_USD", "3"))


def spent_today():
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        return json.loads(STATE.read_text()).get(today, 0.0)
    except (OSError, ValueError):
        return 0.0


def _record(cost):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        usage = json.loads(STATE.read_text())
    except (OSError, ValueError):
        usage = {}
    usage[today] = usage.get(today, 0.0) + cost
    STATE.parent.mkdir(exist_ok=True)
    STATE.write_text(json.dumps(usage, indent=1))


def client():
    global _client
    if _client is None:
        _client = httpx.Client(base_url=BASE_URL, timeout=120)
    return _client


def paid_call(cache_key, max_cost, call):
    """Run call(key) -> (result, actual_cost) unless cached or over the cap. Returns result or None."""
    with _lock:
        if cache_key in _cache:
            return _cache[cache_key]
        if spent_today() + max_cost > daily_cap():
            return None
    key = api_key()
    if key is None:
        return None
    try:
        result, cost = call(key)
    except Exception as error:  # any API problem: the app simply goes on without this extra
        print(f"[meta_api] {type(error).__name__}: {str(error)[:200]}")
        return None
    with _lock:
        _record(cost)
        if result is not None:
            _cache[cache_key] = result
    return result


def post(key, path, body, retries=3):
    """POST JSON with a short retry on rate limits and server errors; raises on other errors."""
    for attempt in range(retries):
        response = client().post(path, headers={"Authorization": f"Bearer {key}"}, json=body)
        if response.status_code != 429 and response.status_code < 500:
            response.raise_for_status()
            return response.json()
        time.sleep(2 * (attempt + 1))
    response.raise_for_status()
    return response.json()
