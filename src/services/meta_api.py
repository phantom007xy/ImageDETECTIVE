"""Meta Model API for the app: one HTTP client, a daily spending cap and a result cache.

Every paid call goes through `paid_call`: it refuses when today's spending would pass the cap,
returns a cached result for the same request (so the same image is never paid for twice), and
reserves the estimated maximum before starting, then refunds unused budget on success.
Failed stages keep their reservation because a remote call may already be billed.
If anything fails, the caller gets None and the app shows everything
else: the verdict never waits on, or depends on, an API call (plan Section 0).

Settings (environment variables): MODEL_API_KEY (or a line in .env), API_DAILY_CAP_USD (default 3).
Spending is kept per UTC day in .app_state/api_usage.json (ignored by git).
"""

import json
import math
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
MAX_CACHE_BYTES = 64 * 1024 * 1024
MAX_CACHE_ITEMS = 128


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
    temporary = STATE.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(usage, indent=1))
    temporary.replace(STATE)


def client():
    global _client
    if _client is None:
        _client = httpx.Client(base_url=BASE_URL, timeout=httpx.Timeout(60, connect=10, pool=5))
    return _client


def _size(value):
    if isinstance(value, (bytes, str)):
        return len(value)
    if isinstance(value, dict):
        return sum(_size(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_size(v) for v in value)
    return int(getattr(value, "nbytes", 64))


def paid_call(cache_key, max_cost, call):
    """Serialize paid work separately from inference and reserve its upper-bound cost first.

    Successful calls refund unused reservation. Failed calls keep it: a remote stage may
    already have been billed. This is a conservative daily budget, not a billing statement.
    The lock also deduplicates simultaneous identical requests in this single-process app.
    """
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    # Optional work must not occupy all of FastAPI's worker threads while another
    # API request is slow. Return the normal unavailable fallback immediately.
    if not _lock.acquire(blocking=False):
        return None
    try:
        if cache_key in _cache:
            value = _cache.pop(cache_key)
            _cache[cache_key] = value
            return value
        if not math.isfinite(max_cost) or max_cost < 0 or spent_today() + max_cost > daily_cap():
            return None
        key = api_key()
        if key is None:
            return None
        day = datetime.now(timezone.utc).date()
        try:
            _record(max_cost)  # persist before starting any potentially billed stage
        except OSError:
            return None
        try:
            result, cost = call(key)
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("Invalid API cost")
            if datetime.now(timezone.utc).date() == day:
                if cost != max_cost:
                    _record(cost - max_cost)
            else:
                _record(cost)  # conservative accounting across a UTC-day boundary
        except Exception as error:
            print(f"[meta_api] {type(error).__name__}: optional service unavailable")
            return None
        if result is not None:
            if _size(result) <= MAX_CACHE_BYTES:
                _cache[cache_key] = result
                while len(_cache) > MAX_CACHE_ITEMS or sum(_size(v) for v in _cache.values()) > MAX_CACHE_BYTES:
                    _cache.pop(next(iter(_cache)))
        return result
    finally:
        _lock.release()


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
