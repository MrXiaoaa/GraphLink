from __future__ import annotations

import os
from collections.abc import Iterable


def clear_proxy_for_local_endpoints(extra_no_proxy: Iterable[str] = ()) -> None:
    """Disable inherited proxies and preserve a conservative NO_PROXY list."""
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(key, None)
    base = ["127.0.0.1", "localhost"]
    for item in extra_no_proxy:
        text = str(item).strip()
        if text:
            base.append(text)
    os.environ["NO_PROXY"] = ",".join(dict.fromkeys(base))


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value
