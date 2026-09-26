# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp[cli]>=1.4.0",
#     "httpx>=0.27",
# ]
# ///
"""IronBeam futures market-data MCP server for Claude Code.

Exposes real-time CME futures data — Level 1 quotes, Level 2 depth, and
recent trades — from an IronBeam brokerage account, so Claude can pull live
numbers on demand (e.g. the front-month E-mini Nasdaq-100 contract, NQ).

Credentials are read from environment variables:
    IRONBEAM_USERNAME   IronBeam account username
    IRONBEAM_API_KEY    IronBeam API key
    IRONBEAM_PASSWORD   account password (Enterprise accounts only; optional)
    IRONBEAM_ENV        "demo" (default) or "live"

Endpoint paths target the IronBeam xAPI v2. If a call fails, verify the
paths and field names against the official reference at
https://docs.ironbeamapi.com/ — note that IronBeam offers no REST endpoint
for historical OHLC bars (those come over the WebSocket feed instead).
"""

from __future__ import annotations

import os
import time
from urllib.parse import quote

import httpx
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("ironbeam")

_BASE_URLS = {
    "demo": "https://demo.ironbeamapi.com/v2",
    "live": "https://live.ironbeamapi.com/v2",
}

_token: str | None = None


def _base_url() -> str:
    env = os.environ.get("IRONBEAM_ENV", "demo").lower()
    if env not in _BASE_URLS:
        raise ValueError(f"IRONBEAM_ENV must be 'demo' or 'live', got '{env}'.")
    return _BASE_URLS[env]


def _authenticate() -> str:
    username = os.environ.get("IRONBEAM_USERNAME")
    api_key = os.environ.get("IRONBEAM_API_KEY")
    if not username or not api_key:
        raise RuntimeError(
            "Missing credentials. Set IRONBEAM_USERNAME and IRONBEAM_API_KEY "
            "(and IRONBEAM_PASSWORD for Enterprise accounts)."
        )
    body = {
        "username": username,
        "apiKey": api_key,
        "password": os.environ.get("IRONBEAM_PASSWORD", ""),
    }
    resp = httpx.post(f"{_base_url()}/auth", json=body, timeout=20)
    resp.raise_for_status()
    data = resp.json()
    token = data.get("token")
    if not token:
        raise RuntimeError(f"Authentication failed: {data.get('message') or data}")
    return token


def _get(path: str, params: dict | None = None) -> dict:
    """GET an authenticated endpoint, re-authenticating once on a 401."""
    global _token
    if _token is None:
        _token = _authenticate()
    url = f"{_base_url()}{path}"

    def _call() -> httpx.Response:
        return httpx.get(
            url,
            params=params,
            headers={"Authorization": f"Bearer {_token}"},
            timeout=20,
        )

    resp = _call()
    if resp.status_code == 401:
        _token = _authenticate()
        resp = _call()
    try:
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"IronBeam API error {resp.status_code} on {path}: {resp.text[:300]}"
        ) from exc
    return resp.json()


@mcp.tool()
def search_symbols(text: str, limit: int = 10) -> list[dict]:
    """Find IronBeam contract symbols by text.

    Call this first to get the exact symbol string for a futures contract,
    then pass that string to get_quote / get_depth / get_recent_trades.

    Args:
        text: Search text, e.g. "NQ" for E-mini Nasdaq-100 futures.
        limit: Maximum number of matches to return.
    """
    data = _get(
        "/info/symbols",
        {"text": text, "limit": limit, "preferActive": True},
    )
    return [
        {
            "symbol": s.get("symbol"),
            "description": s.get("description"),
            "type": s.get("symbolType"),
        }
        for s in data.get("symbols", [])
    ]


@mcp.tool()
def get_quote(symbol: str) -> dict:
    """Real-time Level 1 quote (last / bid / ask) for a futures contract.

    Args:
        symbol: Exchange-qualified IronBeam symbol, e.g. "XCME:NQ.Z26"
            for the December-2026 E-mini Nasdaq-100 contract. Use
            search_symbols first if you don't know the exact string.
    """
    data = _get("/market/quotes", {"symbols": symbol})
    quotes = data.get("quotes") or data.get("Quotes") or []
    if not quotes:
        raise ValueError(
            f"No quote returned for '{symbol}'. Check the symbol and that "
            "your account has a real-time CME data subscription."
        )
    q = quotes[0]
    return {
        "symbol": q.get("s", symbol),
        "last": q.get("l"),
        "bid": q.get("b"),
        "ask": q.get("a"),
    }


@mcp.tool()
def get_depth(symbol: str, levels: int = 10) -> dict:
    """Level 2 market depth (order book / DOM) for a futures contract.

    Args:
        symbol: Exchange-qualified IronBeam symbol, e.g. "XCME:NQ.Z26".
        levels: Number of price levels per side to return.
    """
    data = _get("/market/depth", {"symbols": symbol})
    depths = data.get("depths") or data.get("Depths") or []
    if not depths:
        raise ValueError(f"No depth returned for '{symbol}'.")
    d = depths[0]
    return {
        "symbol": d.get("s", symbol),
        "bids": (d.get("b") or [])[:levels],
        "asks": (d.get("a") or [])[:levels],
    }


@mcp.tool()
def get_recent_trades(symbol: str, count: int = 50) -> list[dict]:
    """Recent executed trades (time & sales) for a futures contract.

    Args:
        symbol: Exchange-qualified IronBeam symbol, e.g. "XCME:NQ.Z26".
        count: Number of trades to return (1-100).
    """
    count = max(1, min(count, 100))
    to_ms = int(time.time() * 1000)
    from_ms = to_ms - 24 * 60 * 60 * 1000
    # earlier=true walks backward from `to_ms` so the newest trades come first.
    path = (
        f"/market/trades/{quote(symbol, safe='')}"
        f"/{from_ms}/{to_ms}/{count}/true"
    )
    data = _get(path)
    trades = data.get("trades") or data.get("traders") or []
    return [
        {
            "symbol": t.get("symbol", symbol),
            "price": t.get("price"),
            "size": t.get("size"),
        }
        for t in trades
    ]


if __name__ == "__main__":
    mcp.run()
