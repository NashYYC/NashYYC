# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp[cli]>=1.4.0",
#     "yfinance>=0.2.50",
# ]
# ///
"""Market-data MCP server for Claude Code.

Exposes free Yahoo Finance data — quotes, OHLCV history, and common
technical indicators — as MCP tools so Claude can pull live numbers on
demand. This is the pull-based companion to TradingView webhook alerts:
ask Claude a question, it queries the market.
"""

from __future__ import annotations

import pandas as pd
import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("tradingview")


def _history(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.Ticker(symbol).history(period=period, interval=interval)
    if df.empty:
        raise ValueError(
            f"No data for symbol '{symbol}'. Check the ticker — "
            "e.g. AAPL, TSLA, BTC-USD, EURUSD=X."
        )
    return df


def _round(value, digits: int = 2):
    if value is None or pd.isna(value):
        return None
    return round(float(value), digits)


def _volume(value):
    return None if pd.isna(value) else int(value)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    # Wilder's smoothing via an EWMA with alpha = 1 / period.
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)


@mcp.tool()
def get_quote(symbol: str) -> dict:
    """Latest price snapshot for a ticker.

    Args:
        symbol: Ticker, e.g. "AAPL", "TSLA", "BTC-USD", "EURUSD=X".
    """
    df = _history(symbol, period="5d", interval="1d")
    latest = df.iloc[-1]
    prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else None
    price = float(latest["Close"])
    change = price - prev_close if prev_close else None
    change_pct = (change / prev_close * 100) if prev_close else None
    return {
        "symbol": symbol.upper(),
        "price": _round(price),
        "previous_close": _round(prev_close),
        "change": _round(change),
        "change_percent": _round(change_pct),
        "day_open": _round(latest["Open"]),
        "day_high": _round(latest["High"]),
        "day_low": _round(latest["Low"]),
        "volume": _volume(latest["Volume"]),
        "as_of": str(df.index[-1].date()),
    }


@mcp.tool()
def get_history(
    symbol: str,
    period: str = "1mo",
    interval: str = "1d",
    limit: int = 30,
) -> list[dict]:
    """OHLCV price history for a ticker.

    Args:
        symbol: Ticker symbol.
        period: Lookback window — 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max.
        interval: Bar size — 1m, 5m, 15m, 1h, 1d, 1wk, 1mo.
            Intraday intervals are limited by Yahoo to roughly the last 60 days.
        limit: Maximum number of most-recent bars to return.
    """
    df = _history(symbol, period, interval).tail(limit)
    return [
        {
            "time": str(ts),
            "open": _round(row["Open"]),
            "high": _round(row["High"]),
            "low": _round(row["Low"]),
            "close": _round(row["Close"]),
            "volume": _volume(row["Volume"]),
        }
        for ts, row in df.iterrows()
    ]


@mcp.tool()
def get_indicators(
    symbol: str,
    period: str = "6mo",
    interval: str = "1d",
) -> dict:
    """Common technical indicators for a ticker: RSI, SMA, EMA, MACD.

    Args:
        symbol: Ticker symbol.
        period: Lookback window used for the calculation.
        interval: Bar size — 1m, 5m, 15m, 1h, 1d, 1wk, 1mo.
    """
    close = _history(symbol, period, interval)["Close"]
    macd_line = _ema(close, 12) - _ema(close, 26)
    signal_line = _ema(macd_line, 9)
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "as_of": str(close.index[-1]),
        "price": _round(close.iloc[-1]),
        "rsi_14": _round(_rsi(close, 14).iloc[-1]),
        "sma_20": _round(close.rolling(20).mean().iloc[-1]),
        "sma_50": _round(close.rolling(50).mean().iloc[-1]),
        "ema_12": _round(_ema(close, 12).iloc[-1]),
        "ema_26": _round(_ema(close, 26).iloc[-1]),
        "macd": _round(macd_line.iloc[-1], 4),
        "macd_signal": _round(signal_line.iloc[-1], 4),
        "macd_histogram": _round((macd_line - signal_line).iloc[-1], 4),
    }


if __name__ == "__main__":
    mcp.run()
