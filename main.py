#!/usr/bin/env python3
"""
BTC Pre-Correction Warning System
=================================
A 4-pillar scoring system that flags conditions which have historically
preceded Bitcoin corrections. Each pillar scores 0 or 1; the total (0-4)
maps to a threat level.

Pillars
-------
1. Derivatives Overload   - OI expanding while price stalls + hot leverage (funding OR futures basis)
2. On-Chain Distribution  - rising exchange reserves / multi-day inflow spike near highs
3. Technical Exhaustion   - bearish RSI divergence + fading breakout volume
4. Market Psychology      - Fear & Greed in "Extreme Greed" for an extended period

Install:
    pip install requests pandas numpy ta ccxt yfinance

Optional (better Pillar 2 data): put this line in a .env file next to the script
    CRYPTOQUANT_API_KEY=your_key

DISCLAIMER: Educational tool, not financial advice. Signals are heuristics and
will produce false positives and false negatives.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests

try:
    from ta.momentum import RSIIndicator
except ImportError:  # pragma: no cover
    sys.exit("Missing dependency: pip install ta")

try:
    import ccxt  # type: ignore
except ImportError:
    ccxt = None

try:
    import yfinance as yf  # type: ignore
except ImportError:
    yf = None


def load_env_file() -> None:
    """Minimal .env loader (KEY=VALUE lines) so no extra package is needed.
    Looks in the current directory and next to this script; never overrides
    variables that are already set in the real environment."""
    here = os.path.dirname(os.path.abspath(__file__))
    for path in (os.path.join(os.getcwd(), ".env"), os.path.join(here, ".env")):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.replace("export ", "").strip()
                os.environ.setdefault(key, val.strip().strip('"').strip("'"))


load_env_file()


# ----------------------------------------------------------------------------
# Configuration (tune these thresholds to your own risk preferences)
# ----------------------------------------------------------------------------
HTTP_TIMEOUT = 10
HTTP_RETRIES = 3

# Pillar 1
OI_EXPANSION_PCT = 5.0        # OI growth over 24h (%) considered "rapid"
PRICE_STALL_PCT = 2.0         # |price change 24h| below this = "stalling" (%)
FUNDING_THRESHOLD = 0.0001    # +0.01% per funding interval
FUNDING_MIN_POSITIVE = 0.66   # share of recent intervals that must be positive
FUNDING_LOOKBACK = 9          # ~3 days of 8h intervals
BASIS_HOT_PCT = 10.0          # annualized futures basis (%) considered overheated
BASIS_MIN_DAYS = 7            # ignore near-expiry contracts (noisy)

# Pillar 2
PRICE_NEAR_HIGH_PCT = 5.0     # price within X% of 30d high counts as "high"
INFLOW_SPIKE_MULT = 1.5       # last-3d avg inflow vs prior baseline
RESERVE_RISE_PCT = 0.5        # reserve growth over 7d (%)

# Pillar 3
RSI_PERIOD = 14
SWING_ORDER = 3               # candles each side to confirm a swing high
RSI_DIVERGENCE_MIN = 1.0      # minimum RSI drop between swing highs
DIVERGENCE_LOOKBACK = {"1d": 60, "4h": 90, "1h": 120}
DIVERGENCE_RECENCY = {"1d": 15, "4h": 24, "1h": 48}   # latest swing must be this recent

# Pillar 4
GREED_LEVEL = 75
GREED_DAYS_WINDOW = 7
GREED_DAYS_REQUIRED = 5

# Colors (disabled automatically when output isn't a terminal)
USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
C = {
    "green": "\033[92m" if USE_COLOR else "",
    "yellow": "\033[93m" if USE_COLOR else "",
    "red": "\033[91m" if USE_COLOR else "",
    "bold": "\033[1m" if USE_COLOR else "",
    "dim": "\033[2m" if USE_COLOR else "",
    "end": "\033[0m" if USE_COLOR else "",
}


# ----------------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------------
@dataclass
class PillarResult:
    """Container for one pillar's outcome."""
    name: str
    score: int = 0
    data_ok: bool = False            # False => fallback default (0) was used
    summary: str = ""
    details: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------
# HTTP helper
# ----------------------------------------------------------------------------
def http_get_json(url: str, params: Optional[dict] = None,
                  headers: Optional[dict] = None) -> Optional[Any]:
    """
    GET a URL and return parsed JSON, or None on failure.
    Handles rate limits (HTTP 429) with Retry-After / exponential backoff
    and retries transient network errors.
    """
    for attempt in range(1, HTTP_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, headers=headers,
                                timeout=HTTP_TIMEOUT)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 2 ** attempt))
                time.sleep(min(wait, 30))
                continue
            if resp.status_code in (451, 403):   # geo-block / forbidden: don't retry
                return None
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError):
            if attempt < HTTP_RETRIES:
                time.sleep(2 ** (attempt - 1))
    return None


# ----------------------------------------------------------------------------
# Market data (OHLCV)
# ----------------------------------------------------------------------------
def _ohlcv_via_ccxt(timeframe: str, limit: int) -> Optional[pd.DataFrame]:
    """Try several spot exchanges through CCXT until one returns candles."""
    if ccxt is None:
        return None
    for ex_id in ("kraken", "coinbase", "bitstamp", "binance"):
        for symbol in ("BTC/USDT", "BTC/USD"):
            try:
                ex = getattr(ccxt, ex_id)({"enableRateLimit": True, "timeout": 10000})
                raw = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
                if raw and len(raw) > 50:
                    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
                    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
                    return df.set_index("ts").astype(float)
            except Exception:
                continue
    return None


def _ohlcv_via_yfinance(timeframe: str) -> Optional[pd.DataFrame]:
    """Fallback OHLCV source using Yahoo Finance."""
    if yf is None:
        return None
    try:
        period = "1y" if timeframe == "1d" else "60d"
        interval = "1h" if timeframe == "4h" else timeframe   # yfinance has no 4h
        df = yf.download("BTC-USD", period=period, interval=interval,
                         progress=False, auto_adjust=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df = df.dropna().astype(float)
        if timeframe == "4h":
            df = df.resample("4h").agg({"open": "first", "high": "max", "low": "min",
                                        "close": "last", "volume": "sum"}).dropna()
        return df
    except Exception:
        return None


def fetch_ohlcv(timeframe: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch BTC OHLCV, preferring CCXT and falling back to yfinance."""
    df = _ohlcv_via_ccxt(timeframe, limit)
    if df is None:
        df = _ohlcv_via_yfinance(timeframe)
    return df


# ----------------------------------------------------------------------------
# Extra free data: futures basis (scored) and market context (display only)
# ----------------------------------------------------------------------------
def fetch_futures_basis() -> Optional[tuple[float, str]]:
    """
    Median annualized basis of Deribit BTC dated futures (mark vs index).
    A high basis means traders pay a premium to stay leveraged long.
    Returns (annualized_pct, note) or None.
    """
    data = http_get_json("https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                         {"currency": "BTC", "kind": "future"})
    try:
        rows = data["result"]
    except (TypeError, KeyError):
        return None
    now = datetime.now(timezone.utc)
    vals = []
    for r in rows:
        parts = str(r.get("instrument_name", "")).split("-")
        if len(parts) != 2 or parts[1] == "PERPETUAL":
            continue
        try:
            expiry = datetime.strptime(parts[1], "%d%b%y").replace(hour=8, tzinfo=timezone.utc)
        except ValueError:
            continue
        days = (expiry - now).total_seconds() / 86400
        mark, index = r.get("mark_price"), r.get("underlying_price")
        if days < BASIS_MIN_DAYS or not mark or not index:
            continue
        vals.append((mark / index - 1) * 365 / days * 100)
    if not vals:
        return None
    return float(np.median(vals)), f"median of {len(vals)} Deribit expiries"


def fetch_market_context() -> list[str]:
    """
    Informational readings that are NOT scored (weaker or noisier evidence):
    long/short ratios, Coinbase premium, and Deribit DVOL. Each fails independently.
    """
    lines: list[str] = []

    # Long/short ratios: try Binance, then Bybit, then OKX
    def _ls_binance() -> list[str]:
        base = "https://fapi.binance.com/futures/data/"
        prm = {"symbol": "BTCUSDT", "period": "1h", "limit": 24}
        out = []
        g = http_get_json(base + "globalLongShortAccountRatio", prm)
        t = http_get_json(base + "topLongShortPositionRatio", prm)
        if g:
            v = [float(r["longShortRatio"]) for r in g]
            out.append(f"Retail long/short accounts (Binance): {v[-1]:.2f} "
                       f"(24h avg {np.mean(v):.2f}); above ~1.8 = crowded longs")
        if t:
            v = [float(r["longShortRatio"]) for r in t]
            out.append(f"Top-trader long/short positions (Binance): {v[-1]:.2f} "
                       f"(24h avg {np.mean(v):.2f})")
        return out

    def _ls_bybit() -> list[str]:
        d = http_get_json("https://api.bybit.com/v5/market/account-ratio",
                          {"category": "linear", "symbol": "BTCUSDT", "period": "1h", "limit": 24})
        v = [float(r["buyRatio"]) / float(r["sellRatio"]) for r in d["result"]["list"]]
        return [f"Retail long/short accounts (Bybit): {v[0]:.2f} "
                f"(24h avg {np.mean(v):.2f}); above ~1.8 = crowded longs"]

    def _ls_okx() -> list[str]:
        d = http_get_json("https://www.okx.com/api/v5/rubik/stat/contracts/long-short-account-ratio",
                          {"ccy": "BTC", "period": "1H"})
        v = [float(r[1]) for r in d["data"][:24]]
        return [f"Retail long/short accounts (OKX): {v[0]:.2f} "
                f"(24h avg {np.mean(v):.2f}); above ~1.8 = crowded longs"]

    for fn in (_ls_binance, _ls_bybit, _ls_okx):
        try:
            out = fn()
        except Exception:
            out = None
        if out:
            lines.extend(out)
            break
    else:
        lines.append("Long/short ratios: unavailable (Binance, Bybit and OKX all failed "
                     "from this network)")

    # Coinbase premium (US spot demand gauge): Coinbase BTC-USD vs BTC-USDT elsewhere
    ua = {"User-Agent": "btc-warning/1.0"}

    def _cb_price() -> float:
        d = http_get_json("https://api.exchange.coinbase.com/products/BTC-USD/ticker", headers=ua)
        if d and d.get("price"):
            return float(d["price"])
        d = http_get_json("https://api.coinbase.com/v2/prices/BTC-USD/spot", headers=ua)
        return float(d["data"]["amount"])

    def _ref_binance() -> float:
        return float(http_get_json("https://api.binance.com/api/v3/ticker/price",
                                   {"symbol": "BTCUSDT"})["price"])

    def _ref_bybit() -> float:
        d = http_get_json("https://api.bybit.com/v5/market/tickers",
                          {"category": "spot", "symbol": "BTCUSDT"})
        return float(d["result"]["list"][0]["lastPrice"])

    def _ref_okx() -> float:
        d = http_get_json("https://www.okx.com/api/v5/market/ticker", {"instId": "BTC-USDT"})
        return float(d["data"][0]["last"])

    def _ref_kraken() -> float:
        d = http_get_json("https://api.kraken.com/0/public/Ticker", {"pair": "XBTUSDT"})
        return float(next(iter(d["result"].values()))["c"][0])

    cb_price = None
    try:
        cb_price = _cb_price()
    except Exception:
        pass
    ref_price, ref_name = None, ""
    for name, fn in (("Binance", _ref_binance), ("Bybit", _ref_bybit),
                     ("OKX", _ref_okx), ("Kraken", _ref_kraken)):
        try:
            ref_price, ref_name = fn(), name
            break
        except Exception:
            continue
    if cb_price and ref_price:
        prem = (cb_price / ref_price - 1) * 100
        lines.append(f"Coinbase premium vs {ref_name} USDT: {prem:+.3f}% "
                     f"(snapshot; negative = weak US demand; USDT/USD gap adds noise)")
    else:
        missing = []
        if not cb_price:
            missing.append("Coinbase price")
        if not ref_price:
            missing.append("USDT reference price (Binance/Bybit/OKX/Kraken)")
        lines.append(f"Coinbase premium: unavailable ({' and '.join(missing)} not reachable)")

    # Deribit DVOL (implied volatility): very low vs recent range = complacency
    try:
        now_ms = int(time.time() * 1000)
        d = http_get_json("https://www.deribit.com/api/v2/public/get_volatility_index_data",
                          {"currency": "BTC", "start_timestamp": now_ms - 30 * 86400 * 1000,
                           "end_timestamp": now_ms, "resolution": "1D"})
        closes = [float(r[4]) for r in d["result"]["data"]]
        rank = sum(c <= closes[-1] for c in closes) / len(closes) * 100
        lines.append(f"Deribit DVOL: {closes[-1]:.1f} ({rank:.0f}th percentile of last 30d); "
                     f"low percentile = complacent options market")
    except Exception:
        lines.append("Deribit DVOL: unavailable")
    return lines


# ----------------------------------------------------------------------------
# PILLAR 1: Derivatives Overload
# ----------------------------------------------------------------------------
def _oi_change_binance() -> Optional[float]:
    """24h open-interest % change on Binance USDT-M BTC perpetual."""
    data = http_get_json("https://fapi.binance.com/futures/data/openInterestHist",
                         {"symbol": "BTCUSDT", "period": "1h", "limit": 25})
    if not data or len(data) < 2:
        return None
    first, last = float(data[0]["sumOpenInterest"]), float(data[-1]["sumOpenInterest"])
    return (last / first - 1) * 100 if first else None


def _oi_change_bybit() -> Optional[float]:
    """24h open-interest % change on Bybit linear BTC perpetual."""
    data = http_get_json("https://api.bybit.com/v5/market/open-interest",
                         {"category": "linear", "symbol": "BTCUSDT",
                          "intervalTime": "1h", "limit": 25})
    try:
        rows = data["result"]["list"]          # newest first
        last, first = float(rows[0]["openInterest"]), float(rows[-1]["openInterest"])
        return (last / first - 1) * 100 if first else None
    except (TypeError, KeyError, IndexError, ValueError):
        return None


def _oi_change_okx() -> Optional[float]:
    """Fallback OI % change from OKX (USD-valued, so slightly price-contaminated)."""
    data = http_get_json("https://www.okx.com/api/v5/rubik/stat/contracts/open-interest-volume",
                         {"ccy": "BTC", "period": "1H"})
    try:
        rows = data["data"]                      # newest first: [ts, oi, vol]
        last, first = float(rows[0][1]), float(rows[min(24, len(rows) - 1)][1])
        return (last / first - 1) * 100 if first else None
    except (TypeError, KeyError, IndexError, ValueError):
        return None


def _funding_binance() -> Optional[list[float]]:
    data = http_get_json("https://fapi.binance.com/fapi/v1/fundingRate",
                         {"symbol": "BTCUSDT", "limit": FUNDING_LOOKBACK})
    try:
        return [float(r["fundingRate"]) for r in data] or None
    except (TypeError, KeyError, ValueError):
        return None


def _funding_bybit() -> Optional[list[float]]:
    data = http_get_json("https://api.bybit.com/v5/market/funding/history",
                         {"category": "linear", "symbol": "BTCUSDT",
                          "limit": FUNDING_LOOKBACK})
    rates: list[float] = []
    try:
        rates = [float(r["fundingRate"]) for r in data["result"]["list"]]
    except (TypeError, KeyError, ValueError):
        return None
    # Add the live (not yet settled) rate so the reading is fresher
    live = http_get_json("https://api.bybit.com/v5/market/tickers",
                         {"category": "linear", "symbol": "BTCUSDT"})
    try:
        rates.insert(0, float(live["result"]["list"][0]["fundingRate"]))
    except (TypeError, KeyError, IndexError, ValueError):
        pass
    return rates or None


def _funding_okx() -> Optional[list[float]]:
    data = http_get_json("https://www.okx.com/api/v5/public/funding-rate-history",
                         {"instId": "BTC-USDT-SWAP", "limit": FUNDING_LOOKBACK})
    try:
        rates = [float(r["fundingRate"]) for r in data["data"]]
    except (TypeError, KeyError, ValueError):
        return None
    live = http_get_json("https://www.okx.com/api/v5/public/funding-rate",
                         {"instId": "BTC-USDT-SWAP"})
    try:
        rates.insert(0, float(live["data"][0]["fundingRate"]))
    except (TypeError, KeyError, IndexError, ValueError):
        pass
    return rates or None


def evaluate_pillar1(price_change_24h: Optional[float],
                     basis: Optional[tuple[float, str]]) -> PillarResult:
    """
    Score 1 if ALL are true:
      * Aggregate OI grew > OI_EXPANSION_PCT over 24h
      * Price is stalling (|24h change| < PRICE_STALL_PCT)
      * Leverage is hot: cross-venue funding > FUNDING_THRESHOLD (and persistently
        positive) OR annualized futures basis > BASIS_HOT_PCT
    """
    res = PillarResult("Derivatives Overload")
    try:
        oi_changes = {k: v for k, v in
                      {"Binance": _oi_change_binance(), "Bybit": _oi_change_bybit()}.items()
                      if v is not None}
        if not oi_changes:                       # both primary venues failed
            v = _oi_change_okx()
            if v is not None:
                oi_changes["OKX (USD-valued)"] = v
        funding = {k: v for k, v in
                   {"Binance": _funding_binance(), "Bybit": _funding_bybit(),
                    "OKX": _funding_okx()}.items() if v}
        basis_pct = basis[0] if basis else None

        if not oi_changes or price_change_24h is None or (not funding and basis_pct is None):
            res.summary = "Insufficient data - defaulted to 0"
            if not oi_changes:
                res.details.append("Open interest unavailable from all venues")
            if not funding and basis_pct is None:
                res.details.append("Neither funding rates nor futures basis available")
            if price_change_24h is None:
                res.details.append("Price change unavailable")
            return res

        res.data_ok = True
        avg_oi = float(np.mean(list(oi_changes.values())))
        oi_expanding = avg_oi > OI_EXPANSION_PCT
        price_stalling = abs(price_change_24h) < PRICE_STALL_PCT

        res.details += [
            f"OI 24h change (avg of {', '.join(oi_changes)}): {avg_oi:+.2f}%  "
            f"[need > +{OI_EXPANSION_PCT}%] -> {'YES' if oi_expanding else 'no'}",
            f"Price 24h change: {price_change_24h:+.2f}%  "
            f"[need within +/-{PRICE_STALL_PCT}%] -> {'YES' if price_stalling else 'no'}",
        ]

        funding_hot = False
        if funding:
            avg_funding = float(np.mean([np.mean(v) for v in funding.values()]))
            all_rates = np.concatenate([np.array(v) for v in funding.values()])
            pos_share = float((all_rates > 0).mean())
            funding_hot = avg_funding > FUNDING_THRESHOLD and pos_share >= FUNDING_MIN_POSITIVE
            res.details.append(
                f"Avg funding/interval ({', '.join(funding)}, incl. live rate where available): "
                f"{avg_funding * 100:+.4f}%  [need > +{FUNDING_THRESHOLD * 100:.2f}%] ; "
                f"positive share {pos_share:.0%} -> {'YES' if funding_hot else 'no'}")
        else:
            res.details.append("Funding rates unavailable")

        basis_hot = basis_pct is not None and basis_pct > BASIS_HOT_PCT
        if basis:
            res.details.append(f"Annualized futures basis: {basis_pct:.1f}% ({basis[1]})  "
                               f"[need > {BASIS_HOT_PCT}%] -> {'YES' if basis_hot else 'no'}")
        else:
            res.details.append("Futures basis unavailable")

        if oi_expanding and price_stalling and (funding_hot or basis_hot):
            res.score, res.summary = 1, "Leverage building into a stall - overload"
        else:
            res.summary = "Derivatives positioning not at overload levels"
    except Exception as exc:  # last-resort guard
        res.summary = f"Error ({exc}) - defaulted to 0"
    return res


# ----------------------------------------------------------------------------
# PILLAR 2: On-Chain Flows / Whale Distribution
# ----------------------------------------------------------------------------
CQ_NOTES: list[str] = []   # diagnostics shown in the report (e.g. plan limits)


def _cryptoquant_series(endpoint: str, value_key_candidates: tuple[str, ...],
                        api_key: str, limit: int = 30) -> Optional[pd.Series]:
    """Fetch a daily CryptoQuant series (oldest -> newest). Records why it failed."""
    url = f"https://api.cryptoquant.com/v1/btc/exchange-flows/{endpoint}"
    try:
        resp = requests.get(url, params={"exchange": "all_exchange", "window": "day",
                                         "limit": limit},
                            headers={"Authorization": f"Bearer {api_key}"},
                            timeout=HTTP_TIMEOUT)
    except requests.RequestException as exc:
        CQ_NOTES.append(f"CryptoQuant {endpoint}: network error ({type(exc).__name__})")
        return None
    if resp.status_code != 200:
        try:
            msg = str(resp.json())[:100]
        except ValueError:
            msg = ""
        CQ_NOTES.append(f"CryptoQuant {endpoint}: HTTP {resp.status_code} {msg} "
                        f"(likely not included in your plan)")
        return None
    try:
        rows = resp.json()["result"]["data"]
        key = next(k for k in value_key_candidates if k in rows[0])
        s = pd.Series([float(r[key]) for r in rows])
        if "date" in rows[0]:
            s.index = pd.to_datetime([r["date"] for r in rows])
            s = s.sort_index()
        return s.reset_index(drop=True)
    except (TypeError, KeyError, IndexError, StopIteration, ValueError):
        CQ_NOTES.append(f"CryptoQuant {endpoint}: unexpected response format")
        return None


def _onchain_volume_proxy() -> Optional[pd.Series]:
    """
    Free proxy: daily on-chain transfer volume (BTC) from blockchain.com.
    NOT exchange-specific - a spike only hints at coin movement. Weak signal.
    """
    data = http_get_json("https://api.blockchain.info/charts/estimated-transaction-volume",
                         {"timespan": "35days", "format": "json", "cors": "true"})
    try:
        return pd.Series([float(p["y"]) for p in data["values"]])
    except (TypeError, KeyError, ValueError):
        return None


def evaluate_pillar2(price_near_high: Optional[bool]) -> PillarResult:
    """
    Score 1 if price is near its 30d high AND (exchange reserves are rising
    over 7d OR the last-3d average inflow spikes vs the prior baseline).
    Uses CryptoQuant if CRYPTOQUANT_API_KEY is set; otherwise a weak free proxy.
    """
    res = PillarResult("On-Chain Distribution")
    try:
        if price_near_high is None:
            res.summary = "Price context unavailable - defaulted to 0"
            return res

        api_key = os.environ.get("CRYPTOQUANT_API_KEY")
        if not api_key:
            CQ_NOTES.append("CRYPTOQUANT_API_KEY not found in environment or .env")
        inflow = reserve = None
        source = "proxy"

        if api_key:
            inflow = _cryptoquant_series("inflow", ("inflow_total", "inflow"), api_key)
            reserve = _cryptoquant_series("reserve", ("reserve",), api_key)
            source = "CryptoQuant"
        if inflow is None and reserve is None:
            inflow = _onchain_volume_proxy()
            source = "blockchain.com volume PROXY (not exchange-specific)"

        if inflow is None and reserve is None:
            res.summary = "On-chain data unavailable - defaulted to 0"
            return res

        res.data_ok = True
        spike = reserves_rising = False

        if inflow is not None and len(inflow) >= 10:
            recent = inflow.tail(3).mean()
            base = inflow.iloc[:-3].tail(27).mean()
            ratio = recent / base if base else 0.0
            spike = ratio > INFLOW_SPIKE_MULT
            res.details.append(f"Inflow/volume 3d avg vs baseline: {ratio:.2f}x "
                               f"[need > {INFLOW_SPIKE_MULT}x] -> {'YES' if spike else 'no'}")
        if reserve is not None and len(reserve) >= 8:
            chg = (reserve.iloc[-1] / reserve.iloc[-8] - 1) * 100
            reserves_rising = chg > RESERVE_RISE_PCT
            res.details.append(f"Exchange reserve 7d change: {chg:+.2f}% "
                               f"[need > +{RESERVE_RISE_PCT}%] -> {'YES' if reserves_rising else 'no'}")

        res.details.append(f"Price within {PRICE_NEAR_HIGH_PCT}% of 30d high -> "
                           f"{'YES' if price_near_high else 'no'}")
        res.details.append(f"Source: {source}")
        res.details.extend(CQ_NOTES)

        if price_near_high and (spike or reserves_rising):
            res.score, res.summary = 1, "Coins moving to exchanges near highs - distribution risk"
        else:
            res.summary = "No clear distribution signal"
    except Exception as exc:
        res.summary = f"Error ({exc}) - defaulted to 0"
    return res


# ----------------------------------------------------------------------------
# PILLAR 3: Technical Momentum & Exhaustion
# ----------------------------------------------------------------------------
def find_swing_highs(series: pd.Series, order: int = SWING_ORDER) -> list[int]:
    """Return integer positions of confirmed swing highs (unique local maxima)."""
    vals = series.to_numpy()
    out = []
    for i in range(order, len(vals) - order):
        window = vals[i - order:i + order + 1]
        if vals[i] == window.max() and (window == vals[i]).sum() == 1:
            out.append(i)
    return out


def analyze_timeframe(df: pd.DataFrame, tf: str) -> dict:
    """
    Check one timeframe for bearish RSI divergence and declining volume.
    The final (still-forming) candle is dropped so partial volume doesn't mislead.
    """
    df = df.iloc[:-1].copy()
    df["rsi"] = RSIIndicator(df["close"], window=RSI_PERIOD).rsi()
    df["vol_sma5"] = df["volume"].rolling(5).mean()
    df["vol_sma20"] = df["volume"].rolling(20).mean()
    df = df.dropna()

    out = {"tf": tf, "divergence": False, "vol_declining": False, "note": "insufficient swings"}
    window = df.tail(DIVERGENCE_LOOKBACK[tf]).reset_index(drop=True)
    swings = find_swing_highs(window["high"])
    if len(swings) < 2:
        return out

    i1, i2 = swings[-2], swings[-1]
    recent_enough = (len(window) - 1 - i2) <= DIVERGENCE_RECENCY[tf]
    price_hh = window["high"][i2] > window["high"][i1]
    rsi1, rsi2 = window["rsi"][i1], window["rsi"][i2]
    rsi_lh = rsi2 < rsi1 - RSI_DIVERGENCE_MIN
    out["divergence"] = bool(recent_enough and price_hh and rsi_lh)

    vol_peak_lower = window["volume"][i2] < window["volume"][i1]
    vol_trend_down = window["vol_sma5"].iloc[-1] < window["vol_sma20"].iloc[-1]
    out["vol_declining"] = bool(vol_peak_lower and vol_trend_down)

    out["note"] = (f"highs {window['high'][i1]:,.0f} -> {window['high'][i2]:,.0f}; "
                   f"RSI {rsi1:.1f} -> {rsi2:.1f}; recent={recent_enough}; "
                   f"swing vol {'lower' if vol_peak_lower else 'higher'}; "
                   f"SMA5 {'<' if vol_trend_down else '>='} SMA20")
    return out


def evaluate_pillar3(frames: dict[str, Optional[pd.DataFrame]]) -> PillarResult:
    """
    Score 1 if, on the 1d, 4h or 1h timeframe, price makes a higher high while
    RSI makes a lower high AND breakout volume is declining.
    """
    res = PillarResult("Technical Exhaustion")
    try:
        usable = {tf: df for tf, df in frames.items() if df is not None and len(df) > 60}
        if not usable:
            res.summary = "OHLCV data unavailable - defaulted to 0"
            return res

        res.data_ok = True
        triggered = False
        for tf, df in usable.items():
            r = analyze_timeframe(df, tf)
            hit = r["divergence"] and r["vol_declining"]
            triggered |= hit
            res.details.append(f"[{tf}] bearish divergence: {'YES' if r['divergence'] else 'no'} | "
                               f"volume declining: {'YES' if r['vol_declining'] else 'no'} "
                               f"-> {'SIGNAL' if hit else 'clear'}")
            res.details.append(f"     {r['note']}")
        for tf in set(frames) - set(usable):
            res.details.append(f"[{tf}] data unavailable")

        if triggered:
            res.score, res.summary = 1, "Momentum exhaustion: bearish divergence on fading volume"
        else:
            res.summary = "No confirmed divergence + volume fade"
    except Exception as exc:
        res.summary = f"Error ({exc}) - defaulted to 0"
    return res


# ----------------------------------------------------------------------------
# PILLAR 4: Market Psychology (Fear & Greed)
# ----------------------------------------------------------------------------
def evaluate_pillar4() -> PillarResult:
    """
    Score 1 if today's Fear & Greed >= GREED_LEVEL and at least
    GREED_DAYS_REQUIRED of the last GREED_DAYS_WINDOW days were also >= GREED_LEVEL.
    """
    res = PillarResult("Market Psychology")
    try:
        data = http_get_json("https://api.alternative.me/fng/", {"limit": 14})
        rows = data["data"]                       # newest first
        scores = [int(r["value"]) for r in rows]
        label = rows[0].get("value_classification", "")
        if not scores:
            raise ValueError("empty response")

        res.data_ok = True
        window = scores[:GREED_DAYS_WINDOW]
        days_greedy = sum(s >= GREED_LEVEL for s in window)
        extended = scores[0] >= GREED_LEVEL and days_greedy >= GREED_DAYS_REQUIRED

        res.details += [
            f"Current index: {scores[0]} ({label})",
            f"Last {len(window)} days: {window}",
            f"Days >= {GREED_LEVEL}: {days_greedy}/{len(window)} "
            f"[need {GREED_DAYS_REQUIRED}] -> {'YES' if extended else 'no'}",
        ]
        if extended:
            res.score, res.summary = 1, "Sustained Extreme Greed - crowd euphoria"
        else:
            res.summary = "Sentiment not in sustained Extreme Greed"
    except Exception as exc:
        res.summary = f"Sentiment unavailable ({type(exc).__name__}) - defaulted to 0"
    return res


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def threat_level(total: int) -> tuple[str, str]:
    """Map the total score to (label, color)."""
    if total <= 1:
        return "SAFE ZONE", C["green"]
    if total == 2:
        return "CAUTION ZONE", C["yellow"]
    return "DANGER ZONE - CORRECTION IMMINENT", C["red"]


def print_report(results: list[PillarResult], price: Optional[float],
                 context: Optional[list[str]] = None) -> None:
    """Print the formatted Pre-Correction Diagnostic Report."""
    line = "=" * 74
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"\n{C['bold']}{line}\n  BTC PRE-CORRECTION DIAGNOSTIC REPORT\n{line}{C['end']}")
    print(f"  Generated: {now}")
    print(f"  BTC price: {f'${price:,.0f}' if price else 'unavailable'}\n")

    for i, r in enumerate(results, 1):
        color = C["red"] if r.score else C["green"]
        flag = "" if r.data_ok else f"  {C['yellow']}[DATA FALLBACK]{C['end']}"
        print(f"{C['bold']}Pillar {i}: {r.name}{C['end']}  ->  "
              f"{color}Score {r.score}/1{C['end']}{flag}")
        print(f"  {r.summary}")
        for d in r.details:
            print(f"  {C['dim']}- {d}{C['end']}")
        print()

    if context:
        print(f"{C['bold']}Market context (informational, not scored){C['end']}")
        for c in context:
            print(f"  {C['dim']}- {c}{C['end']}")
        print()

    total = sum(r.score for r in results)
    label, color = threat_level(total)
    failed = [r.name for r in results if not r.data_ok]
    print(line)
    print(f"{C['bold']}  FINAL RISK SCORE: {total} / 4{C['end']}")
    print(f"  {C['bold']}{color}THREAT LEVEL: {label}{C['end']}")
    if failed:
        print(f"  {C['yellow']}Warning: fallback defaults used for: {', '.join(failed)}. "
              f"Score may understate risk.{C['end']}")
    print(line)
    print(f"{C['dim']}  Heuristic signals only - not financial advice.{C['end']}\n")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main() -> None:
    print("Fetching market data...", file=sys.stderr)
    daily = fetch_ohlcv("1d", 300)
    four_h = fetch_ohlcv("4h", 300)
    hourly = fetch_ohlcv("1h", 300)

    # Shared price context for Pillars 1 and 2
    price = float(hourly["close"].iloc[-1]) if hourly is not None else (
        float(daily["close"].iloc[-1]) if daily is not None else None)

    change_24h = None
    if hourly is not None and len(hourly) > 25:
        change_24h = (hourly["close"].iloc[-1] / hourly["close"].iloc[-25] - 1) * 100

    near_high = None
    if daily is not None and len(daily) >= 30:
        high_30d = daily["high"].tail(30).max()
        near_high = bool(daily["close"].iloc[-1] >= high_30d * (1 - PRICE_NEAR_HIGH_PCT / 100))

    print("Evaluating pillars...", file=sys.stderr)
    results = [
        evaluate_pillar1(change_24h, fetch_futures_basis()),
        evaluate_pillar2(near_high),
        evaluate_pillar3({"1d": daily, "4h": four_h, "1h": hourly}),
        evaluate_pillar4(),
    ]
    print_report(results, price, fetch_market_context())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted.")