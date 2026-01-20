"""
Build a clean US "tech" universe (Option C):
  Universe = XLK holdings ∪ SOXX holdings ∪ IGV holdings ∪ Nasdaq-100 constituents (QQQ proxy)

Then filter to top N by average daily dollar volume (close * volume) using yfinance,
and save to tech_universe_C.csv

Dependencies:
  pip install pandas requests yfinance openpyxl lxml

Notes:
- XLK holdings are downloaded as an XLSX from SSGA.
- SOXX/IGV holdings are downloaded as CSV from iShares "Download Holdings" endpoints.
- Nasdaq-100 constituents are scraped from Nasdaq's official Nasdaq-100 companies page.
"""

from __future__ import annotations

import io
import re
import time
from dataclasses import dataclass
from typing import Iterable, List, Set, Optional

import pandas as pd
import requests
import yfinance as yf


# -----------------------
# Source URLs (as of Jan 2026)
# -----------------------
SSGA_XLK_HOLDINGS_XLSX = (
    "https://www.ssga.com/library-content/products/fund-data/etfs/us/holdings-daily-us-en-xlk.xlsx"
)

# iShares "Download Holdings" endpoints (these URLs may change if iShares updates their site)
ISHARES_SOXX_HOLDINGS_CSV = (
    "https://www.ishares.com/ch/professionals/en/products/239705/ishares-phlx-semiconductor-etf/1495092304805.ajax?dataType=fund&fileName=SOXX_holdings&fileType=csv"
)
ISHARES_IGV_HOLDINGS_CSV = (
    "https://www.ishares.com/ch/professionals/en/products/239771/ishares-north-american-techsoftware-etf/1495092304805.ajax?dataType=fund&fileName=IGV_holdings&fileType=csv"
)

NASDAQ100_COMPANIES_PAGE = "https://www.nasdaq.com/solutions/global-indexes/nasdaq-100/companies"


# -----------------------
# Helpers
# -----------------------
def _http_get_bytes(url: str, timeout: int = 30) -> bytes:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.content


def _normalize_ticker(t: str) -> str:
    """
    Normalize tickers for yfinance:
    - Convert BRK.B -> BRK-B etc.
    - Strip whitespace
    """
    t = t.strip().upper()
    # Common conversions for yfinance
    t = t.replace(".", "-")
    return t


def _drop_non_equity_symbols(symbols: Iterable[str]) -> List[str]:
    """
    Basic hygiene: drop blanks and obvious non-common formats.
    ETFs themselves may appear in holdings sometimes; we just keep valid tickers.
    """
    out = []
    for s in symbols:
        if not isinstance(s, str):
            continue
        s = s.strip().upper()
        if not s:
            continue
        # Drop cash placeholders
        if s in {"CASH", "USD", "US DOLLAR"}:
            continue
        out.append(_normalize_ticker(s))
    return out


# -----------------------
# Holdings loaders
# -----------------------
def load_xlk_holdings() -> Set[str]:
    """
    Loads XLK holdings from SSGA XLSX.
    The sheet/columns can change; we look for a column containing 'Ticker' or 'Symbol'.
    """
    content = _http_get_bytes(SSGA_XLK_HOLDINGS_XLSX)
    # Read all sheets and find the first with a ticker-like column
    # skip thefirst four rows, the header start from the fifth row
    xls = pd.ExcelFile(content)
    for sheet in xls.sheet_names:
        df = xls.parse(sheet, skiprows=4)
        cols = {c.lower(): c for c in df.columns}
        print(cols)
        ticker_col = None
        for key in ["ticker", "symbol", "identifier"]:
            if key in cols:
                ticker_col = cols[key]
                break
        if ticker_col is None:
            continue

        tickers = _drop_non_equity_symbols(df[ticker_col].dropna().astype(str).tolist())
        # Heuristic: XLK should have dozens of holdings; accept if it looks right
        if len(tickers) >= 30:
            return set(tickers)

    raise RuntimeError("Could not find a holdings sheet/column for XLK in the downloaded XLSX.")


def load_ishares_holdings_csv(url: str) -> set[str]:
    content = _http_get_bytes(url)
    text = content.decode("utf-8", errors="ignore")
    lines = text.splitlines()

    # Find the first line that looks like the real CSV header
    header_idx = None
    for i, line in enumerate(lines):
        low = line.lower()
        # iShares holdings tables typically have one of these columns
        if ("ticker" in low) and ("," in line):
            # guard against metadata lines like "Fund Holdings as of,...."
            # require that it has multiple commas (i.e., multiple columns)
            if line.count(",") >= 5:
                header_idx = i
                break

    if header_idx is None:
        # Sometimes iShares returns HTML or an error message instead of CSV
        preview = "\n".join(lines[:30])
        raise RuntimeError(
            f"Could not locate holdings header in iShares response from {url}.\n"
            f"First lines were:\n{preview}"
        )

    csv_text = "\n".join(lines[header_idx:])

    # Use python engine for more robust CSV parsing
    df = pd.read_csv(io.StringIO(csv_text), engine="python")

    # Try common ticker column names
    possible_cols = ["Ticker", "Issuer Ticker", "Symbol", "Holding Ticker"]
    ticker_col = None
    for c in possible_cols:
        if c in df.columns:
            ticker_col = c
            break

    if ticker_col is None:
        # fallback: any column containing 'ticker' or 'symbol'
        for c in df.columns:
            if "ticker" in str(c).lower() or "symbol" in str(c).lower():
                ticker_col = c
                break

    if ticker_col is None:
        raise RuntimeError(
            f"Parsed iShares CSV but couldn't find a ticker column. Columns: {list(df.columns)}"
        )

    tickers = _drop_non_equity_symbols(df[ticker_col].dropna().astype(str).tolist())
    return set(tickers)


def load_nasdaq100_constituents() -> Set[str]:
    """
    Scrape Nasdaq-100 constituents from Nasdaq official page.
    The page includes a simple 'Symbol Company Name' list we can extract via regex.
    """
    html = _http_get_bytes(NASDAQ100_COMPANIES_PAGE).decode("utf-8", errors="ignore")

    # We look for lines like: "Symbol Company Name" followed by "AAPL APPLE INC."
    # We'll capture tokens that look like tickers (letters/numbers, sometimes with -).
    # Because Nasdaq page is mostly uppercase, this works well.
    # Filter to plausible tickers: 1-5 chars or longer ADS tickers? (keep up to 6-7)
    candidates = re.findall(r"\b[A-Z]{1,6}(?:-[A-Z])?\b", html)

    # Nasdaq page has lots of other uppercase words; restrict by checking against the "Symbol Company Name" section
    # A more robust method: use read_html to parse any tables that contain "Symbol" and "Company Name".
    try:
        tables = pd.read_html(html)
        best = None
        for t in tables:
            cols = [str(c).strip().lower() for c in t.columns]
            if "symbol" in cols and any("company" in c for c in cols):
                best = t
                break
        if best is not None:
            sym_col = best.columns[[str(c).strip().lower() == "symbol" for c in best.columns]][0]
            tickers = _drop_non_equity_symbols(best[sym_col].dropna().astype(str).tolist())
            if len(tickers) >= 80:
                return set(tickers)
    except Exception:
        pass

    # Fallback: heuristic filtering for tickers that appear near the section
    # Keep only those that also exist as yfinance tickers by trying a lightweight validation later.
    plausible = []
    for s in candidates:
        if s in {"NASDAQ", "INDEX", "COMPANY", "NAME", "TECHNOLOGY"}:
            continue
        # Avoid common uppercase noise
        if len(s) == 1:
            continue
        plausible.append(_normalize_ticker(s))

    # Deduplicate but keep order
    seen = set()
    out = []
    for s in plausible:
        if s not in seen:
            seen.add(s)
            out.append(s)

    # Nasdaq-100 is ~100 names; return a clipped set to avoid huge noise
    return set(out[:200])


# -----------------------
# Liquidity filtering (yfinance)
# -----------------------
def compute_avg_dollar_volume(
    tickers: List[str],
    lookback_days: int = 20,
    chunk_size: int = 50,
    pause: float = 1.5,
) -> pd.DataFrame:
    """
    Compute average dollar volume (mean(close * volume)) over last N calendar days using daily bars.
    Returns a dataframe with columns: ticker, avg_dollar_volume
    """
    results = []

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]
        print(f"[liquidity] Downloading daily for {i}..{i+len(chunk)-1} ({len(chunk)} tickers)")
        df = yf.download(
            tickers=" ".join(chunk),
            period=f"{lookback_days}d",
            interval="1d",
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )

        if df is None or df.empty:
            time.sleep(pause)
            continue

        if isinstance(df.columns, pd.MultiIndex):
            # MultiIndex: (field, ticker) or (ticker, field) depending on yfinance version
            # yfinance usually returns columns like: ('Close', 'AAPL')
            # We detect orientation by checking the first level labels
            lvl0 = set(map(str, df.columns.get_level_values(0)))
            if "Close" in lvl0 and "Volume" in lvl0:
                close = df["Close"]
                vol = df["Volume"]
            else:
                # orientation swapped
                close = df.xs("Close", level=1, axis=1, drop_level=False)
                vol = df.xs("Volume", level=1, axis=1, drop_level=False)
                # convert to ticker columns
                close.columns = close.columns.get_level_values(0)
                vol.columns = vol.columns.get_level_values(0)

            dv = close * vol
            avg = dv.mean(axis=0, skipna=True)
            for t, v in avg.items():
                if pd.notna(v) and v > 0:
                    results.append({"ticker": _normalize_ticker(str(t)), "avg_dollar_volume": float(v)})
        else:
            # Single ticker case
            if "Close" in df.columns and "Volume" in df.columns and len(chunk) == 1:
                dv = (df["Close"] * df["Volume"]).mean()
                results.append({"ticker": chunk[0], "avg_dollar_volume": float(dv)})

        time.sleep(pause)

    out = pd.DataFrame(results).drop_duplicates("ticker", keep="last")
    return out


# -----------------------
# Main
# -----------------------
def main(
    top_n: int = 200,
    min_avg_dollar_volume: Optional[float] = None,  # e.g., 20e6
    liquidity_lookback_days: int = 90,
):
    print("Loading holdings...")
    xlk = load_xlk_holdings()
    soxx = load_ishares_holdings_csv(ISHARES_SOXX_HOLDINGS_CSV)
    igv = load_ishares_holdings_csv(ISHARES_IGV_HOLDINGS_CSV)
    ndx = load_nasdaq100_constituents()  # QQQ proxy

    universe = set().union(xlk, soxx, igv, ndx)
    universe = set(_drop_non_equity_symbols(universe))

    # Optional: remove the ETFs themselves if they appear
    universe -= {"XLK", "QQQ", "SOXX", "IGV"}

    tickers = sorted(universe)
    print(f"Raw union size: {len(tickers)} tickers")

    # Liquidity filter
    liq = compute_avg_dollar_volume(
        tickers, lookback_days=liquidity_lookback_days, chunk_size=50, pause=1.5
    )

    # Merge and filter
    df = pd.DataFrame({"ticker": tickers}).merge(liq, on="ticker", how="left")
    df = df.dropna(subset=["avg_dollar_volume"])

    if min_avg_dollar_volume is not None:
        df = df[df["avg_dollar_volume"] >= float(min_avg_dollar_volume)]

    df = df.sort_values("avg_dollar_volume", ascending=False)

    if top_n is not None and top_n > 0:
        df = df.head(top_n)

    df.to_csv("tech_universe_C.csv", index=False)
    print(f"Saved: tech_universe_C.csv ({len(df)} tickers)")
    print(df.head(20).to_string(index=False))


if __name__ == "__main__":
    # Example settings:
    # - Keep top 200 by liquidity
    # - Require >= $20M avg daily dollar volume (uncomment if you want)
    main(
        top_n=200,
        # min_avg_dollar_volume=20e6,
        liquidity_lookback_days=30,
    )
