from __future__ import annotations

import time
from pathlib import Path
import pandas as pd
import yfinance as yf


# =========================
# Config
# =========================
UNIVERSE_CSV = "tech_universe_C.csv"
OUT_DIR = Path("data_yf")

# Yahoo limits (safe defaults)
CHUNK_SIZE = 25
PAUSE_SECONDS = 2.0

# History windows
HOURLY_PERIOD = "730d"   # ~2 years (Yahoo limit)
DAILY_PERIOD  = "10y"
WEEKLY_PERIOD = "10y"


# =========================
# Helpers
# =========================
def load_universe(csv_path: str) -> list[str]:
    df = pd.read_csv(csv_path)
    tickers = (
        df["ticker"]
        .dropna()
        .astype(str)
        .str.upper()
        .str.replace(".", "-", regex=False)
        .unique()
        .tolist()
    )
    return tickers


def yf_download_long(
    tickers: list[str],
    interval: str,
    period: str,
) -> pd.DataFrame:
    """
    Download data in chunks and return long-format dataframe:
    columns = [datetime, ticker, open, high, low, close, adj_close, volume]
    """
    all_parts = []

    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        print(f"[{interval}] downloading {i}..{i+len(chunk)-1}")

        df = yf.download(
            tickers=" ".join(chunk),
            interval=interval,
            period=period,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )

        if df is None or df.empty:
            time.sleep(PAUSE_SECONDS)
            continue

        # Multi-ticker case
        if isinstance(df.columns, pd.MultiIndex):
            df = (
                df.stack(level=0)
                .reset_index()
                .rename(columns={"level_1": "ticker"})
            )
        else:
            # Single ticker fallback
            df = df.reset_index()
            df["ticker"] = chunk[0]

        # Normalize column names
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]
        df = df.rename(columns={"date": "datetime", "datetime": "datetime"})

        keep = ["datetime", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
        for c in keep:
            if c not in df.columns:
                df[c] = pd.NA

        all_parts.append(df[keep])

        time.sleep(PAUSE_SECONDS)

    if not all_parts:
        return pd.DataFrame(columns=["datetime","ticker","open","high","low","close","adj_close","volume"])

    out = pd.concat(all_parts, ignore_index=True)
    out = out.dropna(subset=["datetime", "ticker"])
    out = out.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    return out


# =========================
# Main
# =========================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_universe(UNIVERSE_CSV)
    print(f"Loaded {len(tickers)} tickers")

    # ---- Hourly ----
    hourly = yf_download_long(
        tickers=tickers,
        interval="1h",
        period=HOURLY_PERIOD,
    )
    hourly.to_parquet(OUT_DIR / "hourly.parquet", index=False)
    print(f"Saved hourly: {len(hourly):,} rows")

    # ---- Daily ----
    daily = yf_download_long(
        tickers=tickers,
        interval="1d",
        period=DAILY_PERIOD,
    )
    daily.to_parquet(OUT_DIR / "daily.parquet", index=False)
    print(f"Saved daily: {len(daily):,} rows")

    # ---- Weekly ----
    weekly = yf_download_long(
        tickers=tickers,
        interval="1wk",
        period=WEEKLY_PERIOD,
    )
    weekly.to_parquet(OUT_DIR / "weekly.parquet", index=False)
    print(f"Saved weekly: {len(weekly):,} rows")


if __name__ == "__main__":
    main()
