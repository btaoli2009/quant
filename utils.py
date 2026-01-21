import pandas as pd
import numpy as np
from pathlib import Path

def load_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    df = df.dropna(subset=["datetime", "ticker", "close"])
    df = df.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    return df

def add_features(df: pd.DataFrame, vol_windows: list[int], prefix: str) -> pd.DataFrame:
    """
    df columns expected: datetime, ticker, close, volume
    Returns: dataframe with new feature cols prefixed by prefix, per ticker.
    """
    out = df.copy()
    out["log_close"] = np.log(out["close"].astype(float))
    out[f"{prefix}log_ret"] = out.groupby("ticker")["log_close"].diff()
    out[f"{prefix}log_dollar_vol"] = np.log1p(out["close"].astype(float) * out["volume"].fillna(0).astype(float))

    for w in vol_windows:
        mp = min(w, max(3, int(np.ceil(0.8 * w))))
        out[f"{prefix}vol_{w}"] = out.groupby("ticker")[f"{prefix}log_ret"].transform(
            lambda s: s.rolling(w, min_periods=mp).std()
        )

    # Drop helper
    out = out.drop(columns=["log_close"])
    return out

def make_keys_hourly(df_hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Create keys for alignment:
    - day_key: date (UTC) of the hourly bar
    - week_key: ISO week start date (Monday) in UTC
    """
    out = df_hourly.copy()
    # day_key as date
    out["day_key"] = out["datetime"].dt.floor("D")
    # week_key as Monday start
    out["week_key"] = out["datetime"].dt.to_period("W-MON").dt.start_time
    return out

def shift_daily_for_intraday(df_daily_feat: pd.DataFrame) -> pd.DataFrame:
    """
    For intraday prediction at hour t, we can only use daily info up to yesterday.
    Implement by shifting daily features forward by 1 day_key (per ticker).
    That way, when we merge on day_key=today, we get yesterday's features.
    """
    out = df_daily_feat.copy()
    out["day_key"] = out["datetime"].dt.floor("D")
    feature_cols = [c for c in out.columns if c not in {"datetime", "ticker", "day_key", "open", "high", "low", "close", "adj_close", "volume"}]

    # shift features by 1 day per ticker
    out[feature_cols] = out.groupby("ticker")[feature_cols].shift(1)
    return out[["ticker", "day_key"] + feature_cols]

def shift_weekly_for_intraday(df_weekly_feat: pd.DataFrame) -> pd.DataFrame:
    """
    For intraday prediction at hour t, we can only use weekly info up to last week.
    Implement by shifting weekly features forward by 1 week_key (per ticker).
    """
    out = df_weekly_feat.copy()
    out["week_key"] = out["datetime"].dt.to_period("W-MON").dt.start_time
    feature_cols = [c for c in out.columns if c not in {"datetime", "ticker", "week_key", "open", "high", "low", "close", "adj_close", "volume"}]
    out[feature_cols] = out.groupby("ticker")[feature_cols].shift(1)
    return out[["ticker", "week_key"] + feature_cols]

def build_context_features(df_ctx: pd.DataFrame, vol_windows: list[int], prefix: str) -> pd.DataFrame:
    """
    Context ETFs: treat each ticker as a 'context feature', but we want a wide format:
    datetime -> columns like CTX_SPY_log_ret, CTX_QQQ_vol_24, ...
    """
    df = add_features(df_ctx, vol_windows=vol_windows, prefix=f"{prefix}")
    keep = ["datetime", "ticker"] + [c for c in df.columns if c.startswith(prefix)]
    df = df[keep].dropna(subset=["datetime"])
    # pivot to wide
    wide = df.pivot_table(index="datetime", columns="ticker", values=[c for c in df.columns if c.startswith(prefix)])
    # flatten columns
    wide.columns = [f"{c[0]}_{c[1]}" for c in wide.columns.to_flat_index()]
    wide = wide.reset_index()
    wide["datetime"] = pd.to_datetime(wide["datetime"], utc=True)
    return wide.sort_values("datetime").reset_index(drop=True)
