from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from utils import load_parquet

DATA_DIR = Path("data_yf")
OUT_DIR = Path("data_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Input files from your downloader
HOURLY_PATH = DATA_DIR / "hourly.parquet"
DAILY_PATH  = DATA_DIR / "daily.parquet"
WEEKLY_PATH = DATA_DIR / "weekly.parquet"

# Optional context ETFs (set to None to disable)
CTX_HOURLY_PATH = DATA_DIR / "context_hourly.parquet"
CTX_DAILY_PATH  = DATA_DIR / "context_daily.parquet"
CTX_WEEKLY_PATH = DATA_DIR / "context_weekly.parquet"
USE_CONTEXT = CTX_HOURLY_PATH.exists() and CTX_DAILY_PATH.exists() and CTX_WEEKLY_PATH.exists()

# Sample config
HORIZON_HOURS = 1             # predict next-hour return
WIN_HOURLY = 240              # ~10 trading days
WIN_DAILY  = 60               # ~3 months
WIN_WEEKLY = 52               # ~1 year

# Feature windows
HOURLY_VOL_WINS = [24, 120]
DAILY_VOL_WINS  = [5, 20]
WEEKLY_VOL_WINS = [4, 13]

# Subsample (optional): build fewer samples for speed, e.g. every 3 hours
SAMPLE_EVERY_N_HOURS = 1


# -----------------------
# Helpers
# -----------------------

















# -----------------------
# Main pipeline
# -----------------------
def main():
    print("Loading raw data...")
    hourly = load_parquet(HOURLY_PATH)
    daily  = load_parquet(DAILY_PATH)
    weekly = load_parquet(WEEKLY_PATH)

    print("Adding features...")
    hourly_f = add_features(hourly, HOURLY_VOL_WINS, prefix="h_")
    daily_f  = add_features(daily,  DAILY_VOL_WINS,  prefix="d_")
    weekly_f = add_features(weekly, WEEKLY_VOL_WINS, prefix="w_")

    # Keys for hourly
    hourly_fk = make_keys_hourly(hourly_f)

    # Shift daily/weekly for intraday alignment (no leakage)
    daily_shifted  = shift_daily_for_intraday(daily_f)
    weekly_shifted = shift_weekly_for_intraday(weekly_f)

    # Merge aligned daily/weekly features onto hourly rows
    print("Aligning daily/weekly features onto hourly timeline...")
    base = hourly_fk.merge(daily_shifted, on=["ticker", "day_key"], how="left")
    base = base.merge(weekly_shifted, on=["ticker", "week_key"], how="left")

    # Optional: context ETFs
    if USE_CONTEXT:
        print("Building context features...")
        ctx_h = load_parquet(CTX_HOURLY_PATH)
        ctx_d = load_parquet(CTX_DAILY_PATH)
        ctx_w = load_parquet(CTX_WEEKLY_PATH)

        ctx_h_wide = build_context_features(ctx_h, vol_windows=HOURLY_VOL_WINS, prefix="ctx_h_")
        ctx_d_wide = build_context_features(ctx_d, vol_windows=DAILY_VOL_WINS,  prefix="ctx_d_")
        ctx_w_wide = build_context_features(ctx_w, vol_windows=WEEKLY_VOL_WINS, prefix="ctx_w_")

        base = base.merge(ctx_h_wide, on="datetime", how="left")
        # For daily/weekly context, we also need no-leakage shifts:
        # easiest: merge on datetime after shifting by 1 bar in those wide tables
        ctx_d_wide_shift = ctx_d_wide.sort_values("datetime").copy()
        ctx_w_wide_shift = ctx_w_wide.sort_values("datetime").copy()
        # shift all context feature columns by 1 row (global) since these are index-level series
        d_cols = [c for c in ctx_d_wide_shift.columns if c != "datetime"]
        w_cols = [c for c in ctx_w_wide_shift.columns if c != "datetime"]
        ctx_d_wide_shift[d_cols] = ctx_d_wide_shift[d_cols].shift(1)
        ctx_w_wide_shift[w_cols] = ctx_w_wide_shift[w_cols].shift(1)

        # Align daily context by day_key (merge on day_key with shifted daily context day)
        ctx_d_wide_shift["day_key"] = ctx_d_wide_shift["datetime"].dt.floor("D")
        base = base.merge(
            ctx_d_wide_shift.drop(columns=["datetime"]),
            on="day_key",
            how="left",
            suffixes=("", "_dup_d")
        )

        # Align weekly context by week_key
        ctx_w_wide_shift["week_key"] = ctx_w_wide_shift["datetime"].dt.to_period("W-MON").dt.start_time
        base = base.merge(
            ctx_w_wide_shift.drop(columns=["datetime"]),
            on="week_key",
            how="left",
            suffixes=("", "_dup_w")
        )

        # Clean any accidental dup columns
        base = base[[c for c in base.columns if not c.endswith("_dup_d") and not c.endswith("_dup_w")]]

    # Build prediction target: next-hour log return
    print("Building targets...")
    base = base.sort_values(["ticker", "datetime"]).reset_index(drop=True)
    base["y_next_h_ret"] = base.groupby("ticker")["h_log_ret"].shift(-HORIZON_HOURS)

    # Optional: direction label
    base["y_next_h_up"] = (base["y_next_h_ret"] > 0).astype("int8")

    # Optional subsampling (every N hours per ticker)
    if SAMPLE_EVERY_N_HOURS > 1:
        base["row_in_ticker"] = base.groupby("ticker").cumcount()
        base = base[base["row_in_ticker"] % SAMPLE_EVERY_N_HOURS == 0].drop(columns=["row_in_ticker"])

    # Drop rows where target is missing
    base = base.dropna(subset=["y_next_h_ret"])

    # Save the aligned feature table (still "flat")
    aligned_path = OUT_DIR / "aligned_hourly_with_daily_weekly.parquet"
    base.to_parquet(aligned_path, index=False)
    print(f"Saved aligned feature table: {aligned_path}  rows={len(base):,}")

    # Next step would be: windowing into tensors for the TCN.
    # We keep it flat for now so you can inspect/verify alignment and missingness.

    print("Done.")


if __name__ == "__main__":
    main()
