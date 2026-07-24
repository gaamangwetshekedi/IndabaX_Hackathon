"""Data loading and leakage-safe feature engineering.

The raw competition files are never modified. Daily Baltic Dry Index records
are de-duplicated in memory and converted into richer monthly statistics before
being merged with the monthly economic and cross-country series.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


RAW_FILES = {
    "shipping": "01_baltic_dry_index_daily.csv",
    "brent": "02_brent_crude_monthly.csv",
    "policy": "03_botswana_policy_rate.csv",
    "botswana_prices": "04_fao_botswana_prices.csv",
    "cross_country": "05_human_capital_project.csv",
}

TARGET_COLUMN = "bwa_food_inflation"


@dataclass(frozen=True)
class DataBundle:
    """Container returned by :func:`load_and_engineer_data`."""

    monthly: pd.DataFrame
    origin_features: pd.DataFrame
    target: pd.Series
    audit: dict


def _month_start(values: pd.Series) -> pd.Series:
    """Normalise arbitrary dates to month-start timestamps."""

    return pd.to_datetime(values).dt.to_period("M").dt.to_timestamp()


def _monthly_shipping_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Aggregate daily shipping observations without discarding volatility.

    The competition explicitly rewards daily-to-monthly features beyond a
    simple mean. We retain level, dispersion, trend, momentum, extremes and
    within-month timing information. Exact duplicate source rows are removed.
    """

    duplicate_count = int(raw.duplicated().sum())
    daily = raw.drop_duplicates().copy()
    daily["Date"] = pd.to_datetime(daily["Date"])
    daily = daily.sort_values("Date")
    daily["month"] = _month_start(daily["Date"])
    daily["day"] = daily["Date"].dt.day
    daily["daily_return"] = daily.groupby("month")["BDI_Close"].pct_change()
    daily["intraday_spread"] = (
        (daily["BDI_High"] - daily["BDI_Low"])
        / daily["BDI_Close"].replace(0.0, np.nan)
    )

    rows: list[dict] = []
    for month, group in daily.groupby("month", sort=True):
        close = group["BDI_Close"].astype(float)
        returns = group["daily_return"].dropna().astype(float)
        first = float(close.iloc[0])
        last = float(close.iloc[-1])
        early = group.loc[group["day"] <= 15, "BDI_Close"]
        late = group.loc[group["day"] > 15, "BDI_Close"]
        x = np.arange(len(close), dtype=float)
        slope = float(np.polyfit(x, close.to_numpy(), 1)[0]) if len(close) > 1 else 0.0
        mean = float(close.mean())
        std = float(close.std(ddof=0))
        rows.append(
            {
                "Date": month,
                "shipping_mean": mean,
                "shipping_median": float(close.median()),
                "shipping_std": std,
                "shipping_min": float(close.min()),
                "shipping_max": float(close.max()),
                "shipping_range": float(close.max() - close.min()),
                "shipping_cv": std / mean if mean else 0.0,
                "shipping_month_return": last / first - 1.0 if first else 0.0,
                "shipping_trend_per_day": slope,
                "shipping_late_minus_early": (
                    float(late.mean() - early.mean())
                    if len(early) and len(late)
                    else 0.0
                ),
                "shipping_abs_daily_move_mean": float(returns.abs().mean()) if len(returns) else 0.0,
                "shipping_abs_daily_move_max": float(returns.abs().max()) if len(returns) else 0.0,
                "shipping_extreme_up_days": int((returns >= 0.03).sum()),
                "shipping_extreme_down_days": int((returns <= -0.03).sum()),
                "shipping_intraday_spread_mean": float(group["intraday_spread"].fillna(0.0).mean()),
                "shipping_trading_days": int(len(group)),
            }
        )

    monthly = pd.DataFrame(rows).set_index("Date").sort_index()
    audit = {
        "shipping_raw_rows": int(len(raw)),
        "shipping_exact_duplicates_removed": duplicate_count,
        "shipping_clean_rows": int(len(daily)),
        "shipping_months": int(len(monthly)),
    }
    return monthly, audit


def _load_monthly_sources(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load and merge all five supplied datasets onto a monthly calendar."""

    paths = {key: data_dir / name for key, name in RAW_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))

    shipping_raw = pd.read_csv(paths["shipping"])
    shipping, shipping_audit = _monthly_shipping_features(shipping_raw)

    brent = pd.read_csv(paths["brent"])
    brent["Date"] = _month_start(brent["Date"])
    brent = brent.set_index("Date").sort_index()
    brent = brent.rename(columns={"Brent_USD_per_barrel": "brent_usd"})

    policy = pd.read_csv(paths["policy"])
    policy["Date"] = _month_start(policy["Date"])
    policy = policy.set_index("Date").sort_index()

    fao = pd.read_csv(paths["botswana_prices"])
    fao["Date"] = _month_start(fao["Date"])
    fao_wide = fao.pivot(index="Date", columns="Item Code", values="Value").rename(
        columns={23012: "bwa_general_index", 23013: "bwa_food_index", 23014: TARGET_COLUMN}
    )

    cross = pd.read_csv(paths["cross_country"])
    cross["Date"] = _month_start(cross["Date"])
    cross["column"] = (
        cross["REF_AREA"].str.lower()
        + "_"
        + cross["INDICATOR"].str.extract(r"(2301[234])", expand=False)
    )
    cross_wide = cross.pivot(index="Date", columns="column", values="Value")
    cross_wide.columns.name = None
    # Dataset 4 is the authoritative Botswana target. The duplicate Botswana
    # columns in Dataset 5 are excluded from modeling to avoid double counting.
    cross_wide = cross_wide[[c for c in cross_wide.columns if not c.startswith("bwa_")]]

    calendar = pd.date_range("2000-01-01", "2023-12-01", freq="MS")
    monthly = pd.DataFrame(index=calendar)
    for frame in (shipping, brent, policy, fao_wide, cross_wide):
        monthly = monthly.join(frame, how="left")
    monthly.index.name = "Date"

    calculated_target = monthly["bwa_food_index"].pct_change(12, fill_method=None) * 100.0
    identity_error = (monthly[TARGET_COLUMN] - calculated_target).abs().dropna()

    audit = {
        **shipping_audit,
        "monthly_rows": int(len(monthly)),
        "monthly_start": str(monthly.index.min().date()),
        "monthly_end": str(monthly.index.max().date()),
        "target_non_null": int(monthly[TARGET_COLUMN].notna().sum()),
        "target_start": str(monthly[TARGET_COLUMN].dropna().index.min().date()),
        "target_end": str(monthly[TARGET_COLUMN].dropna().index.max().date()),
        "target_identity_max_abs_error": float(identity_error.max()),
        "target_identity_mean_abs_error": float(identity_error.mean()),
        "cross_country_areas": sorted(cross["REF_AREA"].unique().tolist()),
        "cross_country_indicators": sorted(cross["INDICATOR"].unique().tolist()),
        "source_rows": {
            "shipping": int(len(shipping_raw)),
            "brent": int(len(brent)),
            "policy": int(len(policy)),
            "botswana_prices": int(len(fao)),
            "cross_country": int(len(cross)),
        },
    }
    return monthly, audit


def _add_lags(frame: pd.DataFrame, source: str, lags: list[int], out: dict[str, pd.Series]) -> None:
    """Add current/lagged values to an output feature dictionary."""

    for lag in lags:
        label = "now" if lag == 0 else f"lag{lag}"
        out[f"{source}_{label}"] = frame[source].shift(lag)


def build_origin_features(monthly: pd.DataFrame) -> pd.DataFrame:
    """Build features observable at each forecast origin.

    A row dated ``t`` contains only observations dated ``t`` or earlier. A
    direct model maps that row to the next 12 target months, eliminating any
    need to invent 2024 Brent, shipping or policy-rate values.
    """

    features: dict[str, pd.Series] = {}
    month_number = pd.Series(monthly.index.month, index=monthly.index, dtype=float)
    features["origin_month_sin"] = np.sin(2.0 * np.pi * month_number / 12.0)
    features["origin_month_cos"] = np.cos(2.0 * np.pi * month_number / 12.0)

    # The latest published Botswana inflation is known at a forecast origin.
    _add_lags(monthly, TARGET_COLUMN, [0, 1, 2, 3, 6, 12, 18, 24], features)
    for window in [3, 6, 12, 24]:
        rolling = monthly[TARGET_COLUMN].rolling(window)
        features[f"target_roll_mean_{window}"] = rolling.mean()
        features[f"target_roll_std_{window}"] = rolling.std(ddof=0)
    features["target_momentum_3"] = monthly[TARGET_COLUMN] - monthly[TARGET_COLUMN].shift(3)
    features["target_momentum_12"] = monthly[TARGET_COLUMN] - monthly[TARGET_COLUMN].shift(12)

    for source in ["bwa_general_index", "bwa_food_index", "brent_usd", "policy_rate"]:
        _add_lags(monthly, source, [0, 1, 3, 6, 12], features)
        features[f"{source}_change1"] = monthly[source].diff(1)
        features[f"{source}_change3"] = monthly[source].diff(3)
        features[f"{source}_roll6_mean"] = monthly[source].rolling(6).mean()

    shipping_sources = [
        "shipping_mean",
        "shipping_std",
        "shipping_range",
        "shipping_cv",
        "shipping_month_return",
        "shipping_trend_per_day",
        "shipping_late_minus_early",
        "shipping_abs_daily_move_mean",
        "shipping_abs_daily_move_max",
        "shipping_extreme_up_days",
        "shipping_extreme_down_days",
    ]
    for source in shipping_sources:
        _add_lags(monthly, source, [0, 1, 3, 6, 12], features)
    features["shipping_mean_roll3"] = monthly["shipping_mean"].rolling(3).mean()
    features["shipping_mean_roll6"] = monthly["shipping_mean"].rolling(6).mean()

    # Cross-country series are from the separately supplied comparison file.
    for country in ["zaf", "nam", "ken", "zwe"]:
        inflation = f"{country}_23014"
        _add_lags(monthly, inflation, [0, 1, 2, 3, 6, 12], features)
        features[f"{inflation}_roll3"] = monthly[inflation].rolling(3).mean()
        # Price-index growth is retained as a stable regional pressure signal.
        food_index = f"{country}_23013"
        features[f"{food_index}_yoy"] = monthly[food_index].pct_change(12, fill_method=None) * 100.0

    origin = pd.DataFrame(features, index=monthly.index).replace([np.inf, -np.inf], np.nan)
    origin.index.name = "Date"
    return origin


def sequence_channels(monthly: pd.DataFrame) -> pd.DataFrame:
    """Return the compact monthly channel set used by the GRU."""

    channels = pd.DataFrame(index=monthly.index)
    channels["bwa_food_inflation"] = monthly[TARGET_COLUMN]
    channels["bwa_food_index_yoy"] = monthly["bwa_food_index"].pct_change(12, fill_method=None) * 100.0
    channels["shipping_level"] = np.log1p(monthly["shipping_mean"].clip(lower=0.0))
    channels["shipping_volatility"] = monthly["shipping_cv"]
    channels["brent"] = monthly["brent_usd"]
    channels["policy_rate"] = monthly["policy_rate"]
    channels["south_africa_food_inflation"] = monthly["zaf_23014"]
    channels["namibia_food_inflation"] = monthly["nam_23014"]
    channels["kenya_food_inflation"] = monthly["ken_23014"]
    return channels.replace([np.inf, -np.inf], np.nan)


def load_and_engineer_data(data_dir: str | Path) -> DataBundle:
    """Load all sources and return merged data plus model-ready features."""

    monthly, audit = _load_monthly_sources(Path(data_dir))
    origin_features = build_origin_features(monthly)
    target = monthly[TARGET_COLUMN].copy()
    return DataBundle(monthly=monthly, origin_features=origin_features, target=target, audit=audit)
