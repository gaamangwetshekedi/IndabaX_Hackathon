"""Cross-country linkage analysis for the supplied comparison dataset.

The file named ``human_capital_project`` contains price indicators rather than
direct health, education or nutrition outcomes. The analysis therefore treats
regional food-price series as human-capital pressure proxies and states that
limitation explicitly instead of inventing unavailable outcomes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _continued_beta(a: float, b: float, x: float) -> float:
    """Numerical Recipes continued fraction for the incomplete beta function."""

    maximum_iterations = 200
    epsilon = 3e-12
    floor = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < floor:
        d = floor
    d = 1.0 / d
    result = d
    for iteration in range(1, maximum_iterations + 1):
        m2 = 2 * iteration
        aa = iteration * (b - iteration) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        result *= d * c
        aa = -(a + iteration) * (qab + iteration) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < floor:
            d = floor
        c = 1.0 + aa / c
        if abs(c) < floor:
            c = floor
        d = 1.0 / d
        delta = d * c
        result *= delta
        if abs(delta - 1.0) < epsilon:
            break
    return result


def _regularized_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta function using only the standard library."""

    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_term = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    front = math.exp(log_term + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _continued_beta(a, b, x) / a
    return 1.0 - front * _continued_beta(b, a, 1.0 - x) / b


def _student_two_sided_p(t_statistic: float, degrees_freedom: int) -> float:
    x = degrees_freedom / (degrees_freedom + t_statistic**2)
    return float(np.clip(_regularized_beta(x, degrees_freedom / 2.0, 0.5), 0.0, 1.0))


def _f_survival(f_statistic: float, numerator_df: int, denominator_df: int) -> float:
    if f_statistic <= 0.0:
        return 1.0
    x = numerator_df * f_statistic / (numerator_df * f_statistic + denominator_df)
    cdf = _regularized_beta(x, numerator_df / 2.0, denominator_df / 2.0)
    return float(np.clip(1.0 - cdf, 0.0, 1.0))


def _ols(y: np.ndarray, x: np.ndarray, names: list[str]) -> dict:
    """Fit OLS and return coefficients with finite-sample t tests."""

    design = np.column_stack([np.ones(len(x)), x])
    labels = ["intercept", *names]
    inverse = np.linalg.pinv(design.T @ design)
    coefficients = inverse @ design.T @ y
    residuals = y - design @ coefficients
    degrees_freedom = len(y) - design.shape[1]
    sigma2 = float(residuals @ residuals / degrees_freedom)
    standard_errors = np.sqrt(np.maximum(np.diag(inverse) * sigma2, 1e-18))
    t_values = coefficients / standard_errors
    p_values = [_student_two_sided_p(float(t), degrees_freedom) for t in t_values]
    total = float(np.sum((y - np.mean(y)) ** 2))
    rss = float(residuals @ residuals)
    r_squared = 1.0 - rss / total if total else 0.0
    return {
        "n": int(len(y)),
        "r_squared": float(r_squared),
        "rss": rss,
        "coefficients": [
            {
                "term": label,
                "coefficient": float(coefficient),
                "std_error": float(error),
                "t_stat": float(t_value),
                "p_value": float(p_value),
            }
            for label, coefficient, error, t_value, p_value in zip(
                labels, coefficients, standard_errors, t_values, p_values
            )
        ],
    }


def _granger_test(target: pd.Series, driver: pd.Series, lags: int = 2) -> dict:
    """Test whether driver lags add predictive information beyond target lags."""

    frame = pd.DataFrame({"target": target, "driver": driver})
    for lag in range(1, lags + 1):
        frame[f"target_lag{lag}"] = frame["target"].shift(lag)
        frame[f"driver_lag{lag}"] = frame["driver"].shift(lag)
    frame = frame.dropna()
    y = frame["target"].to_numpy(dtype=float)
    restricted_names = [f"target_lag{lag}" for lag in range(1, lags + 1)]
    unrestricted_names = restricted_names + [f"driver_lag{lag}" for lag in range(1, lags + 1)]
    restricted = _ols(y, frame[restricted_names].to_numpy(dtype=float), restricted_names)
    unrestricted = _ols(y, frame[unrestricted_names].to_numpy(dtype=float), unrestricted_names)
    restrictions = lags
    denominator_df = len(y) - (1 + len(unrestricted_names))
    numerator = max(0.0, (restricted["rss"] - unrestricted["rss"]) / restrictions)
    denominator = unrestricted["rss"] / denominator_df
    f_statistic = numerator / denominator if denominator > 0.0 else 0.0
    return {
        "lags": int(lags),
        "n": int(len(y)),
        "f_stat": float(f_statistic),
        "p_value": _f_survival(float(f_statistic), restrictions, denominator_df),
    }


def _indicator_relationship(
    target: pd.Series,
    indicator: pd.Series,
    *,
    label: str,
    rationale: str,
) -> dict:
    """Estimate one indicator's lagged association and predictive contribution."""

    frame = pd.DataFrame(
        {
            "target": target,
            "target_lag1": target.shift(1),
            "indicator_lag1": indicator.shift(1),
        }
    ).dropna()
    regression = _ols(
        frame["target"].to_numpy(dtype=float),
        frame[["target_lag1", "indicator_lag1"]].to_numpy(dtype=float),
        ["botswana_lag1", "indicator_lag1"],
    )
    indicator_coefficient = next(
        row for row in regression["coefficients"] if row["term"] == "indicator_lag1"
    )
    return {
        "label": label,
        "rationale": rationale,
        "definition": "Mean of South Africa and Namibia; one-month lag for regression",
        "regression_n": regression["n"],
        "regression_r_squared": regression["r_squared"],
        "lag1_coefficient": indicator_coefficient["coefficient"],
        "lag1_std_error": indicator_coefficient["std_error"],
        "lag1_p_value": indicator_coefficient["p_value"],
        "granger": _granger_test(target, indicator, lags=2),
    }


def analyse_linkages(monthly: pd.DataFrame) -> dict:
    """Quantify two regional price-pressure channels relevant to households."""

    frame = pd.DataFrame(index=monthly.index)
    frame["botswana"] = monthly["bwa_food_inflation"]
    frame["botswana_lag1"] = frame["botswana"].shift(1)
    frame["south_africa_lag1"] = monthly["zaf_23014"].shift(1)
    frame["namibia_lag1"] = monthly["nam_23014"].shift(1)
    regression_frame = frame.dropna()
    regression = _ols(
        regression_frame["botswana"].to_numpy(dtype=float),
        regression_frame[["botswana_lag1", "south_africa_lag1", "namibia_lag1"]].to_numpy(dtype=float),
        ["botswana_lag1", "south_africa_lag1", "namibia_lag1"],
    )

    correlations = {}
    for country, column in {
        "South Africa": "zaf_23014",
        "Namibia": "nam_23014",
        "Kenya": "ken_23014",
        "Zimbabwe": "zwe_23014",
    }.items():
        aligned = pd.concat([monthly["bwa_food_inflation"], monthly[column]], axis=1).dropna()
        correlations[country] = float(aligned.iloc[:, 0].corr(aligned.iloc[:, 1]))

    granger = {
        "South Africa": _granger_test(monthly["bwa_food_inflation"], monthly["zaf_23014"], lags=2),
        "Namibia": _granger_test(monthly["bwa_food_inflation"], monthly["nam_23014"], lags=2),
    }
    regional_food_inflation = monthly[["zaf_23014", "nam_23014"]].mean(axis=1)
    regional_general_inflation = (
        monthly[["zaf_23012", "nam_23012"]]
        .pct_change(12, fill_method=None)
        .mean(axis=1)
        * 100.0
    )
    indicator_tests = {
        "regional_food_inflation": _indicator_relationship(
            monthly["bwa_food_inflation"],
            regional_food_inflation,
            label="Regional food inflation",
            rationale=(
                "Captures direct pressure on household food affordability and diet quality."
            ),
        ),
        "regional_general_inflation": _indicator_relationship(
            monthly["bwa_food_inflation"],
            regional_general_inflation,
            label="Regional general CPI inflation",
            rationale=(
                "Captures broader cost-of-living pressure on real household purchasing power."
            ),
        ),
    }
    return {
        "regression": regression,
        "correlations": correlations,
        "granger": granger,
        "indicator_tests": indicator_tests,
        "limitation": (
            "The supplied Human Capital Project file contains FAO consumer-price "
            "indicators, not direct health, education or nutrition outcomes. Results "
            "therefore describe regional price-pressure linkages, not causal human-capital effects."
        ),
    }


def regional_projection(monthly: pd.DataFrame, botswana_forecast: np.ndarray) -> pd.DataFrame:
    """Create a transparent 2024 pressure-proxy projection for visualisation."""

    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    result = pd.DataFrame(index=dates)
    result["Botswana forecast"] = np.asarray(botswana_forecast, dtype=float)
    for country, column in {
        "South Africa seasonal baseline": "zaf_23014",
        "Namibia seasonal baseline": "nam_23014",
    }.items():
        result[country] = monthly.loc["2023-01-01":"2023-12-01", column].to_numpy(dtype=float)
    result.index.name = "Date"
    return result


def forward_pressure_summary(projection: pd.DataFrame) -> dict:
    """Summarise the forecast as a transparent household-affordability exposure proxy."""

    forecast = projection["Botswana forecast"].astype(float)
    peak_month = pd.Timestamp(forecast.idxmax())
    trough_month = pd.Timestamp(forecast.idxmin())
    comparison_columns = [
        "South Africa seasonal baseline",
        "Namibia seasonal baseline",
    ]
    regional_baseline = projection[comparison_columns].mean(axis=1)
    first_half_average = float(forecast.iloc[:6].mean())
    second_half_average = float(forecast.iloc[6:].mean())
    return {
        "annual_average_food_inflation_yoy": float(forecast.mean()),
        "peak_month": peak_month.strftime("%Y-%m"),
        "peak_food_inflation_yoy": float(forecast.max()),
        "lowest_month": trough_month.strftime("%Y-%m"),
        "lowest_food_inflation_yoy": float(forecast.min()),
        "months_at_or_above_10_percent": int((forecast >= 10.0).sum()),
        "first_half_average_yoy": first_half_average,
        "second_half_average_yoy": second_half_average,
        "half_year_easing_percentage_points": first_half_average - second_half_average,
        "december_food_inflation_yoy": float(forecast.iloc[-1]),
        "regional_seasonal_baseline_average_yoy": float(regional_baseline.mean()),
        "interpretation": (
            "These forecast statistics measure exposure to food-affordability pressure. "
            "They do not estimate a causal change in health, education or nutrition outcomes."
        ),
    }
