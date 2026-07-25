"""
Generate the final Phase 1 predictions CSV using the selected classical model
(SARIMA -- the best performer in outputs/model_comparison.json).

The hackathon guide requires a 12-month-ahead forecast (Jan-Dec 2024), and all
5 source datasets deliberately end December 2023 so this is a genuine forecast,
not a contemporaneous regression. This script forecasts the 12 months
immediately following the last available data point and labels the output
Jan-2024 through Dec-2024 accordingly.
"""
import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pipeline import build_dataset

HORIZON = 12
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    df = build_dataset()
    df = df.dropna(subset=["food_inflation"]).reset_index(drop=True)
    y = df["food_inflation"].values
    last_date = df["Date"].max()

    model = SARIMAX(y, order=(1, 1, 1), seasonal_order=(1, 0, 1, 12),
                     enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = model.get_forecast(HORIZON)
    preds = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=HORIZON, freq="MS")
    out = pd.DataFrame({
        "year_month": future_dates.strftime("%Y-%m"),
        "forecast": np.round(preds, 4),
    })
    out.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)

    ci_out = pd.DataFrame({
        "year_month": future_dates.strftime("%Y-%m"),
        "forecast": np.round(preds, 4),
        "lower_95": np.round(ci[:, 0], 4),
        "upper_95": np.round(ci[:, 1], 4),
    })
    ci_out.to_csv(os.path.join(OUTPUT_DIR, "predictions_with_ci.csv"), index=False)

    print("Forecasting", future_dates[0].strftime("%Y-%m"), "to", future_dates[-1].strftime("%Y-%m"),
          "(12 months immediately after the last available data point, Dec-2023).")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
