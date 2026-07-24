# Git Push Legends - IndabaX Botswana 2026

Forecasting Botswana's monthly food-price inflation under global economic
shocks. The target is FAO Item Code `23014` for January-December 2024.

Team members: Tshekedi Gaamangwe, Leatile Mosinyi, Jason Matlhare,
Lone Mompati, Khumo Bontsibokae and Yaone Sekwakwa.

## What this repository contains

- A rich daily-to-monthly Baltic Dry Index transformation.
- Leakage-safe integration of all five supplied datasets.
- A tuned seasonal ARIMA (SARIMA) classical model.
- A compact gated recurrent unit implemented directly in NumPy.
- Explicit ADF and KPSS stationarity diagnostics.
- Chronological rolling-origin backtesting for 2021, 2022 and 2023.
- A single winner selected by aggregate out-of-sample RMSE.
- Reproducible charts, reports, linkage analysis and submission validation.

The project intentionally avoids heavyweight deep-learning frameworks. The GRU
includes update, reset and candidate gates; full backpropagation through time;
Adam optimisation; input dropout; L2 regularisation; gradient clipping; and
temporal early stopping. This keeps the model auditable and portable.

## Reproduce everything

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
python -m unittest discover -v
```

The five competition CSV files must remain in the repository root with their
original names. Raw source files are never modified.

## Outputs

| Deliverable | Path |
|---|---|
| Best-model predictions | `outputs/predictions.csv` |
| Feature Engineering Report | `output/pdf/feature_engineering_report.pdf` |
| Model Comparison Report | `output/pdf/model_comparison_report.pdf` |
| HCP Linkage Memo | `output/pdf/hcp_linkage_memo.pdf` |
| HCP Visualisations | `output/pdf/hcp_visualisations.pdf` |
| Metrics and forecast details | `outputs/results.json` |
| ADF/KPSS results | `outputs/stationarity_tests.json` |
| SARIMA tuning folds | `outputs/sarima_tuning_scores.csv` |
| Pitch pattern check | `outputs/pitch_notes.md` |
| HCP statistical evidence | `outputs/hcp_linkage_results.json` |
| HCP forward-pressure summary | `outputs/hcp_forward_pressure_summary.json` |
| Integrity manifest | `outputs/submission_manifest.json` |
| Reproducible code ZIP | `output/Git_Push_Legends_Code_Repository.zip` |
| Convenience submission pack | `output/Git_Push_Legends_Phase1_Submission_Pack.zip` |

The predictions file has exactly two columns and twelve rows:

```text
year_month,forecast
2024-01,...
...
2024-12,...
```

## Validation strategy

Random splitting is not used. Eight compact SARIMA specifications are compared
on forecasts for 2018-2020. Final model comparison uses separate rolling
origins:

- December 2020 -> January-December 2021
- December 2021 -> January-December 2022
- December 2022 -> January-December 2023

At each origin, SARIMA is fitted only through that date. A GRU training sequence
is eligible only if its entire future 12-month label window is already observed,
and its scalers are fitted inside the training fold. Both models are refitted
through December 2023 and forecast all of 2024. No actual or assumed 2024
shipping, Brent or policy-rate values are used.

## Model comparison

SARIMA models the target's autoregressive, differenced, moving-average and
12-month seasonal structure. Its specification is chosen only on the 2018-2020
tuning origins. The GRU consumes 24 monthly observations of nine compact
cross-dataset channels and directly produces 12 forecasts. Seasonal naive is
reported as a diagnostic benchmark but is not substituted for either required
model.

The final CSV comes from whichever required model has lower aggregate 2021-2023
RMSE. The two forecasts are never averaged.

ADF and KPSS tests are reported for the target level, first difference and
12-month seasonal difference. The level tests provide mixed evidence, while the
first-differenced series is supported as stationary; the selected SARIMA
therefore uses one non-seasonal difference.

## Human-capital limitation

The supplied `05_human_capital_project.csv` contains FAO consumer-price
indicators for five countries. It does not contain direct education, health,
nutrition or World Bank Human Capital Index outcomes. We therefore test two
distinct supplied price-pressure indicators: regional food inflation as a food
affordability and diet-quality channel, and regional general CPI inflation as a
broader household purchasing-power channel. Separate lagged regressions and
two-lag predictive tests report coefficients and p-values for both.

The forward section uses our Phase 1 forecast to quantify the projected 2024
exposure: annual-average and peak food inflation, the number of months at or
above 10%, and the first-half to second-half easing. We describe these as
household-affordability pressure proxies and do not invent causal health,
education or nutrition effects that the supplied data cannot identify.

## Pitch note: June-October movement

The generated `outputs/pitch_notes.md` compares the 2024 June-October forecast
movement with the corresponding observed 2023 pattern and provides a one-line,
evidence-based explanation for presentation day.
