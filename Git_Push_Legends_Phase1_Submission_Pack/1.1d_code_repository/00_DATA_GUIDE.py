# %% [markdown]
# # 📊 Hackathon Data Guide
# ## Forecasting Botswana's Human Capital Under Global Economic Shocks
# ### Deep Learning IndabaX — AI & Data Science Hackathon
#
# **Team size:** 3–5 participants per team.
#
# **What you must do:** Forecast Botswana's monthly food price inflation
# (`FAO_CP_23014`) for **12 consecutive months — January 2024 through
# December 2024**.
#
# **What you're given:** 5 separate datasets at different granularities.
# It is YOUR job to merge, transform, and engineer features from these.
#
# **⚠️ REQUIRED:** You must submit BOTH a classical baseline model AND a
# deep learning model. See Section 6 below — this is non-negotiable and
# scored separately from accuracy.
#
# ---
#
# ## Why 5 Separate Files Instead of One Merged CSV?
#
# We deliberately provide the raw datasets separately because merging them
# IS the feature engineering challenge. A naive monthly average loses signal.
# A strong submission will extract features from daily data that monthly
# aggregation cannot capture — and that's exactly what the rubric rewards.
#
# **Your 5 datasets:** All datasets end at December 2023. You forecast
# 12 months forward into 2024 using only data through the end of 2023.
#
# | # | File | Granularity | Rows | Date Range |
# |---|------|-------------|------|------------|
# | 1 | `01_baltic_dry_index_daily.csv` | **Daily** | ~5,990 | Jan 2000 – Dec 2023 |
# | 2 | `02_brent_crude_monthly.csv` | Monthly | 288 | Jan 2000 – Dec 2023 |
# | 3 | `03_botswana_policy_rate.csv` | Monthly | 288 | Jan 2000 – Dec 2023 |
# | 4 | `04_fao_botswana_prices.csv` | Monthly | 852 | Jan 2000 – Dec 2023 |
# | 5 | `05_human_capital_project.csv` | Monthly | 4,320 | Jan 2000 – Dec 2023 |
#
# This is a pure forecasting setup. You have 24 years of training history
# across all sources. You then forecast 12 months ahead (Jan 2024 – Dec
# 2024) with no knowledge of what actually happened to any variable during
# the forecast window. This is how real-world forecasting works.

# %%
import pandas as pd
import numpy as np

# ── Update these paths to match your folder structure ──
DATA_DIR = "."  # folder containing the 5 CSV files

# %% [markdown]
# ---
# # 1. Dataset Descriptions

# %% [markdown]
# ## Dataset 1: Baltic Dry Index (DAILY)
#
# **File:** `01_baltic_dry_index_daily.csv`
#
# The Baltic Dry Index measures the cost of shipping raw materials (iron ore,
# coal, grain) by sea. It's a leading indicator of global trade activity and
# supply chain stress.
#
# | Column | Description |
# |--------|-------------|
# | `Date` | Trading day (YYYY-MM-DD) |
# | `BDI_Close` | Closing index value for that day |
# | `BDI_High` | Intraday high |
# | `BDI_Low` | Intraday low |
#
# **Why it matters for Botswana:** Botswana imports most manufactured goods
# and fuel by sea. A spike in shipping costs takes 2–4 months to feed into
# local consumer prices.
#
# ⚠️ **This is DAILY data (~6,500 rows).** You must decide how to aggregate
# it to monthly frequency for merging. A simple monthly average is the
# baseline — but it throws away valuable information. See Section 3.

# %%
bdi = pd.read_csv(f"{DATA_DIR}/01_baltic_dry_index_daily.csv", parse_dates=["Date"])
print(f"BDI: {bdi.shape}")
print(f"Date range: {bdi['Date'].min().date()} to {bdi['Date'].max().date()}")
print(bdi.head())

# %% [markdown]
# ## Dataset 2: Brent Crude Oil Prices (Monthly)
#
# **File:** `02_brent_crude_monthly.csv`
#
# International benchmark oil price. Source: US Energy Information
# Administration (EIA) via FRED.
#
# | Column | Description |
# |--------|-------------|
# | `Date` | Month (15th = monthly average convention) |
# | `Brent_USD_per_barrel` | Average price in USD per barrel for that month |
#
# **Why it matters:** Oil prices drive transport and energy costs in
# Botswana. A Brent spike cascades into food prices within 1–3 months.

# %%
brent = pd.read_csv(f"{DATA_DIR}/02_brent_crude_monthly.csv", parse_dates=["Date"])
print(f"Brent: {brent.shape}")
print(brent.head())

# %% [markdown]
# ## Dataset 3: Botswana Policy Rate (Monthly)
#
# **File:** `03_botswana_policy_rate.csv`
#
# The Bank of Botswana benchmark interest rate. The central bank's primary
# tool for controlling inflation.
#
# | Column | Description |
# |--------|-------------|
# | `Date` | First of each month (YYYY-MM-DD) |
# | `policy_rate` | Interest rate in percent (%) |
#
# **Why it matters:** Rate hikes dampen inflation but with a lag (typically
# 6–12 months). This is a policy response variable.

# %%
pr = pd.read_csv(f"{DATA_DIR}/03_botswana_policy_rate.csv", parse_dates=["Date"])
print(f"Policy Rate: {pr.shape}")
print(pr.head())

# %% [markdown]
# ## Dataset 4: FAO Botswana Price Indices (Monthly)
#
# **File:** `04_fao_botswana_prices.csv`
#
# Consumer price indices and food price inflation for Botswana from the
# UN Food and Agriculture Organization (FAO).
#
# **Three indicators in this file:**
#
# | Item Code | Indicator | Role |
# |-----------|-----------|------|
# | 23012 | Consumer Prices, General Indices (2015 = 100) | Feature |
# | 23013 | Consumer Prices, Food Indices (2015 = 100) | Feature |
# | **23014** | **Food Price Inflation (% YoY)** | **⭐ TARGET VARIABLE** |
#
# ⚠️ **Item 23014 is what you must forecast.** Year-on-year percentage
# change in Botswana's food consumer price index.

# %%
fao = pd.read_csv(f"{DATA_DIR}/04_fao_botswana_prices.csv", parse_dates=["Date"])
print(f"FAO: {fao.shape}")
print(f"Indicators: {fao[['Item Code','Item']].drop_duplicates().to_string(index=False)}")
print(f"\nTarget variable (23014) last 5 values:")
target = fao[fao['Item Code'] == 23014].tail(5)
print(target[['Date','Value']].to_string(index=False))

# %% [markdown]
# ## Dataset 5: Human Capital Project — Cross-Country (Monthly)
#
# **File:** `05_human_capital_project.csv`
#
# The same FAO indicators, but for Botswana AND 4 comparison countries.
#
# **Countries included:**
# - BWA — Botswana (same data as Dataset 4)
# - ZAF — South Africa (Botswana's largest trading partner)
# - NAM — Namibia (shares customs union with Botswana)
# - KEN — Kenya (East African comparator)
# - ZWE — Zimbabwe (neighbouring economy)

# %%
hcp = pd.read_csv(f"{DATA_DIR}/05_human_capital_project.csv", parse_dates=["Date"])
print(f"HCP: {hcp.shape}")
print(f"Countries: {hcp['REF_AREA'].unique().tolist()}")

# %% [markdown]
# ---
# # 2. Basic Merge — The Minimum Viable Approach
#
# This section shows the simplest way to get all 5 datasets into one table.
# **This is the baseline. It will work, but it will not score well on
# feature engineering.**

# %%
def add_ym(df):
    df = df.copy()
    df["year_month"] = df["Date"].dt.to_period("M").astype(str)
    return df

# NAIVE: just monthly mean of BDI
bdi_naive = add_ym(bdi).groupby("year_month").agg(
    BDI_mean=("BDI_Close", "mean")
).reset_index()
bdi_naive["BDI_mean"] = bdi_naive["BDI_mean"].round(2)

brent_m = add_ym(brent)[["year_month", "Brent_USD_per_barrel"]]
pr_m = add_ym(pr)[["year_month", "policy_rate"]]

# Pivot FAO long → wide
fao_m = add_ym(fao)
fao_m["col"] = "FAO_" + fao_m["Item Code"].astype(str)
fao_wide = fao_m.pivot_table(index="year_month", columns="col", values="Value", aggfunc="first").reset_index()
fao_wide.columns.name = None

# Pivot HCP long → wide
hcp_m = add_ym(hcp)
hcp_m["col"] = hcp_m["REF_AREA"] + "_" + hcp_m["INDICATOR"]
hcp_wide = hcp_m.pivot_table(index="year_month", columns="col", values="Value", aggfunc="first").reset_index()
hcp_wide.columns.name = None

# Merge all
merged = bdi_naive.copy()
for df in [brent_m, pr_m, fao_wide, hcp_wide]:
    merged = merged.merge(df, on="year_month", how="outer")
merged = merged.sort_values("year_month").reset_index(drop=True)

print(f"Basic merged: {merged.shape}")

# %% [markdown]
# ---
# # 6. ⭐ The Two-Model Requirement
#
# **You MUST submit BOTH a classical baseline AND a deep learning model.**
#
# This is a hard requirement, not a suggestion. Submissions missing one
# or the other are penalised heavily in the Model Comparison rubric.
#
# ## Why This Requirement Exists
#
# This is a Deep Learning IndabaX event — so DL exposure matters. But on
# ~300 monthly observations, deep learning will probably lose to classical
# methods on pure accuracy. That's not a failure; that's a real ML lesson.
#
# By requiring both, we let you:
# - Learn deep learning hands-on (the IndabaX mission)
# - Understand when DL helps vs hurts (the real-world lesson)
# - Get credit for honest analysis even if your DL model loses
#
# ## What Counts as Each Model Type
#
# **Acceptable classical baselines** (pick one):
# - ARIMA / SARIMA (statsmodels)
# - ETS / Exponential Smoothing (statsmodels)
# - Gradient Boosted Trees on lagged features (XGBoost, LightGBM, CatBoost)
# - Linear/Ridge regression with lag features (sklearn)
#
# **Acceptable deep learning approaches** (pick one):
# - LSTM or GRU (PyTorch / TensorFlow)
# - Temporal Fusion Transformer (pytorch-forecasting)
# - Neural ODE (torchdyn or torchdiffeq)
# - Transformer-based forecasters (Informer, Autoformer, PatchTST, etc.)
#
# ## What Goes on the Leaderboard
#
# You submit ONE Predictions CSV containing your **best forecast of
# Botswana's food price inflation** (`FAO_CP_23014`, Item Code 23014 from
# Dataset 4) — that is, the year-on-year % change in Botswana's food
# consumer price index, predicted for each of the **12 months from
# January 2024 through December 2024**.
#
# You choose which of your two models produces the forecast you submit
# (whichever you believe is most accurate). You may submit either the
# classical baseline's forecast OR the deep learning model's forecast —
# but not both, and not an average of the two. One CSV, one model's output.
#
# The leaderboard ranks by RMSE between your 12 predicted values and the
# 12 actual values for Jan 2024 – Dec 2024 (withheld from you, held on
# the platform's back end).
#
# Your Model Comparison Report (Deliverable 1.1c) must show validation
# metrics for BOTH models so judges can verify your "best" claim was
# made honestly. Submitting your worse model and pretending it was your
# best — when the report shows the other one performed better on
# validation — will be flagged as a dishonest comparison and penalised.
#
# ## Practical Advice for Deep Learning on Small Data
#
# ~290 monthly observations is small for DL, and you are forecasting 12
# months ahead — a full year. Take these seriously:
#
# 1. **Use a small architecture.** A 2-layer LSTM with 32–64 hidden units
#    is reasonable. Don't reach for a 12-layer Transformer.
#
# 2. **Regularise heavily.** Dropout (0.2–0.4), weight decay, early stopping.
#    Without these you'll overfit immediately.
#
# 3. **Split carefully.** Use a chronological train/validation split
#    (e.g. train through 2020, validate 2021-2022, hold out 2023 as final
#    check before forecasting 2024-2025). Never shuffle for time series.
#
# 4. **Consider multi-step strategies.** For a 12-month horizon you have
#    three main options: (a) direct multi-output (one model predicts all
#    12 months at once), (b) recursive (predict month 1, feed into model
#    to predict month 2, etc. — error compounds), or (c) iterative
#    per-horizon models (separate model for each forecast lead). Each
#    has tradeoffs. Document your choice.
#
# 5. **Set a small batch size and many epochs with early stopping.**
#    Batch of 16, max 200 epochs, stop when validation RMSE doesn't improve
#    for 20 epochs.
#
# 6. **Try transfer learning.** Pre-train on the cross-country HCP data
#    (5 countries × multiple indicators ≈ more data), then fine-tune on
#    Botswana. This is one of the few ways DL can compete on small data.
#
# 7. **Always compare to a naive baseline.** If your LSTM cannot beat
#    "last year's same month" (a seasonal naive forecast), something is
#    wrong with your DL setup.
#
# ## Example: SARIMA Classical Baseline (Sketch)
#
# ```python
# from statsmodels.tsa.statespace.sarimax import SARIMAX
#
# # Take the target variable from the merged dataset
# target = improved.set_index("year_month")["FAO_23014"].dropna()
#
# # Fit SARIMA with seasonal order — tune (p,d,q)(P,D,Q,s) yourself
# model = SARIMAX(target,
#                 exog=improved[["Brent_USD_per_barrel_lag3",
#                                "BDI_mean_lag3"]].loc[target.index],
#                 order=(1, 1, 1),
#                 seasonal_order=(1, 0, 1, 12))
# fit = model.fit(disp=False)
#
# # Forecast 6 months
# forecast = fit.forecast(steps=6, exog=future_exog)
# ```
#
# ## Example: LSTM Deep Learning Model (Sketch)
#
# ```python
# import torch
# import torch.nn as nn
#
# class FoodInflationLSTM(nn.Module):
#     def __init__(self, n_features, hidden=32, dropout=0.3):
#         super().__init__()
#         self.lstm = nn.LSTM(n_features, hidden, num_layers=2,
#                            dropout=dropout, batch_first=True)
#         self.head = nn.Linear(hidden, 1)
#
#     def forward(self, x):
#         out, _ = self.lstm(x)
#         return self.head(out[:, -1, :])
#
# # Use sliding windows of past N months as input sequences
# # Use early stopping on validation RMSE
# # Document your hyperparameters and training procedure
# ```
#
# ## The Model Comparison Report (Deliverable 1.1c)
#
# Your 5-page report MUST include:
#
# 1. **Architecture description** for both models (specs, hyperparameters)
# 2. **Training procedure** for both (data splits, optimisation, early stopping)
# 3. **Per-model metrics** (RMSE, MAE, at least one more) on validation data
# 4. **Residual diagnostics** for both (residual plots, ACF of residuals)
# 5. **Forecast vs actual visualisation** for both
# 6. **Honest analytical conclusion**: which model won and WHY, citing
#    specific data characteristics (sample size, structural breaks,
#    seasonality, etc.) that favoured one approach over the other.
#
# Teams that honestly conclude "our classical model beat our DL model
# because of small sample size" score HIGHER than teams who claim DL is
# best without evidence.

# %% [markdown]
# ---
# # 7. What You Submit
#
# **Six deliverables uploaded to VenturePulse by end of Week 3:**
#
# | Ref | Deliverable | Format | Limit |
# |-----|-------------|--------|-------|
# | 1.1a | Best-Model Predictions CSV | .csv | 5 MB |
# | 1.1b | Feature Engineering Report | .pdf | 4 pages |
# | 1.1c | Model Comparison Report | .pdf | 5 pages |
# | 1.1d | Code Repository (BOTH models) | GitHub URL | — |
# | 1.2a | HCP Linkage Memo | .pdf | 2 pages |
# | 1.2b | HCP Visualisations | .pdf/.png | 10 MB |
#
# **Predictions CSV format** (exactly 2 columns, 12 rows):
#
# ```
# year_month,forecast
# 2024-01,5.85
# 2024-02,5.70
# 2024-03,5.10
# 2024-04,4.20
# 2024-05,4.00
# 2024-06,4.05
# 2024-07,4.40
# 2024-08,5.10
# 2024-09,5.00
# 2024-10,5.20
# 2024-11,4.80
# 2024-12,4.70
# ```
#
# The platform computes RMSE against actuals. Lower RMSE = higher score.
#
# ---
# # 8. Key Reminders
#
# 1. **Use at least 3 of the 5 datasets.** Single-dataset models capped.
#
# 2. **Submit BOTH a classical and DL model.** Missing one caps your
#    Model Comparison score significantly.
#
# 3. **Engineer features from daily BDI.** A simple monthly mean caps
#    your Daily→Monthly sub-score.
#
# 4. **Document everything.** Justify every variable, lag, and
#    transformation. The AI evaluator reads what you write — if it's
#    not in the report, you don't get credit.
#
# 5. **The target is food price INFLATION (% YoY), not the food price
#    INDEX.** Item 23014, not 23013. Don't confuse them.
#
# 6. **All input data ends Dec 2023.** You forecast 12 months into 2024
#    blind — no actual BDI, Brent, or Policy Rate values are available
#    for the prediction window. You must either forecast these features
#    forward yourself (use them as inputs to your inflation model) OR
#    rely on lagged versions from 2023 and earlier.
#
# 7. **Cross-country data is a feature, not decoration.** South Africa's
#    food inflation may lead Botswana's by 1–2 months.
#
# 8. **Be honest about which model wins.** Dishonest comparisons that
#    claim DL is best when the metrics show otherwise will be penalised.
#
# Good luck. Build something rigorous and honest.
