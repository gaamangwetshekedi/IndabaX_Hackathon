"""End-to-end competition pipeline for Git Push Legends."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from .data import load_and_engineer_data, sequence_channels
from .hcp import analyse_linkages, forward_pressure_summary, regional_projection
from .models import (
    NumpyGRU,
    aggregate_backtest,
    forecast_sarima,
    latest_sequence,
    make_sequence_samples,
    seasonal_naive,
    select_sarima_spec,
    stationarity_diagnostics,
)
from .reporting import (
    BLUE,
    GOLD,
    NAVY,
    bar_chart,
    create_feature_report,
    create_hcp_memo,
    create_hcp_visuals,
    create_model_report,
    line_chart,
)


def _json_ready(value):
    """Convert NumPy/Pandas values into serialisable Python values."""

    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_code_archive(root: Path, output: Path) -> None:
    """Package reproducible code and supplied data without transient files."""

    top_level = [
        "README.md",
        "requirements.txt",
        "config.json",
        "run_pipeline.py",
        "00_DATA_GUIDE.py",
        "01_baltic_dry_index_daily.csv",
        "02_brent_crude_monthly.csv",
        "03_botswana_policy_rate.csv",
        "04_fao_botswana_prices.csv",
        "05_human_capital_project.csv",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in top_level:
            archive.write(root / relative, arcname=relative)
        for directory in ["src", "tests"]:
            for path in sorted((root / directory).rglob("*.py")):
                archive.write(path, arcname=str(path.relative_to(root)))


def _write_submission_pack(root: Path, output: Path, files: dict[str, Path]) -> None:
    """Create a flat convenience archive matching the six upload references."""

    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for archive_name, path in files.items():
            archive.write(path, arcname=archive_name)


def _validate_predictions(path: Path) -> None:
    """Enforce the exact two-column, 12-row competition contract."""

    frame = pd.read_csv(path)
    expected_months = pd.date_range("2024-01-01", periods=12, freq="MS").strftime("%Y-%m").tolist()
    if list(frame.columns) != ["year_month", "forecast"]:
        raise AssertionError("Predictions must contain exactly year_month and forecast")
    if len(frame) != 12 or frame["year_month"].tolist() != expected_months:
        raise AssertionError("Predictions must contain January-December 2024 in order")
    if not np.isfinite(frame["forecast"].to_numpy(dtype=float)).all():
        raise AssertionError("Forecasts must all be finite numeric values")


def run(root: str | Path = ".") -> dict:
    """Execute data preparation, modeling, reporting and final QA."""

    root = Path(root).resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    outputs = root / "outputs"
    processed = root / "data" / "processed"
    figures = root / "reports" / "figures"
    pdf_dir = root / "output" / "pdf"
    for directory in (outputs, processed, figures, pdf_dir):
        directory.mkdir(parents=True, exist_ok=True)

    bundle = load_and_engineer_data(root)
    monthly = bundle.monthly
    origin_features = bundle.origin_features
    target = bundle.target
    channels = sequence_channels(monthly)
    monthly.to_csv(processed / "monthly_merged.csv", index_label="Date")
    origin_features.to_csv(processed / "origin_features.csv", index_label="Date")
    _write_json(processed / "data_audit.json", bundle.audit)

    context = int(config["gru_context_months"])
    hidden = int(config["gru_hidden_units"])
    sequence = make_sequence_samples(channels, target, context=context)
    stationarity = stationarity_diagnostics(target)
    _write_json(outputs / "stationarity_tests.json", stationarity)
    sarima_order, sarima_seasonal_order, sarima_tuning = select_sarima_spec(
        target,
        years=tuple(config["sarima_tuning_years"]),
    )
    sarima_tuning.to_csv(outputs / "sarima_tuning_scores.csv", index=False)

    backtest_rows: list[dict] = []
    gru_runs: list[dict] = []
    sarima_runs: list[dict] = []
    for year in config["evaluation_years"]:
        origin = pd.Timestamp(year=int(year) - 1, month=12, day=1)
        dates = pd.date_range(f"{year}-01-01", periods=12, freq="MS")
        actual = target.loc[dates].to_numpy(dtype=float)

        sarima_prediction, sarima_fit = forecast_sarima(
            target,
            origin,
            sarima_order,
            sarima_seasonal_order,
        )
        sarima_runs.append({"forecast_year": int(year), **sarima_fit})

        sequence_train = sequence.through(origin)
        gru = NumpyGRU(
            input_size=sequence_train.x.shape[2],
            hidden_size=hidden,
            output_size=12,
            seed=int(config["random_seed"]),
        )
        gru.fit(sequence_train.x, sequence_train.y)
        gru_prediction = gru.predict(latest_sequence(channels, origin, context))[0]
        naive_prediction = seasonal_naive(target, int(year))

        gru_runs.append(
            {
                "forecast_year": int(year),
                "training_samples": int(len(sequence_train.x)),
                "best_epoch": int(gru.best_epoch),
                "epochs_run": int(len(gru.history)),
                "parameter_count": int(gru.parameter_count),
            }
        )
        for model_name, prediction in {
            "SARIMA": sarima_prediction,
            "NumPy GRU": gru_prediction,
            "Seasonal Naive": naive_prediction,
        }.items():
            for date, observed, estimated in zip(dates, actual, prediction):
                backtest_rows.append(
                    {
                        "date": date.strftime("%Y-%m"),
                        "forecast_year": int(year),
                        "model": model_name,
                        "actual": float(observed),
                        "predicted": float(estimated),
                        "residual": float(estimated - observed),
                    }
                )

    backtests = pd.DataFrame(backtest_rows)
    backtests.to_csv(outputs / "rolling_backtest_predictions.csv", index=False)
    model_names = ["SARIMA", "NumPy GRU", "Seasonal Naive"]
    metrics = {name: aggregate_backtest(backtest_rows, name) for name in model_names}
    eligible = ["SARIMA", "NumPy GRU"]
    winner = min(eligible, key=lambda name: metrics[name]["rmse"])

    final_origin = pd.Timestamp(config["forecast_origin"])
    final_sarima_forecast, final_sarima_fit = forecast_sarima(
        target,
        final_origin,
        sarima_order,
        sarima_seasonal_order,
    )

    final_sequence_train = sequence.through(final_origin)
    final_gru = NumpyGRU(
        input_size=final_sequence_train.x.shape[2],
        hidden_size=hidden,
        output_size=12,
        seed=int(config["random_seed"]),
    )
    final_gru.fit(final_sequence_train.x, final_sequence_train.y)
    final_gru_forecast = final_gru.predict(latest_sequence(channels, final_origin, context))[0]
    forecasts = {"SARIMA": final_sarima_forecast, "NumPy GRU": final_gru_forecast}
    selected_forecast = forecasts[winner]

    forecast_dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    predictions = pd.DataFrame(
        {
            "year_month": forecast_dates.strftime("%Y-%m"),
            "forecast": np.round(selected_forecast, 6),
        }
    )
    predictions_path = outputs / "predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    _validate_predictions(predictions_path)
    pd.DataFrame(
        {
            "year_month": forecast_dates.strftime("%Y-%m"),
            "SARIMA": final_sarima_forecast,
            "NumPy GRU": final_gru_forecast,
            "selected_model": winner,
        }
    ).to_csv(outputs / "model_forecast_candidates.csv", index=False)
    pd.DataFrame(final_gru.history).to_csv(outputs / "final_gru_training_history.csv", index=False)

    actual_june = float(target.loc["2023-06-01"])
    actual_october = float(target.loc["2023-10-01"])
    forecast_june = float(selected_forecast[5])
    forecast_october = float(selected_forecast[9])
    pitch_explanation = (
        f"The June-October easing follows 2023's actual decline from {actual_june:.1f}% to "
        f"{actual_october:.1f}%; the {winner} extends that seasonal/base-effect disinflation, "
        f"although its steeper 2024 decline from {forecast_june:.1f}% to {forecast_october:.1f}% "
        "is a key forecast uncertainty."
    )
    forecast_pattern_check = {
        "actual_2023_june": actual_june,
        "actual_2023_october": actual_october,
        "actual_2023_drop": actual_june - actual_october,
        "forecast_2024_june": forecast_june,
        "forecast_2024_october": forecast_october,
        "forecast_2024_drop": forecast_june - forecast_october,
        "pitch_explanation": pitch_explanation,
    }
    _write_json(outputs / "forecast_pattern_check.json", forecast_pattern_check)
    (outputs / "pitch_notes.md").write_text(
        "# Pitch Notes\n\n## June-October forecast movement\n\n"
        + pitch_explanation
        + "\n",
        encoding="utf-8",
    )

    linkage = analyse_linkages(monthly)
    projection = regional_projection(monthly, selected_forecast)
    linkage["forward_pressure"] = forward_pressure_summary(projection)
    projection.to_csv(outputs / "hcp_forward_projection.csv", index_label="Date")
    _write_json(outputs / "hcp_linkage_results.json", linkage)
    _write_json(outputs / "hcp_forward_pressure_summary.json", linkage["forward_pressure"])

    # Figures are generated from saved/observable results, not hard-coded values.
    shipping_plot = monthly.loc["2019-01-01":, ["shipping_mean", "shipping_range"]].rename(
        columns={"shipping_mean": "Monthly mean", "shipping_range": "Within-month range"}
    )
    line_chart(
        shipping_plot,
        "Shipping-cost level and within-month stress",
        "Baltic Dry Index daily records aggregated to monthly features, 2019-2023",
        figures / "shipping_features.png",
        y_label="Index points",
    )

    drivers = monthly.loc["2018-01-01":, ["bwa_food_inflation", "shipping_mean", "brent_usd", "policy_rate"]].copy()
    drivers = (drivers - drivers.mean()) / drivers.std(ddof=0).replace(0.0, 1.0)
    drivers = drivers.rename(
        columns={
            "bwa_food_inflation": "Botswana food inflation",
            "shipping_mean": "Shipping index",
            "brent_usd": "Brent",
            "policy_rate": "Policy rate",
        }
    )
    line_chart(
        drivers,
        "Standardised economic drivers",
        "Z-scores shown to compare series with different units, 2018-2023",
        figures / "economic_drivers.png",
        y_label="Standard deviations",
    )

    metric_frame = pd.DataFrame(
        {name: {"RMSE": metrics[name]["rmse"], "MAE": metrics[name]["mae"]} for name in model_names}
    ).T
    bar_chart(
        metric_frame,
        "Rolling-origin forecast error",
        "Aggregate performance across calendar years 2021-2023; lower is better",
        figures / "model_metrics.png",
    )

    backtest_plot = backtests.pivot_table(index="date", columns="model", values="predicted", aggfunc="first")
    actual_plot = backtests.drop_duplicates("date").set_index("date")["actual"]
    backtest_plot.insert(0, "Actual", actual_plot)
    backtest_plot.index = pd.to_datetime(backtest_plot.index)
    line_chart(
        backtest_plot[["Actual", "SARIMA", "NumPy GRU"]],
        "Out-of-sample forecasts versus actual inflation",
        "Three rolling 12-month evaluation windows, 2021-2023",
        figures / "backtest_forecasts.png",
        colors=[NAVY, BLUE, GOLD],
        y_label="Inflation, % YoY",
    )
    residual_plot = backtests.pivot_table(index="date", columns="model", values="residual", aggfunc="first")
    residual_plot.index = pd.to_datetime(residual_plot.index)
    line_chart(
        residual_plot[["SARIMA", "NumPy GRU", "Seasonal Naive"]],
        "Out-of-sample residuals",
        "Prediction minus actual; values nearer zero are better",
        figures / "backtest_residuals.png",
        colors=[BLUE, GOLD, NAVY],
        y_label="Residual, percentage points",
    )

    final_plot = pd.DataFrame(index=pd.date_range("2022-01-01", "2024-12-01", freq="MS"))
    final_plot["Observed"] = target.reindex(final_plot.index)
    final_plot["Selected forecast"] = np.nan
    final_plot.loc[forecast_dates, "Selected forecast"] = selected_forecast
    final_plot.loc[pd.Timestamp("2023-12-01"), "Selected forecast"] = target.loc["2023-12-01"]
    line_chart(
        final_plot,
        f"Botswana food-inflation forecast: {winner}",
        "Observed through December 2023; forecast January-December 2024",
        figures / "final_forecast.png",
        colors=[NAVY, BLUE],
        y_label="Inflation, % YoY",
    )

    hcp_history = monthly.loc["2018-01-01":, ["bwa_food_inflation", "zaf_23014", "nam_23014"]].rename(
        columns={"bwa_food_inflation": "Botswana", "zaf_23014": "South Africa", "nam_23014": "Namibia"}
    )
    line_chart(
        hcp_history,
        "Regional food-price inflation co-movement",
        "Supplied FAO cross-country indicators, 2018-2023",
        figures / "hcp_history.png",
        colors=[NAVY, BLUE, GOLD],
        y_label="Inflation, % YoY",
    )
    line_chart(
        projection,
        "2024 regional price-pressure projection",
        "Botswana model forecast; comparison countries use 2023 seasonal baselines",
        figures / "hcp_projection.png",
        colors=[NAVY, BLUE, GOLD],
        y_label="Inflation, % YoY",
    )

    chart_paths = {
        "shipping": figures / "shipping_features.png",
        "drivers": figures / "economic_drivers.png",
        "metrics": figures / "model_metrics.png",
        "backtest": figures / "backtest_forecasts.png",
        "residuals": figures / "backtest_residuals.png",
        "final": figures / "final_forecast.png",
        "hcp_history": figures / "hcp_history.png",
        "hcp_projection": figures / "hcp_projection.png",
    }
    if winner == "SARIMA":
        conclusion = (
            "SARIMA won. Explicit differencing and annual seasonal structure generalised better than the "
            "higher-variance GRU on the 276-month target series. The GRU remains a valid recurrent "
            "comparison but did not justify replacing the lower-error classical forecast."
        )
    else:
        conclusion = (
            "The NumPy GRU won. Its gated sequence representation captured nonlinear persistence that the "
            "univariate SARIMA missed, while dropout, L2 regularisation and early stopping controlled the "
            "small-sample risk."
        )
    report_context = {
        "metrics": metrics,
        "winner": winner,
        "feature_count": int(origin_features.shape[1]),
        "sarima_order": sarima_order,
        "sarima_seasonal_order": sarima_seasonal_order,
        "sarima_tuning_years": config["sarima_tuning_years"],
        "sarima_final_aic": final_sarima_fit["aic"],
        "stationarity": stationarity,
        "gru_context": context,
        "gru_channels": int(channels.shape[1]),
        "gru_hidden": hidden,
        "gru_parameters": int(final_gru.parameter_count),
        "conclusion": conclusion,
        "forecast_pattern_explanation": pitch_explanation,
    }
    create_feature_report(
        pdf_dir / "feature_engineering_report.pdf",
        bundle.audit,
        int(origin_features.shape[1]),
        chart_paths,
    )
    create_model_report(pdf_dir / "model_comparison_report.pdf", report_context, chart_paths)
    create_hcp_memo(pdf_dir / "hcp_linkage_memo.pdf", linkage, chart_paths)
    create_hcp_visuals(pdf_dir / "hcp_visualisations.pdf", chart_paths)

    results = {
        "team": config["team_name"],
        "members": config["team_members"],
        "winner": winner,
        "metrics": metrics,
        "feature_count": int(origin_features.shape[1]),
        "sarima_order": sarima_order,
        "sarima_seasonal_order": sarima_seasonal_order,
        "sarima_tuning_years": config["sarima_tuning_years"],
        "sarima_final_fit": final_sarima_fit,
        "sarima_backtest_runs": sarima_runs,
        "stationarity": stationarity,
        "gru_training_samples": int(len(final_sequence_train.x)),
        "gru_parameters": int(final_gru.parameter_count),
        "gru_best_epoch": int(final_gru.best_epoch),
        "gru_epochs_run": int(len(final_gru.history)),
        "gru_backtest_runs": gru_runs,
        "forecast": predictions.to_dict("records"),
        "forecast_pattern_check": forecast_pattern_check,
        "data_audit": bundle.audit,
        "hcp": linkage,
    }
    _write_json(outputs / "results.json", results)

    code_archive = root / "output" / "Git_Push_Legends_Code_Repository.zip"
    _write_code_archive(root, code_archive)
    deliverables = [
        predictions_path,
        pdf_dir / "feature_engineering_report.pdf",
        pdf_dir / "model_comparison_report.pdf",
        pdf_dir / "hcp_linkage_memo.pdf",
        pdf_dir / "hcp_visualisations.pdf",
        code_archive,
    ]
    manifest = {
        "team": config["team_name"],
        "generated_from_data_through": "2023-12-31",
        "selected_model": winner,
        "files": [
            {
                "path": str(path.relative_to(root)),
                "bytes": path.stat().st_size,
                "sha256": _hash_file(path),
            }
            for path in deliverables
        ],
    }
    _write_json(outputs / "submission_manifest.json", manifest)
    submission_pack = root / "output" / "Git_Push_Legends_Phase1_Submission_Pack.zip"
    _write_submission_pack(
        root,
        submission_pack,
        {
            "1.1a_predictions.csv": predictions_path,
            "1.1b_feature_engineering_report.pdf": pdf_dir / "feature_engineering_report.pdf",
            "1.1c_model_comparison_report.pdf": pdf_dir / "model_comparison_report.pdf",
            "1.1d_code_repository.zip": code_archive,
            "1.2a_hcp_linkage_memo.pdf": pdf_dir / "hcp_linkage_memo.pdf",
            "1.2b_hcp_visualisations.pdf": pdf_dir / "hcp_visualisations.pdf",
            "submission_manifest.json": outputs / "submission_manifest.json",
        },
    )
    return results
