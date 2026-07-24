"""Classical SARIMA and deep-learning GRU forecasting models.

Both models generate a complete 12-month forecast from a December origin using
only information available at that date. SARIMA uses the observed target
history; the compact NumPy GRU uses origin-safe economic sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


HORIZON = 12
SARIMA_CANDIDATES = (
    ((1, 0, 0), (1, 0, 0, 12)),
    ((1, 0, 1), (1, 0, 0, 12)),
    ((2, 0, 0), (1, 0, 0, 12)),
    ((0, 1, 1), (0, 1, 1, 12)),
    ((1, 1, 0), (1, 0, 0, 12)),
    ((1, 1, 1), (1, 0, 1, 12)),
    ((0, 1, 1), (1, 0, 0, 12)),
    ((1, 0, 1), (0, 1, 1, 12)),
)


def metric_set(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Return competition RMSE plus two complementary error metrics."""

    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    error = predicted - actual
    denominator = np.abs(actual) + np.abs(predicted)
    smape = np.mean(np.where(denominator > 1e-9, 200.0 * np.abs(error) / denominator, 0.0))
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "smape": float(smape),
        "bias": float(np.mean(error)),
    }


def residual_autocorrelation(residuals: np.ndarray, lag: int) -> float:
    """Compute a simple residual autocorrelation at the requested lag."""

    values = np.asarray(residuals, dtype=float).reshape(-1)
    if len(values) <= lag or np.std(values) < 1e-12:
        return 0.0
    return float(np.corrcoef(values[:-lag], values[lag:])[0, 1])


@dataclass
class SupervisedData:
    """Direct multi-horizon samples indexed by their forecast origin."""

    origins: pd.DatetimeIndex
    label_ends: pd.DatetimeIndex
    x: np.ndarray
    y: np.ndarray
    feature_names: list[str]

    def through(self, cutoff: pd.Timestamp) -> "SupervisedData":
        """Keep samples whose full 12-month label window is observable."""

        mask = self.label_ends <= pd.Timestamp(cutoff)
        return SupervisedData(
            origins=self.origins[mask],
            label_ends=self.label_ends[mask],
            x=self.x[mask],
            y=self.y[mask],
            feature_names=self.feature_names,
        )


def make_direct_samples(
    features: pd.DataFrame,
    target: pd.Series,
    horizon: int = HORIZON,
) -> SupervisedData:
    """Create origin-to-next-year training pairs without future covariates."""

    rows_x: list[np.ndarray] = []
    rows_y: list[np.ndarray] = []
    origins: list[pd.Timestamp] = []
    label_ends: list[pd.Timestamp] = []
    for position, origin in enumerate(features.index):
        end = position + horizon + 1
        if end > len(features):
            continue
        x_row = features.iloc[position].to_numpy(dtype=float)
        y_row = target.iloc[position + 1 : end].to_numpy(dtype=float)
        if len(y_row) != horizon or not np.isfinite(x_row).all() or not np.isfinite(y_row).all():
            continue
        rows_x.append(x_row)
        rows_y.append(y_row)
        origins.append(pd.Timestamp(origin))
        label_ends.append(pd.Timestamp(target.index[end - 1]))
    return SupervisedData(
        origins=pd.DatetimeIndex(origins),
        label_ends=pd.DatetimeIndex(label_ends),
        x=np.vstack(rows_x),
        y=np.vstack(rows_y),
        feature_names=list(features.columns),
    )


@dataclass
class Standardizer:
    """Train-only standardisation parameters."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray, axes: int | tuple[int, ...] = 0) -> "Standardizer":
        mean = np.mean(values, axis=axes, keepdims=True)
        scale = np.std(values, axis=axes, keepdims=True)
        scale = np.where(scale < 1e-8, 1.0, scale)
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


def stationarity_diagnostics(target: pd.Series) -> dict[str, dict[str, float | int]]:
    """Run ADF and KPSS tests on levels and two common differences."""

    clean = target.dropna().astype(float)
    variants = {
        "level": clean,
        "first_difference": clean.diff().dropna(),
        "seasonal_difference_12": clean.diff(12).dropna(),
    }
    diagnostics: dict[str, dict[str, float | int]] = {}
    for label, values in variants.items():
        adf_result = adfuller(values, autolag="AIC")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            kpss_result = kpss(values, regression="c", nlags="auto")
        diagnostics[label] = {
            "observations": int(len(values)),
            "adf_statistic": float(adf_result[0]),
            "adf_p_value": float(adf_result[1]),
            "adf_lags": int(adf_result[2]),
            "kpss_statistic": float(kpss_result[0]),
            "kpss_p_value": float(kpss_result[1]),
            "kpss_lags": int(kpss_result[2]),
        }
    return diagnostics


def forecast_sarima(
    target: pd.Series,
    origin: pd.Timestamp,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    steps: int = HORIZON,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Fit SARIMA through ``origin`` and return an out-of-sample forecast."""

    train = target.dropna().astype(float).loc[: pd.Timestamp(origin)]
    trend = "c" if order[1] == 0 else "n"
    fitted = SARIMAX(
        train,
        order=order,
        seasonal_order=seasonal_order,
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    ).fit(disp=False, maxiter=300)
    prediction = np.asarray(fitted.forecast(steps), dtype=float)
    return prediction, {
        "aic": float(fitted.aic),
        "bic": float(fitted.bic),
        "converged": bool(fitted.mle_retvals.get("converged", False)),
        "training_months": int(len(train)),
    }


def select_sarima_spec(
    target: pd.Series,
    years: tuple[int, ...] = (2018, 2019, 2020),
    candidates: tuple[
        tuple[tuple[int, int, int], tuple[int, int, int, int]], ...
    ] = SARIMA_CANDIDATES,
) -> tuple[tuple[int, int, int], tuple[int, int, int, int], pd.DataFrame]:
    """Select a compact SARIMA specification on pre-evaluation years."""

    rows: list[dict] = []
    for order, seasonal_order in candidates:
        for year in years:
            origin = pd.Timestamp(year=int(year) - 1, month=12, day=1)
            prediction, fit = forecast_sarima(
                target,
                origin,
                order,
                seasonal_order,
            )
            actual = target.loc[f"{year}-01-01":f"{year}-12-01"].to_numpy(dtype=float)
            rows.append(
                {
                    "order": ",".join(str(value) for value in order),
                    "seasonal_order": ",".join(str(value) for value in seasonal_order),
                    "year": int(year),
                    **metric_set(actual, prediction),
                    **fit,
                }
            )
    scores = pd.DataFrame(rows)
    summary = (
        scores.groupby(["order", "seasonal_order"], as_index=False)[["rmse", "mae", "smape"]]
        .mean()
        .sort_values(["rmse", "mae"])
    )
    best = summary.iloc[0]
    order = tuple(int(value) for value in str(best["order"]).split(","))
    seasonal_order = tuple(int(value) for value in str(best["seasonal_order"]).split(","))
    return order, seasonal_order, scores


def seasonal_naive(target: pd.Series, forecast_year: int) -> np.ndarray:
    """Forecast a calendar year using the preceding year's monthly values."""

    previous = target.loc[f"{forecast_year - 1}-01-01":f"{forecast_year - 1}-12-01"]
    if len(previous) != HORIZON:
        raise ValueError(f"Expected 12 previous-year values for {forecast_year}")
    return previous.to_numpy(dtype=float)


@dataclass
class SequenceData:
    """Sliding sequence samples for the recurrent neural network."""

    origins: pd.DatetimeIndex
    label_ends: pd.DatetimeIndex
    x: np.ndarray
    y: np.ndarray
    channel_names: list[str]
    context: int

    def through(self, cutoff: pd.Timestamp) -> "SequenceData":
        mask = self.label_ends <= pd.Timestamp(cutoff)
        return SequenceData(
            origins=self.origins[mask],
            label_ends=self.label_ends[mask],
            x=self.x[mask],
            y=self.y[mask],
            channel_names=self.channel_names,
            context=self.context,
        )


def make_sequence_samples(
    channels: pd.DataFrame,
    target: pd.Series,
    context: int = 24,
    horizon: int = HORIZON,
) -> SequenceData:
    """Create context-window to direct-horizon samples for the GRU."""

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    origins: list[pd.Timestamp] = []
    label_ends: list[pd.Timestamp] = []
    for position in range(context - 1, len(channels) - horizon):
        x_window = channels.iloc[position - context + 1 : position + 1].to_numpy(dtype=float)
        y_window = target.iloc[position + 1 : position + horizon + 1].to_numpy(dtype=float)
        if not np.isfinite(x_window).all() or not np.isfinite(y_window).all():
            continue
        xs.append(x_window)
        ys.append(y_window)
        origins.append(pd.Timestamp(channels.index[position]))
        label_ends.append(pd.Timestamp(target.index[position + horizon]))
    return SequenceData(
        origins=pd.DatetimeIndex(origins),
        label_ends=pd.DatetimeIndex(label_ends),
        x=np.stack(xs),
        y=np.stack(ys),
        channel_names=list(channels.columns),
        context=context,
    )


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic activation."""

    clipped = np.clip(values, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))


class NumpyGRU:
    """A compact gated recurrent unit with direct 12-month output.

    The implementation exposes every matrix and gradient, making the model
    auditable without requiring PyTorch or TensorFlow. It uses full-batch Adam,
    input dropout, L2 regularisation, gradient clipping and temporal early
    stopping - all appropriate for roughly 250 monthly training samples.
    """

    def __init__(self, input_size: int, hidden_size: int = 16, output_size: int = HORIZON, seed: int = 17):
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.output_size = int(output_size)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)

        def matrix(rows: int, cols: int) -> np.ndarray:
            limit = np.sqrt(6.0 / (rows + cols))
            return rng.uniform(-limit, limit, size=(rows, cols))

        self.params = {
            "Wz": matrix(input_size, hidden_size),
            "Uz": matrix(hidden_size, hidden_size),
            "bz": np.zeros(hidden_size),
            "Wr": matrix(input_size, hidden_size),
            "Ur": matrix(hidden_size, hidden_size),
            "br": np.zeros(hidden_size),
            "Wn": matrix(input_size, hidden_size),
            "Un": matrix(hidden_size, hidden_size),
            "bn": np.zeros(hidden_size),
            "Why": matrix(hidden_size, output_size),
            "by": np.zeros(output_size),
        }
        self.x_scaler: Standardizer | None = None
        self.y_scaler: Standardizer | None = None
        self.best_epoch = 0
        self.history: list[dict[str, float]] = []

    @property
    def parameter_count(self) -> int:
        return int(sum(value.size for value in self.params.values()))

    def _forward(self, x: np.ndarray) -> tuple[np.ndarray, list[tuple]]:
        batch, steps, _ = x.shape
        h = np.zeros((batch, self.hidden_size))
        cache: list[tuple] = []
        p = self.params
        for step in range(steps):
            xt = x[:, step, :]
            h_prev = h
            z = _sigmoid(xt @ p["Wz"] + h_prev @ p["Uz"] + p["bz"])
            r = _sigmoid(xt @ p["Wr"] + h_prev @ p["Ur"] + p["br"])
            candidate = np.tanh(xt @ p["Wn"] + (r * h_prev) @ p["Un"] + p["bn"])
            h = (1.0 - z) * candidate + z * h_prev
            cache.append((xt, h_prev, z, r, candidate, h))
        prediction = h @ p["Why"] + p["by"]
        return prediction, cache

    def _loss_and_gradients(self, x: np.ndarray, y: np.ndarray, l2: float) -> tuple[float, dict[str, np.ndarray]]:
        prediction, cache = self._forward(x)
        error = prediction - y
        mse = float(np.mean(error**2))
        regularised = ["Wz", "Uz", "Wr", "Ur", "Wn", "Un", "Why"]
        loss = mse + float(l2) * sum(float(np.sum(self.params[name] ** 2)) for name in regularised)
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        d_prediction = 2.0 * error / error.size
        last_h = cache[-1][-1]
        grads["Why"] = last_h.T @ d_prediction + 2.0 * l2 * self.params["Why"]
        grads["by"] = d_prediction.sum(axis=0)
        d_h = d_prediction @ self.params["Why"].T
        p = self.params

        for xt, h_prev, z, r, candidate, _h in reversed(cache):
            d_candidate = d_h * (1.0 - z)
            d_z = d_h * (h_prev - candidate)
            d_h_prev = d_h * z

            d_an = d_candidate * (1.0 - candidate**2)
            grads["Wn"] += xt.T @ d_an
            grads["Un"] += (r * h_prev).T @ d_an
            grads["bn"] += d_an.sum(axis=0)
            d_rh = d_an @ p["Un"].T
            d_r = d_rh * h_prev
            d_h_prev += d_rh * r

            d_ar = d_r * r * (1.0 - r)
            grads["Wr"] += xt.T @ d_ar
            grads["Ur"] += h_prev.T @ d_ar
            grads["br"] += d_ar.sum(axis=0)
            d_h_prev += d_ar @ p["Ur"].T

            d_az = d_z * z * (1.0 - z)
            grads["Wz"] += xt.T @ d_az
            grads["Uz"] += h_prev.T @ d_az
            grads["bz"] += d_az.sum(axis=0)
            d_h_prev += d_az @ p["Uz"].T
            d_h = d_h_prev

        for name in ["Wz", "Uz", "Wr", "Ur", "Wn", "Un"]:
            grads[name] += 2.0 * l2 * p[name]
        total_norm = np.sqrt(sum(float(np.sum(gradient**2)) for gradient in grads.values()))
        if total_norm > 5.0:
            factor = 5.0 / total_norm
            grads = {name: gradient * factor for name, gradient in grads.items()}
        return loss, grads

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        validation_months: int = 24,
        learning_rate: float = 0.01,
        l2: float = 1e-4,
        dropout: float = 0.15,
        max_epochs: int = 500,
        patience: int = 60,
    ) -> "NumpyGRU":
        """Train chronologically, reserving the latest samples for stopping."""

        if len(x) <= validation_months + 24:
            raise ValueError("Not enough samples for temporal GRU training")
        split = len(x) - validation_months
        x_train, x_val = x[:split], x[split:]
        y_train, y_val = y[:split], y[split:]
        self.x_scaler = Standardizer.fit(x_train, axes=(0, 1))
        self.y_scaler = Standardizer.fit(y_train, axes=0)
        x_train = self.x_scaler.transform(x_train)
        x_val = self.x_scaler.transform(x_val)
        y_train = self.y_scaler.transform(y_train)
        y_val = self.y_scaler.transform(y_val)

        first = {name: np.zeros_like(value) for name, value in self.params.items()}
        second = {name: np.zeros_like(value) for name, value in self.params.items()}
        best = {name: value.copy() for name, value in self.params.items()}
        best_loss = np.inf
        stale = 0
        rng = np.random.default_rng(self.seed + 1000)
        self.history = []

        for epoch in range(1, max_epochs + 1):
            if dropout > 0.0:
                mask = rng.binomial(1, 1.0 - dropout, size=x_train.shape) / (1.0 - dropout)
                x_epoch = x_train * mask
            else:
                x_epoch = x_train
            train_loss, grads = self._loss_and_gradients(x_epoch, y_train, l2=l2)
            for name in self.params:
                first[name] = 0.9 * first[name] + 0.1 * grads[name]
                second[name] = 0.999 * second[name] + 0.001 * (grads[name] ** 2)
                m_hat = first[name] / (1.0 - 0.9**epoch)
                v_hat = second[name] / (1.0 - 0.999**epoch)
                self.params[name] -= learning_rate * m_hat / (np.sqrt(v_hat) + 1e-8)

            val_prediction, _ = self._forward(x_val)
            val_loss = float(np.mean((val_prediction - y_val) ** 2))
            self.history.append({"epoch": float(epoch), "train_loss": train_loss, "validation_loss": val_loss})
            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best = {name: value.copy() for name, value in self.params.items()}
                self.best_epoch = epoch
                stale = 0
            else:
                stale += 1
            if stale >= patience:
                break
        self.params = best
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_scaler is None or self.y_scaler is None:
            raise RuntimeError("The GRU must be fitted before prediction")
        values = np.asarray(x, dtype=float)
        if values.ndim == 2:
            values = values[None, :, :]
        prediction, _ = self._forward(self.x_scaler.transform(values))
        return self.y_scaler.inverse(prediction)


def latest_sequence(channels: pd.DataFrame, origin: pd.Timestamp, context: int) -> np.ndarray:
    """Return the context window ending at a specific monthly origin."""

    window = channels.loc[:origin].tail(context).to_numpy(dtype=float)
    if window.shape[0] != context or not np.isfinite(window).all():
        raise ValueError(f"Incomplete sequence at {origin.date()}")
    return window


def aggregate_backtest(rows: list[dict], model_name: str) -> dict[str, float]:
    """Aggregate monthly predictions from multiple rolling-origin years."""

    frame = pd.DataFrame([row for row in rows if row["model"] == model_name])
    metrics = metric_set(frame["actual"].to_numpy(), frame["predicted"].to_numpy())
    residual = frame["predicted"].to_numpy() - frame["actual"].to_numpy()
    metrics["residual_acf1"] = residual_autocorrelation(residual, 1)
    metrics["residual_acf12"] = residual_autocorrelation(residual, 12)
    return metrics
