"""One-step-ahead forecasts with chronological per-scenario evaluation."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from src.workload_generator import INTERVAL_SECONDS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "workload.csv"
OUT = ROOT / "results"
LAGS = 6
SEASONAL_LAG = 120
TRAIN_FRACTION = 0.75


def create_features(values):
    """Each row predicts target interval t from values available through t-1."""
    frame = pd.DataFrame({"actual_rps": pd.to_numeric(values, errors="coerce")})
    if frame.empty or frame.actual_rps.isna().any() or not np.isfinite(frame.actual_rps).all():
        raise ValueError("values must contain finite numeric request rates")
    if (frame.actual_rps < 0).any():
        raise ValueError("values must contain non-negative request rates")
    for lag in range(1, LAGS + 1):
        frame[f"lag_{lag}"] = frame["actual_rps"].shift(lag)
    frame[f"seasonal_lag_{SEASONAL_LAG}"] = frame["actual_rps"].shift(SEASONAL_LAG)
    frame["target_interval"] = np.arange(len(frame))
    return frame.dropna().reset_index(drop=True)


def chronological_split(data, split_target):
    """Split forecast targets at a fixed interval, without shuffling."""
    if (isinstance(split_target, bool) or not isinstance(split_target, (int, np.integer))
            or split_target < 1):
        raise ValueError("split_target must be a positive integer")
    required = {"target_interval", "actual_rps"}
    if not required.issubset(data.columns):
        raise ValueError(f"forecast data missing columns: {sorted(required - set(data.columns))}")
    if data[list(required)].isna().any().any():
        raise ValueError("forecast data contains missing target values")
    train = data[data.target_interval < split_target].copy()
    test = data[data.target_interval >= split_target].copy()
    if train.empty or test.empty:
        raise ValueError("chronological split must leave non-empty train and test sets")
    if int(train.target_interval.max()) >= int(test.target_interval.min()):
        raise ValueError("training targets must precede all test targets")
    return train, test


def build_forecast_frame(scenario, test, predictions):
    """Return forecast records whose origin is exactly one interval before target."""
    if not isinstance(scenario, str) or not scenario:
        raise ValueError("scenario must be a non-empty string")
    target = test.target_interval.astype(int).to_numpy()
    actual = test.actual_rps.to_numpy(dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if len(target) != len(actual) or len(target) != len(predictions):
        raise ValueError("test targets, actuals and predictions must have equal lengths")
    if (not np.isfinite(actual).all() or (actual < 0).any()
            or not np.isfinite(predictions).all() or (predictions < 0).any()):
        raise ValueError("actuals and predictions must be finite and non-negative")
    return pd.DataFrame({
        "scenario": scenario,
        "interval": target,
        "elapsed_seconds": target * INTERVAL_SECONDS,
        "forecast_origin_interval": target - 1,
        "actual_rps": actual,
        "predicted_rps": predictions,
    })


def main():
    if not DATA.exists():
        raise SystemExit("Missing data/workload.csv. Run python -m src.workload_generator first.")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    required = {"scenario", "interval", "elapsed_seconds", "request_rate_rps"}
    if not required.issubset(df.columns):
        raise SystemExit(f"workload.csv missing columns: {sorted(required - set(df.columns))}")
    if df.duplicated(["scenario", "interval"]).any():
        raise SystemExit("workload.csv has duplicate scenario/interval keys")
    interval_values = pd.to_numeric(df.interval, errors="coerce").to_numpy()
    if not np.isfinite(interval_values).all() or not np.equal(interval_values, np.floor(interval_values)).all():
        raise SystemExit("workload.csv intervals must be finite integers")
    if not np.isfinite(df["request_rate_rps"].astype(float)).all():
        raise SystemExit("workload.csv contains non-finite request rates")
    for scenario, subset in df.groupby("scenario"):
        intervals = subset.interval.astype(int).sort_values().to_numpy()
        if not np.array_equal(intervals, np.arange(len(intervals))):
            raise SystemExit(f"{scenario}: intervals must be contiguous from zero")
        elapsed = pd.to_numeric(subset.sort_values("interval").elapsed_seconds, errors="coerce").to_numpy()
        if not np.array_equal(elapsed, intervals * INTERVAL_SECONDS):
            raise SystemExit(f"{scenario}: elapsed_seconds do not match interval spacing")

    all_predictions, metrics = [], []
    scenarios = sorted(df["scenario"].unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(12, 3 * len(scenarios)), constrained_layout=True)
    if len(scenarios) == 1:
        axes = [axes]
    for ax, scenario in zip(axes, scenarios):
        subset = df[df.scenario == scenario].sort_values("interval")
        values = subset.request_rate_rps.astype(float).reset_index(drop=True)
        data = create_features(values)
        split_target = int(len(values) * TRAIN_FRACTION)
        train, test = chronological_split(data, split_target)
        features = [c for c in data.columns if c not in {"actual_rps", "target_interval"}]
        model = LinearRegression().fit(train[features], train.actual_rps)
        predictions = np.maximum(0, model.predict(test[features]))
        actual = test.actual_rps.to_numpy()
        mae = mean_absolute_error(actual, predictions)
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        target = test.target_interval.astype(int).to_numpy()
        result = build_forecast_frame(scenario, test, predictions)
        all_predictions.append(result)
        metrics.append({"scenario": scenario, "mae_rps": round(mae, 3),
                        "rmse_rps": round(rmse, 3), "test_points": len(test),
                        "train_last_interval": split_target - 1,
                        "test_first_interval": int(target[0])})
        ax.plot(target, actual, label="Actual")
        ax.plot(target, predictions, label="One-step forecast")
        ax.axvline(split_target, color="black", linestyle="--", alpha=0.5, label="Train/test split")
        ax.set_title(f"{scenario.title()} (MAE={mae:.1f} RPS)")
        ax.set_xlabel("Target interval (5 seconds)")
        ax.set_ylabel("Requests/sec")
        ax.grid(alpha=0.25)
        ax.legend()
    pd.concat(all_predictions, ignore_index=True).to_csv(OUT / "forecast_results.csv", index=False)
    pd.DataFrame(metrics).to_csv(OUT / "forecast_metrics.csv", index=False)
    fig.suptitle("One-step-ahead forecast on chronological held-out intervals")
    fig.savefig(OUT / "forecast_plot.png", dpi=150)
    plt.close(fig)
    print(pd.DataFrame(metrics).to_string(index=False))
    print(f"Saved forecast files to {OUT}")


if __name__ == "__main__":
    main()
