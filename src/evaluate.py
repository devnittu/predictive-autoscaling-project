"""Compare reactive and forecast policies in an explicit discrete-time simulation."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.scaling_policy import recommend_replicas, update_replicas
from src.workload_generator import INTERVAL_SECONDS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "workload.csv"
FORECASTS = ROOT / "results" / "forecast_results.csv"
METRICS = ROOT / "results" / "forecast_metrics.csv"
OUT = ROOT / "results"

# All delays are discrete 5-second intervals. A forecast for t is available
# before interval t begins; reactive sensing sees demand t-1 by default.
CAPACITY_PER_REPLICA_RPS = 60.0
MIN_REPLICAS, MAX_REPLICAS = 1, 10
INITIAL_REPLICAS = 1
MAX_SCALE_STEP = 1
OBSERVATION_DELAY_INTERVALS = 1
STARTUP_DELAY_INTERVALS = 1
COOLDOWN_INTERVALS = 0


def simulate_policy(group, policy, observation_delay=OBSERVATION_DELAY_INTERVALS,
                    startup_delay=STARTUP_DELAY_INTERVALS,
                    cooldown=COOLDOWN_INTERVALS, demand_history=None):
    """Simulate one policy; missing forecasts hold its replica target unchanged."""
    if policy not in {"reactive", "predictive"}:
        raise ValueError("policy must be 'reactive' or 'predictive'")
    if min(observation_delay, startup_delay, cooldown) < 0:
        raise ValueError("delay and cooldown values must be non-negative")
    required = {"scenario", "interval", "elapsed_seconds", "actual_rps", "predicted_rps"}
    if not required.issubset(group.columns):
        raise ValueError(f"simulation input missing columns: {sorted(required - set(group.columns))}")
    if group.duplicated("interval").any():
        raise ValueError("simulation intervals must be unique")
    intervals = pd.to_numeric(group.interval, errors="coerce").to_numpy(dtype=float)
    elapsed = pd.to_numeric(group.elapsed_seconds, errors="coerce").to_numpy(dtype=float)
    actual_values = pd.to_numeric(group.actual_rps, errors="coerce").to_numpy(dtype=float)
    forecast_values = pd.to_numeric(group.predicted_rps, errors="coerce").to_numpy(dtype=float)
    if (not np.isfinite(intervals).all() or not np.equal(intervals, np.floor(intervals)).all()
            or not np.isfinite(elapsed).all()
            or not np.array_equal(elapsed, intervals * INTERVAL_SECONDS)):
        raise ValueError("simulation intervals and elapsed_seconds must be finite and aligned")
    if not np.isfinite(actual_values).all() or (actual_values < 0).any():
        raise ValueError("actual request rates must be finite and non-negative")
    malformed_forecasts = group.predicted_rps.notna().to_numpy() & np.isnan(forecast_values)
    if (malformed_forecasts.any() or np.isinf(forecast_values).any()
            or (forecast_values[~np.isnan(forecast_values)] < 0).any()):
        raise ValueError("forecasts must be missing or finite and non-negative")
    group = group.sort_values("interval").reset_index(drop=True)
    provisioned = INITIAL_REPLICAS
    active = INITIAL_REPLICAS
    pending = []
    last_action = -10**9
    rows = []
    local_history = group.set_index("interval").actual_rps.to_dict()
    history = {**local_history, **(demand_history or {})}
    for pos, record in enumerate(group.itertuples(index=False)):
        interval = int(record.interval)
        due_now = sum(count for due, count in pending if due <= interval)
        active += due_now
        pending = [(due, count) for due, count in pending if due > interval]

        if policy == "reactive":
            observed_interval = interval - observation_delay
            if observed_interval not in history:
                raise ValueError(f"Demand history is missing interval {observed_interval}")
            signal = float(history[observed_interval])
            signal_label = "observed_demand"
        else:
            raw = record.predicted_rps
            signal = None if pd.isna(raw) else float(raw)
            signal_label = "forecast"

        action = "hold"
        target = provisioned
        if signal is not None and interval - last_action >= cooldown:
            desired = recommend_replicas(signal, CAPACITY_PER_REPLICA_RPS, MIN_REPLICAS, MAX_REPLICAS)
            target = update_replicas(provisioned, desired, MAX_SCALE_STEP)
            if target != provisioned:
                action = "scale_up" if target > provisioned else "scale_down"
                delta = target - provisioned
                if delta > 0:
                    if startup_delay == 0:
                        active += delta
                    else:
                        pending.append((interval + startup_delay, delta))
                else:
                    # Cancel replicas that have not started before retiring active ones.
                    reduction = -delta
                    updated_pending = []
                    for due, count in pending:
                        cancelled = min(count, reduction)
                        count -= cancelled
                        reduction -= cancelled
                        if count:
                            updated_pending.append((due, count))
                    pending = updated_pending
                    active = max(MIN_REPLICAS, active - reduction)
                provisioned = target
                last_action = interval

        actual = float(record.actual_rps)
        capacity = active * CAPACITY_PER_REPLICA_RPS
        rows.append({
            "scenario": record.scenario, "policy": policy, "interval": interval,
            "decision_interval": interval,
            "elapsed_seconds": int(record.elapsed_seconds), "actual_rps": actual,
            "predicted_rps": record.predicted_rps, "scaling_signal": signal_label,
            "scaling_signal_rps": signal,
            "observation_interval": observed_interval if policy == "reactive" else interval - 1,
            "scaling_action": action, "target_replicas": target,
            "provisioned_replicas": provisioned, "active_replicas": active,
            "pending_replicas": sum(count for _, count in pending),
            "assumed_capacity_rps": capacity,
            "estimated_utilization": actual / capacity,
            "estimated_overload_rps": max(0.0, actual - capacity),
        })
    return rows


def summarize_scaling(result):
    summaries = []
    for (scenario, policy), group in result.groupby(["scenario", "policy"], sort=True):
        overload = group.estimated_overload_rps.to_numpy()
        episodes = int(((overload > 0) & (pd.Series(overload).shift(fill_value=0) == 0)).sum())
        recovery_times = []
        starts = [i for i, value in enumerate(overload)
                  if value > 0 and (i == 0 or overload[i - 1] <= 0)]
        for start in starts:
            end = next((i for i in range(start + 1, len(overload)) if overload[i] <= 0), None)
            if end is not None:
                recovery_times.append((end - start) * INTERVAL_SECONDS)
        summaries.append({
            "scenario": scenario, "policy": policy, "test_intervals": len(group),
            "mean_active_replicas": group.active_replicas.mean(),
            "replica_intervals": group.active_replicas.sum(),
            "scale_up_actions": int((group.scaling_action == "scale_up").sum()),
            "scale_down_actions": int((group.scaling_action == "scale_down").sum()),
            "overload_intervals": int((overload > 0).sum()),
            "overload_duration_seconds": int((overload > 0).sum() * INTERVAL_SECONDS),
            "sum_overload_rps_intervals": overload.sum(),
            "estimated_excess_requests": overload.sum() * INTERVAL_SECONDS,
            "max_overload_rps": overload.max(), "overload_episodes": episodes,
            "mean_overload_recovery_seconds": (sum(recovery_times) / len(recovery_times)
                                                if recovery_times else float("nan")),
            "mean_estimated_utilization": group.estimated_utilization.mean(),
        })
    summary = pd.DataFrame(summaries)
    numeric = summary.select_dtypes("number").columns
    summary[numeric] = summary[numeric].round(3)
    return summary


def main():
    for path in (DATA, FORECASTS, METRICS):
        if not path.exists():
            raise SystemExit(f"Missing {path.relative_to(ROOT)}. Run the complete pipeline.")
    workload, forecasts, metrics = (pd.read_csv(p) for p in (DATA, FORECASTS, METRICS))
    expected = {"scenario", "interval", "elapsed_seconds", "request_rate_rps"}
    if not expected.issubset(workload.columns):
        raise SystemExit(f"workload.csv missing columns: {sorted(expected - set(workload.columns))}")
    if workload.duplicated(["scenario", "interval"]).any():
        raise SystemExit("workload.csv has duplicate scenario/interval keys")
    workload_intervals = pd.to_numeric(workload.interval, errors="coerce").to_numpy()
    if not np.isfinite(workload_intervals).all() or not np.equal(workload_intervals, np.floor(workload_intervals)).all():
        raise SystemExit("workload.csv intervals must be finite integers")
    if not np.isfinite(pd.to_numeric(workload.request_rate_rps, errors="coerce")).all():
        raise SystemExit("workload.csv contains non-finite request rates")
    for scenario, subset in workload.groupby("scenario"):
        intervals = subset.interval.astype(int).sort_values().to_numpy()
        if not np.array_equal(intervals, np.arange(len(intervals))):
            raise SystemExit(f"{scenario}: workload intervals must be contiguous from zero")
        elapsed = pd.to_numeric(subset.sort_values("interval").elapsed_seconds, errors="coerce").to_numpy()
        if not np.array_equal(elapsed, intervals * INTERVAL_SECONDS):
            raise SystemExit(f"{scenario}: elapsed_seconds do not match interval spacing")
    if not {"scenario", "interval", "forecast_origin_interval", "actual_rps", "predicted_rps"}.issubset(forecasts.columns):
        raise SystemExit("forecast_results.csv requires scenario, interval, forecast_origin_interval, actual_rps, predicted_rps")
    if not {"scenario", "test_first_interval"}.issubset(metrics.columns):
        raise SystemExit("forecast_metrics.csv requires scenario, test_first_interval")
    if forecasts.duplicated(["scenario", "interval"]).any():
        raise SystemExit("Duplicate forecast scenario/interval keys")
    forecast_targets = pd.to_numeric(forecasts.interval, errors="coerce")
    forecast_origins = pd.to_numeric(forecasts.forecast_origin_interval, errors="coerce")  
    if forecast_targets.isna().any() or forecast_origins.isna().any():
        raise SystemExit("Forecast interval and origin values must be numeric")
    if (forecast_origins != forecast_targets - 1).any():
        raise SystemExit("Forecasts must use only the immediately preceding interval")
    parsed_predictions = pd.to_numeric(forecasts.predicted_rps, errors="coerce")
    if (forecasts.predicted_rps.notna() & parsed_predictions.isna()).any() or np.isinf(parsed_predictions).any():
        raise SystemExit("Forecasts contain malformed or infinite values")
    if (parsed_predictions.dropna() < 0).any():
        raise SystemExit("Forecast request rates cannot be negative")
    if metrics.scenario.duplicated().any() or set(metrics.scenario) != set(workload.scenario):
        raise SystemExit("Forecast metrics must contain exactly one row per workload scenario")

    actual_lookup = workload.set_index(["scenario", "interval"]).request_rate_rps
    for forecast in forecasts.itertuples(index=False):
        key = (forecast.scenario, int(forecast.interval))
        if key not in actual_lookup.index:
            raise SystemExit(f"Forecast interval is absent from workload: {key}")
        if not np.isclose(float(forecast.actual_rps), float(actual_lookup.loc[key])):
            raise SystemExit(f"Forecast actual does not match workload at {key}")

    rows = []
    for metric in metrics.itertuples(index=False):
        scenario = metric.scenario
        first = int(metric.test_first_interval)
        scenario_workload = workload[workload.scenario == scenario]
        if first <= int(scenario_workload.interval.min()) or first > int(scenario_workload.interval.max()):
            raise SystemExit(f"Invalid held-out boundary for {scenario}: {first}")
        if hasattr(metric, "train_last_interval") and int(metric.train_last_interval) != first - 1:
            raise SystemExit(f"Inconsistent chronological split metadata for {scenario}")
        scenario_forecasts = forecasts[forecasts.scenario == scenario]
        if (scenario_forecasts.interval.astype(int) < first).any():
            raise SystemExit(f"Forecast file contains training intervals for {scenario}")
        test = workload[(workload.scenario == scenario) & (workload.interval >= first)].copy()
        test = test.merge(scenario_forecasts[["scenario", "interval", "predicted_rps"]],
                          on=["scenario", "interval"], how="left", validate="one_to_one")
        if test.empty:
            raise SystemExit(f"No held-out workload rows for {scenario}")
        history = workload[workload.scenario == scenario].set_index("interval").request_rate_rps.to_dict()
        for policy in ("reactive", "predictive"):
            rows.extend(simulate_policy(test.rename(columns={"request_rate_rps": "actual_rps"}), policy,
                                        demand_history=history))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "scaling_comparison.csv", index=False)

    summary = summarize_scaling(result)
    summary.to_csv(OUT / "scaling_summary.csv", index=False)

    scenarios = sorted(result.scenario.unique())
    fig, axes = plt.subplots(len(scenarios), 2, figsize=(15, 3.4 * len(scenarios)), constrained_layout=True)
    if len(scenarios) == 1: 
        axes = np.array([axes])
    for row, scenario in zip(axes, scenarios):
        demand_ax, replica_ax = row
        subset = result[result.scenario == scenario]
        actual = subset.drop_duplicates("interval").sort_values("interval")
        demand_ax.plot(actual.interval, actual.actual_rps, color="black", linewidth=1.1,
                       label="Actual workload")
        for policy, label in (("reactive", "Reactive"), ("predictive", "Forecast based")):
            curve = subset[subset.policy == policy]
            demand_ax.step(curve.interval, curve.assumed_capacity_rps, where="post",
                           label=f"{label} capacity")
            replica_ax.step(curve.interval, curve.active_replicas, where="post", label=label)
        demand_ax.set_title(f"{scenario.title()}: workload and capacity")
        demand_ax.set_ylabel("Requests per second")
        demand_ax.grid(alpha=0.25)
        demand_ax.legend()
        replica_ax.set_title(f"{scenario.title()}: active replicas")
        replica_ax.set_ylabel("Active replicas")
        replica_ax.grid(alpha=0.25)
        replica_ax.legend()
    axes[-1, 0].set_xlabel("Held-out target interval (5 seconds)")
    axes[-1, 1].set_xlabel("Held-out target interval (5 seconds)")
    fig.suptitle("Workload, assumed capacity and active replicas in the discrete-time simulation")
    fig.savefig(OUT / "scaling_comparison.png", dpi=150)
    plt.close(fig)

    with (OUT / "summary.txt").open("w", encoding="utf-8") as f:
        f.write("SIMULATED SCALING COMPARISON\n")
        f.write("Simulation estimates only; not measured Kubernetes latency or Prometheus metrics.\n")
        f.write(f"Capacity: {CAPACITY_PER_REPLICA_RPS:g} RPS/replica; observation delay: {OBSERVATION_DELAY_INTERVALS * INTERVAL_SECONDS}s; startup delay: {STARTUP_DELAY_INTERVALS * INTERVAL_SECONDS}s.\n")
        f.write("Reactive uses delayed observed demand; predictive uses the one-step-ahead forecast for the target interval. Missing forecasts hold the current target.\n\n")
        f.write("Overload duration is overload intervals multiplied by interval seconds. `sum_overload_rps_intervals` is RPS-intervals; `estimated_excess_requests` multiplies this by seconds per interval and assumes the sampled RPS holds throughout each interval.\n\n")
        f.write(summary.to_string(index=False) + "\n")
    print(summary.to_string(index=False))
    print(f"Saved outputs under {OUT}")


if __name__ == "__main__":
    main()
