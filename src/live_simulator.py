"""Continuous laptop simulator for forecast-driven autoscaling."""
from collections import deque
import csv
from pathlib import Path
import threading
import time

import numpy as np

from .scaling_policy import (
    CAPACITY_PER_REPLICA_RPS,
    MAX_REPLICAS,
    MIN_REPLICAS,
    recommend_replicas,
    update_replicas,
)

LAGS = 6
SEASONAL_LAG = 120


class ContinuousSimulator:
    """Replay a workload and scale simulated replicas from one-step forecasts."""

    def __init__(self, workload_path, interval_seconds=5.0, startup_delay_seconds=10.0):
        self.workload_path = Path(workload_path)
        self.interval_seconds = float(interval_seconds)
        self.startup_delay_seconds = float(startup_delay_seconds)
        if self.interval_seconds <= 0 or self.startup_delay_seconds < 0:
            raise ValueError("interval and startup delay must be non-negative, with interval positive")
        self.startup_delay_steps = max(
            1, int(np.ceil(self.startup_delay_seconds / self.interval_seconds))
        )
        self._lock = threading.Lock()
        self._thread = None
        self._stop = threading.Event()
        self._reset_locked()

    def _load(self):
        traces = {}
        with self.workload_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                traces.setdefault(row["scenario"], []).append(
                    float(row["request_rate_rps"]))
        if not traces or any(len(values) < SEASONAL_LAG + 1
                             for values in traces.values()):
            raise ValueError("workload must contain at least 121 points per scenario")
        return {name: np.asarray(values, dtype=float) for name, values in traces.items()}

    def _fit(self, values):
        rows, targets = [], []
        for index in range(SEASONAL_LAG, len(values)):
            rows.append(list(values[index - LAGS:index][::-1])
                        + [values[index - SEASONAL_LAG]])
            targets.append(values[index])
        design = np.column_stack([np.ones(len(rows)), np.asarray(rows)])
        return np.linalg.lstsq(design, np.asarray(targets), rcond=None)[0]

    def _reset_locked(self):
        self.traces = self._load()
        self.scenario = next(iter(self.traces))
        self.position = SEASONAL_LAG
        self.history = deque(
            self.traces[self.scenario][self.position - LAGS:self.position].tolist(),
            maxlen=LAGS,
        )
        self.coefficients = self._fit(self.traces[self.scenario])
        self.provisioned = MIN_REPLICAS
        self.active = MIN_REPLICAS
        self.pending = []
        self.points = deque(maxlen=120)
        self.events = deque(maxlen=40)
        self.started_at = time.time()
        self.last = None
        self.load_control = {
            "mode": "trace",
            "rate_rps": None,
            "started_at": None,
            "ends_at": None,
        }
        self.online_residual = 0.0
        self.last_model_prediction = None
        self.last_adapted_prediction = None

    def set_load(self, mode, rate_rps, duration_seconds):
        if mode not in {"spike", "steady"}:
            raise ValueError("mode must be 'spike' or 'steady'")
        if not isinstance(rate_rps, (int, float)) or isinstance(rate_rps, bool):
            raise ValueError("rate_rps must be numeric")
        if not np.isfinite(rate_rps) or rate_rps < 0 or rate_rps > 600:
            raise ValueError("rate_rps must be between 0 and 600")
        if not isinstance(duration_seconds, (int, float)) or isinstance(duration_seconds, bool):
            raise ValueError("duration_seconds must be numeric")
        if not np.isfinite(duration_seconds) or duration_seconds <= 0 or duration_seconds > 300:
            raise ValueError("duration_seconds must be between 0 and 300")
        now = time.time()
        with self._lock:
            self.load_control = {
                "mode": mode,
                "rate_rps": float(rate_rps),
                "started_at": now,
                "ends_at": now + float(duration_seconds),
            }
            self.events.appendleft({
                "timestamp": now,
                "type": "manual_load",
                "message": f"{mode} load started at {rate_rps:.1f} RPS for {duration_seconds:.1f}s",
            })
            return dict(self.load_control)

    def clear_load(self):
        with self._lock:
            self.load_control = {
                "mode": "trace",
                "rate_rps": None,
                "started_at": None,
                "ends_at": None,
            }
            self.events.appendleft({
                "timestamp": time.time(),
                "type": "manual_load",
                "message": "manual load stopped; replaying workload trace",
            })
            return dict(self.load_control)

    def _controlled_rate_locked(self):
        if self.load_control["mode"] == "trace":
            return None
        if time.time() >= self.load_control["ends_at"]:
            self.load_control = {
                "mode": "trace",
                "rate_rps": None,
                "started_at": None,
                "ends_at": None,
            }
            return None
        return self.load_control["rate_rps"]

    def _step_locked(self):
        values = self.traces[self.scenario]
        if self.position >= len(values):
            self.scenario = next(name for name in self.traces
                                 if name != self.scenario) if len(self.traces) > 1 else self.scenario
            self.position = SEASONAL_LAG
            values = self.traces[self.scenario]
            self.history = deque(
                values[self.position - LAGS:self.position].tolist(), maxlen=LAGS)
            self.coefficients = self._fit(values)
            self.provisioned = MIN_REPLICAS
            self.active = MIN_REPLICAS
            self.pending = []

        recent_history = list(self.history)
        features = np.asarray([1.0, *recent_history[::-1],
                               values[self.position - SEASONAL_LAG]])
        model_prediction = max(0.0, float(features @ self.coefficients))
        recent_prediction = float(recent_history[-1]) if recent_history else 0.0
        adapted_prediction = max(
            0.0,
            model_prediction + self.online_residual,
            recent_prediction,
        )
        predicted = adapted_prediction
        actual = self._controlled_rate_locked()
        if actual is None:
            actual = float(values[self.position])
        target = recommend_replicas(predicted)
        previous = self.provisioned
        desired = update_replicas(previous, target, max_step=1)
        emergency = actual > self.active * CAPACITY_PER_REPLICA_RPS
        if emergency:
            target = recommend_replicas(actual)
            desired = max(previous, target)
        if desired > previous:
            self.pending.append((
                self.position + self.startup_delay_steps,
                desired - previous,
            ))
        elif desired < previous:
            reduction = previous - desired
            remaining = []
            for when, count in self.pending:
                cancelled = min(reduction, count)
                reduction -= cancelled
                count -= cancelled
                if count:
                    remaining.append((when, count))
            self.pending = remaining
            if reduction:
                self.active = max(MIN_REPLICAS, self.active - reduction)
        self.provisioned = desired
        due = sum(count for when, count in self.pending if when <= self.position)
        self.active += due
        self.pending = [(when, count) for when, count in self.pending
                        if when > self.position]
        capacity = self.active * CAPACITY_PER_REPLICA_RPS
        overload = max(0.0, actual - capacity)
        utilization = actual / capacity if capacity else 0.0
        forecast_error = actual - predicted
        self.online_residual = 0.7 * self.online_residual + 0.3 * forecast_error
        self.last_model_prediction = model_prediction
        self.last_adapted_prediction = adapted_prediction
        action = ("emergency_scale_up" if emergency and desired > previous else
                  "scale_up" if desired > previous else
                  "scale_down" if desired < previous else "hold")
        point = {
            "timestamp": time.time(), "scenario": self.scenario,
            "interval": self.position, "actual_rps": actual,
            "predicted_rps": predicted, "target_replicas": target,
            "desired_replicas": self.provisioned, "ready_replicas": self.active,
            "pending_replicas": sum(count for _, count in self.pending),
            "capacity_rps": capacity, "overload_rps": overload,
            "utilization": utilization, "forecast_error_rps": forecast_error,
            "model_prediction_rps": model_prediction,
            "recent_observation_prediction_rps": recent_prediction,
            "online_residual_rps": self.online_residual,
            "startup_delay_seconds": self.startup_delay_seconds,
            "startup_delay_steps": self.startup_delay_steps,
            "decision": {
                "forecast_rps": predicted,
                "lag_features_rps": list(self.history)[::-1],
                "seasonal_lag_rps": float(values[self.position - SEASONAL_LAG]),
                "manual_load_active": self.load_control["mode"] != "trace",
                "target_replicas": target,
                "previous_desired_replicas": previous,
                "new_desired_replicas": self.provisioned,
                "reason": (
                    "observed demand exceeded ready capacity; emergency reactive scale"
                    if emergency else
                    "forecast above capacity target"
                    if desired > previous else
                    "forecast below capacity target"
                    if desired < previous else
                    "forecast within current capacity"
                ),
            },
            "action": action,
        }
        self.history.append(actual)
        self.position += 1
        self.points.append(point)
        self.last = point
        if action != "hold":
            self.events.appendleft({
                "timestamp": point["timestamp"], "type": action,
                "message": f"{self.scenario}: {predicted:.1f} RPS -> {self.provisioned} replicas",
            })
        return point

    def _run(self):
        while not self._stop.wait(self.interval_seconds):
            with self._lock:
                self._step_locked()

    def start(self):
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1))

    def snapshot(self):
        with self._lock:
            return {
                "running": bool(self._thread and self._thread.is_alive()),
                "interval_seconds": self.interval_seconds,
                "startup_delay_seconds": self.startup_delay_seconds,
                "startup_delay_steps": self.startup_delay_steps,
                "last": self.last,
                "points": list(self.points),
                "events": list(self.events),
                "load_control": dict(self.load_control),
            }
