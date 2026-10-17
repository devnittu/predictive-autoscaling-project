"""Small Gym-style autoscaling environment used to train the PPO controller.

The agent observes only completed demand intervals. It chooses scale down, hold,
or scale up for the next interval; newly requested replicas start after the
configured delay. Workload values are requests per second (RPS).
"""
from collections import deque
import numpy as np

WINDOW_SECONDS = 5
CAPACITY_PER_REPLICA_RPS = 60.0
MIN_REPLICAS = 1
MAX_REPLICAS = 10
ACTION_NAMES = ("scale_down", "hold", "scale_up")
OBSERVATION_SIZE = 8


def normalize_rate(rate):
    return float(np.log1p(max(0.0, float(rate))) / np.log1p(250.0))


def make_observation(recent_rates, provisioned_replicas, pending_replicas):
    """Encode latest six completed RPS samples plus controller replica state."""
    rates = list(recent_rates)[-6:]
    rates = [0.0] * (6 - len(rates)) + rates
    values = [normalize_rate(x) for x in rates]
    values += [provisioned_replicas / MAX_REPLICAS,
               pending_replicas / MAX_REPLICAS]
    return np.asarray(values, dtype=np.float32)


class AutoscalingEnv:
    """Single workload episode; reset with a mapping of named RPS traces."""

    def __init__(self, traces, startup_delay_steps=1, seed=42):
        if not traces or any(len(trace) < 8 for trace in traces.values()):
            raise ValueError("each workload trace must contain at least eight intervals")
        if startup_delay_steps < 0:
            raise ValueError("startup delay must be non-negative")
        self.traces = {name: np.asarray(trace, dtype=np.float32) for name, trace in traces.items()}
        if any(not np.isfinite(x).all() or (x < 0).any() for x in self.traces.values()):
            raise ValueError("workload rates must be finite and non-negative")
        self.startup_delay_steps = int(startup_delay_steps)
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, scenario=None, start=6):
        if scenario is None:
            scenario = self.rng.choice(list(self.traces))
        if scenario not in self.traces:
            raise ValueError(f"unknown workload scenario: {scenario}")
        trace = self.traces[scenario]
        if start < 6 or start >= len(trace):
            raise ValueError("start must leave six prior samples and at least one target")
        self.scenario = scenario
        self.trace = trace
        self.position = int(start)
        self.history = deque(trace[start - 6:start].astype(float).tolist(), maxlen=6)
        self.provisioned = 1
        self.active = 1
        self.pending = []
        self.episode_return = 0.0
        self.overload_rps_sum = 0.0
        self.replica_intervals = 0
        return make_observation(self.history, self.provisioned, 0)

    def _observation(self):
        return make_observation(self.history, self.provisioned,
                                sum(count for _, count in self.pending))

    def step(self, action):
        if action not in (0, 1, 2):
            raise ValueError("action must be 0 (down), 1 (hold), or 2 (up)")
        if self.position >= len(self.trace):
            raise RuntimeError("episode is finished; call reset")

        due = sum(count for when, count in self.pending if when <= self.position)
        self.active += due
        self.pending = [(when, count) for when, count in self.pending if when > self.position]

        changed = False
        if action == 2 and self.provisioned < MAX_REPLICAS:
            self.provisioned += 1
            changed = True
            if self.startup_delay_steps == 0:
                self.active += 1
            else:
                self.pending.append((self.position + self.startup_delay_steps, 1))
        elif action == 0 and self.provisioned > MIN_REPLICAS:
            self.provisioned -= 1
            changed = True
            reduction = 1
            remaining = []
            for when, count in self.pending:
                cancelled = min(count, reduction)
                count -= cancelled
                reduction -= cancelled
                if count:
                    remaining.append((when, count))
            self.pending = remaining
            if reduction:
                self.active = max(MIN_REPLICAS, self.active - reduction)

        demand = float(self.trace[self.position])
        capacity = self.active * CAPACITY_PER_REPLICA_RPS
        overload = max(0.0, demand - capacity)
        # Penalize unmet rate and reserved capacity while charging a small action cost.
        reward = -(2.0 * overload / CAPACITY_PER_REPLICA_RPS
                   + 0.025 * self.active + (0.015 if changed else 0.0))
        self.episode_return += reward
        self.overload_rps_sum += overload
        self.replica_intervals += self.active
        self.history.append(demand)
        self.position += 1
        terminated = self.position >= len(self.trace)
        info = {
            "scenario": self.scenario,
            "interval": self.position - 1,
            "demand_rps": demand,
            "active_replicas": self.active,
            "provisioned_replicas": self.provisioned,
            "pending_replicas": sum(count for _, count in self.pending),
            "capacity_rps": capacity,
            "overload_rps": overload,
            "action_name": ACTION_NAMES[action],
            "action_applied": changed,
        }
        return self._observation(), float(reward), terminated, False, info
