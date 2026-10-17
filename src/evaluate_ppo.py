"""Evaluate the saved PPO policy on intervals 360-479 of each scenario."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from .ppo_agent import PPOAgent
from .ppo_env import AutoscalingEnv, CAPACITY_PER_REPLICA_RPS
from .train_ppo import ROOT


def load_traces(path):
    traces = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            traces.setdefault(row["scenario"], []).append(float(row["request_rate_rps"]))
    return {k: np.asarray(v, dtype=np.float32) for k, v in traces.items()}


def run(model=ROOT / "models/ppo_autoscaler.npz", workload=ROOT / "data/workload.csv"):
    agent = PPOAgent.load(model)
    traces = load_traces(workload)
    rows, summaries = [], []
    for scenario, trace in traces.items():
        env = AutoscalingEnv({scenario: trace})
        obs = env.reset(scenario, start=360)
        while True:
            action, _, _ = agent.act(obs, deterministic=True)
            obs, _, done, _, info = env.step(action)
            rows.append(info)
            if done: break
        part = [r for r in rows if r["scenario"] == scenario]
        rates = [r["overload_rps"] for r in part]
        summaries.append({"scenario": scenario, "policy": "ppo", "intervals": len(part),
            "overload_intervals": sum(x > 0 for x in rates),
            "overload_duration_seconds": sum(x > 0 for x in rates) * 5,
            "overload_rps_interval_sum": sum(rates),
            "estimated_excess_requests": sum(rates) * 5,
            "mean_active_replicas": float(np.mean([r["active_replicas"] for r in part])),
            "scale_actions": sum(r["action_applied"] for r in part),
            "max_active_replicas": max(r["active_replicas"] for r in part)})
    outdir = ROOT / "results"; outdir.mkdir(exist_ok=True)
    with (outdir / "ppo_scaling_intervals.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (outdir / "ppo_scaling_comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))
    return summaries


if __name__ == "__main__": run()
