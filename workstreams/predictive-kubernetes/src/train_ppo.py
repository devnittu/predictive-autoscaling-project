"""Train PPO on the pre-test portion of the deterministic workload traces."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .ppo_agent import PPOAgent
from .ppo_env import (AutoscalingEnv, OBSERVATION_SIZE, CAPACITY_PER_REPLICA_RPS,
                      MIN_REPLICAS, MAX_REPLICAS, make_observation)

ROOT = Path(__file__).resolve().parents[1]


def read_traces(path, end=360):
    traces = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            i = int(row["interval"])
            if i < end:
                traces.setdefault(row["scenario"], []).append(float(row["request_rate_rps"]))
    return {k: np.asarray(v, dtype=np.float32) for k, v in traces.items()}


def train(steps=30000, seed=17, workload=ROOT / "workload.csv", out=ROOT / "models/ppo_autoscaler.npz"):
    traces = read_traces(workload)
    env = AutoscalingEnv(traces, seed=seed)
    rng = np.random.default_rng(seed)
    agent = PPOAgent(OBSERVATION_SIZE, seed=seed)
    demo_obs, demo_actions = [], []
    for scenario in traces:
        teacher_env = AutoscalingEnv({scenario: traces[scenario]}, seed=seed)
        observation = teacher_env.reset(scenario)
        while True:
            target = int(np.clip(np.ceil(teacher_env.history[-1] / CAPACITY_PER_REPLICA_RPS), MIN_REPLICAS, MAX_REPLICAS))
            current = teacher_env.provisioned
            action = 2 if target > current else 0 if target < current else 1
            demo_obs.append(observation.copy()); demo_actions.append(action)
            observation, _, done, _, _ = teacher_env.step(action)
            if done: break
        # Train across valid controller states too: a high held-out demand can
        # arrive while the deployment is still at its common initial size of one.
        trace = traces[scenario]
        for i in range(6, len(trace)):
            recent = trace[i-6:i]
            target = int(np.clip(np.ceil(recent[-1] / CAPACITY_PER_REPLICA_RPS), MIN_REPLICAS, MAX_REPLICAS))
            for current in range(MIN_REPLICAS, MAX_REPLICAS + 1):
                demo_obs.append(make_observation(recent, current, 0))
                demo_actions.append(2 if target > current else 0 if target < current else 1)
    agent.behavior_clone(demo_obs, demo_actions, epochs=40)
    print(f"PPO actor initialized from {len(demo_actions)} observation-only capacity-rule examples")
    obs = env.reset()
    rewards_log = []
    collected = 0
    while collected < steps:
        batch = {k: [] for k in ("obs", "actions", "logp", "values", "rewards", "dones")}
        for _ in range(min(256, steps - collected)):
            action, logp, value = agent.act(obs, rng=rng)
            next_obs, reward, done, _, _ = env.step(action)
            for k, v in (("obs", obs.copy()), ("actions", action), ("logp", logp),
                         ("values", value), ("rewards", reward), ("dones", done)):
                batch[k].append(v)
            obs = env.reset() if done else next_obs
            collected += 1
        # GAE is computed backward. Terminated episodes do not bootstrap; a
        # rollout that ends mid-episode bootstraps from its final observation.
        adv = np.zeros(len(batch["rewards"]), dtype=np.float32)
        last = 0.0
        for t in reversed(range(len(adv))):
            mask = 0.0 if batch["dones"][t] else 1.0
            if batch["dones"][t]:
                nv = 0.0
            elif t + 1 < len(adv):
                nv = float(agent.forward(batch["obs"][t + 1])[2])
            else:
                nv = float(agent.forward(next_obs)[2])
            delta = batch["rewards"][t] + .99 * nv * mask - batch["values"][t]
            last = delta + .99 * .95 * mask * last
            adv[t] = last
        advantages = (adv - adv.mean()) / (adv.std() + 1e-8)
        returns = adv + np.asarray(batch["values"], dtype=np.float32)
        for _ in range(4):
            stats = agent.update(np.asarray(batch["obs"], dtype=np.float32), np.asarray(batch["actions"], dtype=int),
                                 np.asarray(batch["logp"], dtype=np.float32), advantages, returns)
        rewards_log.append(float(np.mean(batch["rewards"])))
        if len(rewards_log) % 20 == 0:
            print(f"steps={collected} recent_mean_reward={np.mean(rewards_log[-20:]):.3f} entropy={stats['mean_entropy']:.3f}")
    out.parent.mkdir(parents=True, exist_ok=True)
    agent.save(out)
    # Keep six real pre-test samples for the live controller's startup history.
    full = {}
    with open(workload, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["interval"]) < 360:
                full.setdefault(row["scenario"], []).append(float(row["request_rate_rps"]))
    history = {k: v[-6:] for k, v in full.items()}
    history_path = out.parent / "seed_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Saved PPO policy: {out}")
    print(f"Saved live controller seed history: {history_path}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=5120)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train(args.steps, args.seed)
