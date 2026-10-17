"""A small NumPy implementation of PPO for discrete autoscaling actions."""
from __future__ import annotations

import numpy as np


class PPOAgent:
    def __init__(self, obs_size=8, hidden=64, seed=7):
        rng = np.random.default_rng(seed)
        self.params = {
            "w1": rng.normal(0, np.sqrt(2 / obs_size), (obs_size, hidden)),
            "b1": np.zeros(hidden), "wa": rng.normal(0, .01, (hidden, 3)),
            "ba": np.zeros(3), "wv": rng.normal(0, .01, (hidden, 1)),
            "bv": np.zeros(1),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.opt_step = 0

    def forward(self, obs):
        p = self.params
        h = np.tanh(np.asarray(obs) @ p["w1"] + p["b1"])
        logits = h @ p["wa"] + p["ba"]
        logits -= logits.max(axis=-1, keepdims=True)
        probs = np.exp(logits); probs /= probs.sum(axis=-1, keepdims=True)
        value = (h @ p["wv"] + p["bv"]).squeeze(-1)
        return h, probs, value

    def act(self, obs, deterministic=False, rng=None):
        _, probs, value = self.forward(obs)
        if deterministic:
            action = int(np.argmax(probs))
        else:
            action = int((rng or np.random.default_rng()).choice(3, p=probs))
        return action, float(np.log(probs[action] + 1e-8)), float(value)

    def update(self, obs, actions, old_logp, advantages, returns, *, clip=.2,
               value_coef=.5, entropy_coef=.01, learning_rate=3e-4):
        """One full-batch clipped PPO/critic update with Adam and analytic gradients."""
        p = self.params
        h, probs, values = self.forward(obs)
        n = len(actions)
        probs_a = probs[np.arange(n), actions]
        ratio = np.exp(np.log(probs_a + 1e-8) - old_logp)
        active = ((advantages >= 0) & (ratio <= 1 + clip)) | ((advantages < 0) & (ratio >= 1 - clip))
        dlogits = -advantages[:, None] * ratio[:, None] * active[:, None] / n * (-probs)
        dlogits[np.arange(n), actions] += -advantages * ratio * active / n
        dlogits += entropy_coef * probs * (np.log(probs + 1e-8) + (-np.sum(probs * np.log(probs + 1e-8), axis=1, keepdims=True))) / n
        dv = (2 * value_coef * (values - returns) / n)[:, None]
        grads = {
            "wa": h.T @ dlogits, "ba": dlogits.sum(axis=0),
            "wv": h.T @ dv, "bv": dv.sum(axis=0),
        }
        dh = dlogits @ p["wa"].T + dv @ p["wv"].T
        dz = dh * (1 - h * h)
        grads["w1"] = np.asarray(obs).T @ dz
        grads["b1"] = dz.sum(axis=0)
        norm = np.sqrt(sum(np.sum(g*g) for g in grads.values()))
        if norm > 1.0:
            grads = {k: g / norm for k, g in grads.items()}
        self.opt_step += 1
        for k, g in grads.items():
            self.m[k] = .9*self.m[k] + .1*g
            self.v[k] = .999*self.v[k] + .001*g*g
            mh = self.m[k] / (1 - .9**self.opt_step)
            vh = self.v[k] / (1 - .999**self.opt_step)
            p[k] -= learning_rate * mh / (np.sqrt(vh) + 1e-8)
        return {"mean_value": float(values.mean()), "mean_entropy": float((-probs*np.log(probs+1e-8)).sum(axis=1).mean())}

    def behavior_clone(self, obs, actions, epochs=40, learning_rate=.003):
        """Initialize the actor from an observation-only capacity rule before PPO."""
        p = self.params
        x = np.asarray(obs, dtype=np.float32)
        y = np.asarray(actions, dtype=int)
        counts = np.bincount(y, minlength=3).astype(float)
        sample_weight = len(y) / (3 * np.maximum(1.0, counts[y]))
        rng = np.random.default_rng(123)
        for _ in range(epochs):
            for ids in np.array_split(rng.permutation(len(y)), max(1, int(np.ceil(len(y)/128)))):
                xb, yb = x[ids], y[ids]
                h, probs, _ = self.forward(xb)
                d = probs.copy()
                d[np.arange(len(yb)), yb] -= 1
                d *= sample_weight[ids, None] / len(yb)
                grads = {"wa": h.T @ d, "ba": d.sum(axis=0)}
                dh = d @ p["wa"].T
                dz = dh * (1-h*h)
                grads["w1"] = xb.T @ dz
                grads["b1"] = dz.sum(axis=0)
                self.opt_step += 1
                for k, g in grads.items():
                    self.m[k] = .9*self.m[k] + .1*g
                    self.v[k] = .999*self.v[k] + .001*g*g
                    mh = self.m[k] / (1-.9**self.opt_step)
                    vh = self.v[k] / (1-.999**self.opt_step)
                    p[k] -= learning_rate * mh / (np.sqrt(vh)+1e-8)

    def save(self, path):
        np.savez(path, **self.params)

    @classmethod
    def load(cls, path):
        agent = cls()
        with np.load(path) as data:
            for k in agent.params:
                agent.params[k] = data[k].copy()
        return agent

