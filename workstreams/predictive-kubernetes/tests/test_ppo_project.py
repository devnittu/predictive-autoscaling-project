import unittest
import numpy as np
from src.ppo_env import AutoscalingEnv, make_observation, CAPACITY_PER_REPLICA_RPS
from src.ppo_agent import PPOAgent
from src.train_ppo import read_traces


class PPOProjectTests(unittest.TestCase):
    def test_observation_contains_only_passed_history_and_controller_state(self):
        observed = make_observation([10, 20, 30, 40, 50, 60], 2, 1)
        changed_future = make_observation([10, 20, 30, 40, 50, 60], 2, 1)
        np.testing.assert_array_equal(observed, changed_future)
        self.assertEqual(observed.shape, (8,))

    def test_future_target_not_exposed_before_step(self):
        trace = np.asarray([10] * 6 + [200, 20], dtype=np.float32)
        env = AutoscalingEnv({"test": trace})
        obs = env.reset("test", start=6)
        np.testing.assert_array_equal(obs, make_observation(trace[:6], 1, 0))
        _, _, _, _, info = env.step(1)
        self.assertEqual(info["demand_rps"], 200)
        # Target becomes available only in the observation returned after serving it.
        self.assertEqual(env.history[-1], 200)

    def test_startup_delay_and_replica_bounds(self):
        env = AutoscalingEnv({"x": [50] * 30}, startup_delay_steps=1)
        env.reset("x", start=6)
        env.step(2)
        self.assertEqual(env.active, 1)
        self.assertEqual(env.provisioned, 2)
        env.step(1)
        self.assertEqual(env.active, 2)
        for _ in range(8): env.step(2)
        self.assertLessEqual(env.provisioned, 10)

    def test_overload_is_excess_rate_not_interval_count(self):
        env = AutoscalingEnv({"x": [50] * 6 + [155, 50]})
        env.reset("x", start=6)
        _, reward, _, _, info = env.step(1)
        self.assertEqual(info["capacity_rps"], CAPACITY_PER_REPLICA_RPS)
        self.assertEqual(info["overload_rps"], 95)
        self.assertAlmostEqual(reward, -(2 * 95 / 60 + .025))

    def test_ppo_policy_roundtrip_and_update(self):
        agent = PPOAgent(seed=1)
        obs = np.zeros((4, 8), dtype=np.float32)
        actions = np.asarray([0, 1, 2, 1])
        _, probs, values = agent.forward(obs)
        old_logp = np.log(probs[np.arange(4), actions] + 1e-8)
        agent.update(obs, actions, old_logp, np.ones(4), values + 1)
        self.assertTrue(np.isfinite(agent.forward(obs)[1]).all())

    def test_training_reader_excludes_held_out_targets(self):
        from pathlib import Path
        workload = Path(__file__).resolve().parents[1] / "workload.csv"
        traces = read_traces(workload, end=360)
        self.assertEqual(set(traces), {"steady", "spike", "ramp", "periodic"})
        self.assertTrue(all(len(trace) == 360 for trace in traces.values()))



if __name__ == "__main__": unittest.main()
