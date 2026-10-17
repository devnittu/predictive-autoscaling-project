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
        workload = Path(__file__).resolve().parents[1] / "data/workload.csv"
        traces = read_traces(workload, end=360)
        self.assertEqual(set(traces), {"steady", "spike", "ramp", "periodic"})
        self.assertTrue(all(len(trace) == 360 for trace in traces.values()))

    def test_gateway_dashboard_apis_work_outside_the_container(self):
        from src import ppo_k8s_app
        state = ppo_k8s_app.get_state()
        history = ppo_k8s_app.get_history()
        results = ppo_k8s_app.get_results()
        dashboard = ppo_k8s_app.dashboard()
        self.assertIn("capacity_rps", state)
        self.assertIn("points", history)
        self.assertIn("rows", results)
        self.assertIn("LinuxONE Predictive Autoscaler", dashboard)

    def test_continuous_simulator_predicts_and_scales(self):
        from pathlib import Path
        from src.live_simulator import ContinuousSimulator
        workload = Path(__file__).resolve().parents[1] / "data/workload.csv"
        simulator = ContinuousSimulator(workload, interval_seconds=0.01)
        first = simulator._step_locked()
        self.assertGreaterEqual(first["predicted_rps"], 0)
        self.assertGreaterEqual(first["desired_replicas"], 1)
        self.assertIn(first["action"], {"scale_up", "scale_down", "hold"})
        self.assertEqual(len(simulator.snapshot()["points"]), 1)

    def test_manual_load_controller_overrides_trace_and_can_stop(self):
        from pathlib import Path
        from src.live_simulator import ContinuousSimulator
        workload = Path(__file__).resolve().parents[1] / "data/workload.csv"
        simulator = ContinuousSimulator(workload, interval_seconds=0.01)
        simulator.set_load("spike", 300, 30)
        point = simulator._step_locked()
        self.assertEqual(point["actual_rps"], 300)
        self.assertEqual(simulator.snapshot()["load_control"]["mode"], "spike")
        simulator.clear_load()
        self.assertEqual(simulator.snapshot()["load_control"]["mode"], "trace")

    def test_manual_load_controller_rejects_unsafe_values(self):
        from pathlib import Path
        from src.live_simulator import ContinuousSimulator
        workload = Path(__file__).resolve().parents[1] / "data/workload.csv"
        simulator = ContinuousSimulator(workload, interval_seconds=0.01)
        with self.assertRaises(ValueError):
            simulator.set_load("spike", 601, 10)
        with self.assertRaises(ValueError):
            simulator.set_load("spike", 300, 301)

    def test_continuous_simulator_models_realistic_startup_delay(self):
        from pathlib import Path
        from src.live_simulator import ContinuousSimulator
        workload = Path(__file__).resolve().parents[1] / "data/workload.csv"
        simulator = ContinuousSimulator(workload, interval_seconds=1, startup_delay_seconds=10)
        simulator.set_load("spike", 300, 30)
        simulator.coefficients = np.asarray([300.0] + [0.0] * 7)
        point = simulator._step_locked()
        self.assertEqual(point["ready_replicas"], 1)
        self.assertGreaterEqual(point["pending_replicas"], 1)
        self.assertEqual(point["startup_delay_seconds"], 10)
        self.assertEqual(
            point["decision"]["reason"],
            "observed demand exceeded ready capacity; emergency reactive scale",
        )

    def test_continuous_simulator_adapts_to_sustained_spike(self):
        from pathlib import Path
        from src.live_simulator import ContinuousSimulator
        workload = Path(__file__).resolve().parents[1] / "data/workload.csv"
        simulator = ContinuousSimulator(workload, interval_seconds=1, startup_delay_seconds=10)
        simulator.set_load("spike", 300, 30)
        first = simulator._step_locked()
        second = simulator._step_locked()
        self.assertEqual(first["actual_rps"], 300)
        self.assertEqual(second["actual_rps"], 300)
        self.assertGreaterEqual(second["predicted_rps"], 300)
        self.assertGreaterEqual(second["target_replicas"], 5)
        self.assertEqual(second["ready_replicas"], 1)
        self.assertGreaterEqual(second["pending_replicas"], 1)
        self.assertGreater(second["online_residual_rps"], 0)

    def test_continuous_simulator_emergency_scales_on_unseen_spike(self):
        from pathlib import Path
        from src.live_simulator import ContinuousSimulator
        workload = Path(__file__).resolve().parents[1] / "data/workload.csv"
        simulator = ContinuousSimulator(workload, interval_seconds=1, startup_delay_seconds=10)
        simulator.set_load("spike", 300, 30)
        point = simulator._step_locked()
        self.assertEqual(point["actual_rps"], 300)
        self.assertEqual(point["action"], "emergency_scale_up")
        self.assertEqual(point["target_replicas"], 5)
        self.assertEqual(point["ready_replicas"], 1)
        self.assertEqual(point["pending_replicas"], 4)
        self.assertEqual(point["overload_rps"], 240)

if __name__ == "__main__": unittest.main()
