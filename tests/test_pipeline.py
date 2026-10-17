import unittest

import pandas as pd

from src.evaluate import simulate_policy, summarize_scaling
from src.forecaster import build_forecast_frame, chronological_split, create_features
from src.scaling_policy import recommend_replicas, update_replicas
from src.workload_generator import generate_workload


class WorkloadTests(unittest.TestCase):
    def test_generation_is_reproducible_and_covers_scenarios(self):
        a, b = generate_workload(480), generate_workload(480)
        self.assertEqual(a, b)
        self.assertEqual({r["scenario"] for r in a}, {"steady", "spike", "ramp", "periodic"})
        spikes = [r for r in a if r["scenario"] == "spike" and r["interval"] >= 360]
        self.assertTrue(any(r["request_rate_rps"] > 150 for r in spikes))

    def test_generation_rejects_empty_trace(self):
        with self.assertRaises(ValueError):
            generate_workload(0)

    def test_generation_rejects_non_integer_point_count(self):
        for points in (True, 10.5, "10"):
            with self.subTest(points=points), self.assertRaises(ValueError):
                generate_workload(points)

class ForecastTests(unittest.TestCase):
    def test_features_predict_next_interval_without_current_or_future_values(self):
        features = create_features(list(range(300)))
        row = features[features.target_interval == 120].iloc[0]
        self.assertEqual(row.lag_1, 119)
        self.assertEqual(row.seasonal_lag_120, 0)
        self.assertEqual(row.actual_rps, 120)

    def test_train_test_split_is_chronological(self):
        features = create_features(list(range(480)))
        train, test = chronological_split(features, 360)
        self.assertLess(train.target_interval.max(), test.target_interval.min())
        self.assertEqual(test.target_interval.min(), 360)

    def test_held_out_values_never_enter_training_rows(self):
        values = list(range(480))
        changed = values.copy()
        changed[360:] = [value + 100000 for value in changed[360:]]
        train, _ = chronological_split(create_features(values), 360)
        changed_train, _ = chronological_split(create_features(changed), 360)
        pd.testing.assert_frame_equal(train.reset_index(drop=True), changed_train.reset_index(drop=True))

    def test_chronological_split_rejects_empty_side(self):
        with self.assertRaises(ValueError):
            chronological_split(create_features(list(range(240))), 300)

    def test_forecast_inputs_reject_invalid_rates_and_split_values(self):
        with self.assertRaises(ValueError):
            create_features([10.0, -1.0, 20.0])
        features = create_features(list(range(240)))
        for split_target in (True, 0, 10.5):
            with self.subTest(split_target=split_target), self.assertRaises(ValueError):
                chronological_split(features, split_target)

    def test_forecast_export_rejects_invalid_predictions(self):
        features = create_features(list(range(240)))
        test = features[features.target_interval >= 180]
        with self.assertRaises(ValueError):
            build_forecast_frame("scenario", test, [-1.0] * len(test))

    def test_future_changes_cannot_change_features_for_an_earlier_forecast(self):
        values = list(range(300))
        changed = values.copy()
        changed[200:] = [value + 10000 for value in changed[200:]]
        original_row = create_features(values).query("target_interval == 200").iloc[0]
        changed_row = create_features(changed).query("target_interval == 200").iloc[0]
        feature_columns = [c for c in original_row.index if c.startswith("lag_")]
        feature_columns += ["seasonal_lag_120"]
        self.assertEqual(original_row[feature_columns].to_dict(), changed_row[feature_columns].to_dict())

    def test_forecast_export_origin_is_previous_interval(self):
        features = create_features(list(range(240)))
        test = features[features.target_interval >= 180]
        output = build_forecast_frame("scenario", test, test.actual_rps.to_numpy())
        self.assertTrue((output.interval - output.forecast_origin_interval == 1).all())
        self.assertTrue((output.elapsed_seconds == output.interval * 5).all())


class ScalingTests(unittest.TestCase):
    def test_replica_bounds_and_step(self):
        self.assertEqual(recommend_replicas(-2), 1)
        self.assertEqual(recommend_replicas(10000), 10)
        self.assertEqual(update_replicas(2, 8, max_step=1), 3)
        with self.assertRaises(ValueError):
            recommend_replicas(10, capacity_per_replica=0)

    def test_observation_and_startup_delays(self):
        group = pd.DataFrame([
            {"scenario": "x", "interval": i + 1, "elapsed_seconds": (i + 1) * 5,
             "actual_rps": rate, "predicted_rps": 180.0}
            for i, rate in enumerate([50.0, 50.0, 50.0])
        ])
        history = {0: 50.0, 1: 50.0, 2: 50.0, 3: 50.0}
        reactive = simulate_policy(group, "reactive", observation_delay=1,
                                   startup_delay=1, demand_history=history)
        predictive = simulate_policy(group, "predictive", observation_delay=1,
                                     startup_delay=1, demand_history=history)
        self.assertEqual(reactive[1]["observation_interval"], 1)
        self.assertEqual(reactive[1]["scaling_signal_rps"], 50.0)
        self.assertEqual(predictive[0]["active_replicas"], 1)
        self.assertEqual(predictive[1]["active_replicas"], 2)

    def test_missing_forecast_does_not_fall_back_to_actual_demand(self):
        group = pd.DataFrame([{
            "scenario": "spike", "interval": 10, "elapsed_seconds": 50,
            "actual_rps": 500.0, "predicted_rps": float("nan")
        }])
        row = simulate_policy(group, "predictive")[0]
        self.assertIsNone(row["scaling_signal_rps"])
        self.assertEqual(row["scaling_action"], "hold")
        self.assertEqual(row["active_replicas"], 1)

    def test_invalid_forecasts_and_actual_rates_are_rejected(self):
        base = pd.DataFrame([{
            "scenario": "s", "interval": 10, "elapsed_seconds": 50,
            "actual_rps": 50.0, "predicted_rps": 60.0
        }])
        for bad_forecast in (-1.0, float("inf"), "not-a-rate"):
            group = base.copy()
            group["predicted_rps"] = bad_forecast
            with self.subTest(predicted_rps=bad_forecast), self.assertRaisesRegex(ValueError, "forecasts"):
                simulate_policy(group, "predictive")
        group = base.copy()
        group["actual_rps"] = float("nan")
        with self.assertRaisesRegex(ValueError, "actual request rates"):
            simulate_policy(group, "predictive")

    def test_simulation_rejects_misaligned_or_duplicate_intervals(self):
        base = pd.DataFrame([
            {"scenario": "s", "interval": 10, "elapsed_seconds": 50,
             "actual_rps": 50.0, "predicted_rps": 60.0},
            {"scenario": "s", "interval": 11, "elapsed_seconds": 55,
             "actual_rps": 50.0, "predicted_rps": 60.0},
        ])
        misaligned = base.copy()
        misaligned.loc[1, "elapsed_seconds"] = 54
        with self.assertRaisesRegex(ValueError, "aligned"):
            simulate_policy(misaligned, "predictive")
        duplicate = pd.concat([base, base.iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "unique"):
            simulate_policy(duplicate, "predictive")

    def test_predictive_actions_do_not_depend_on_target_actuals(self):
        base = pd.DataFrame([
            {"scenario": "s", "interval": i + 1, "elapsed_seconds": (i + 1) * 5,
             "actual_rps": actual, "predicted_rps": forecast}
            for i, (actual, forecast) in enumerate(zip([40.0, 250.0, 80.0], [50.0, 200.0, 80.0]))
        ])
        changed = base.copy()
        changed["actual_rps"] = [900.0, 2.0, 700.0]
        original = simulate_policy(base, "predictive")
        altered = simulate_policy(changed, "predictive")
        self.assertEqual([r["scaling_action"] for r in original],
                         [r["scaling_action"] for r in altered])
        self.assertEqual([r["active_replicas"] for r in original],
                         [r["active_replicas"] for r in altered])

    def test_reactive_signal_uses_only_the_previous_interval(self):
        group = pd.DataFrame([
            {"scenario": "s", "interval": i, "elapsed_seconds": i * 5,
            "actual_rps": actual, "predicted_rps": 50.0}
            for i, actual in enumerate([40.0, 250.0, 80.0], start=1)
        ])
        history = {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0}
        rows = simulate_policy(group, "reactive", demand_history=history)
        self.assertEqual([r["observation_interval"] for r in rows], [0, 1, 2])
        self.assertEqual([r["scaling_signal_rps"] for r in rows], [10.0, 20.0, 30.0])

    def test_reactive_requires_exact_delayed_history(self):
        group = pd.DataFrame([{
            "scenario": "s", "interval": 10, "elapsed_seconds": 50,
            "actual_rps": 500.0, "predicted_rps": 10.0
        }])
        with self.assertRaisesRegex(ValueError, "missing interval 9"):
            simulate_policy(group, "reactive", observation_delay=1)

    def test_cooldown_blocks_intervening_actions(self):
        group = pd.DataFrame([
            {"scenario": "s", "interval": i, "elapsed_seconds": i * 5,
             "actual_rps": 50.0, "predicted_rps": forecast}
            for i, forecast in enumerate([180.0, 50.0, 50.0, 50.0])
        ])
        rows = simulate_policy(group, "predictive", startup_delay=0, cooldown=2)
        self.assertEqual([r["scaling_action"] for r in rows],
                         ["scale_up", "hold", "scale_down", "hold"])

    def test_scale_down_cancels_replicas_that_have_not_started(self):
        group = pd.DataFrame([
            {"scenario": "x", "interval": i, "elapsed_seconds": i * 5,
             "actual_rps": 50.0, "predicted_rps": forecast}
            for i, forecast in enumerate([180.0, 50.0, 50.0])
        ])
        rows = simulate_policy(group, "predictive", startup_delay=2)
        self.assertEqual(rows[0]["pending_replicas"], 1)
        self.assertEqual(rows[1]["scaling_action"], "scale_down")
        self.assertEqual(rows[2]["active_replicas"], 1)
        self.assertEqual(rows[2]["pending_replicas"], 0)

    def test_summary_overload_metrics(self):
        group = pd.DataFrame([
            {"scenario": "s", "policy": "reactive", "interval": i,
             "active_replicas": 1, "scaling_action": "hold",
             "estimated_overload_rps": val, "estimated_utilization": val / 60}
            for i, val in enumerate([0, 20, 10, 0, 30])
        ])
        summary = summarize_scaling(group).iloc[0]
        self.assertEqual(summary.overload_intervals, 3)
        self.assertEqual(summary.overload_duration_seconds, 15)
        self.assertEqual(summary.sum_overload_rps_intervals, 60)
        self.assertEqual(summary.estimated_excess_requests, 300)
        self.assertEqual(summary.overload_episodes, 2)


if __name__ == "__main__":
    unittest.main()
