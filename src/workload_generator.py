"""Generate deterministic synthetic workload scenarios."""
import csv
import math
import random
from pathlib import Path

INTERVAL_SECONDS = 5
DURATION_MINUTES = 40
SCENARIOS = ("steady", "spike", "ramp", "periodic")
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "workload.csv"


def generate_rate(scenario, i, rng=None):
    """Return scenario demand at interval i; randomness is locally seeded by caller."""
    rng = rng or random.Random(42)
    if scenario == "steady":
        rate = 50
    elif scenario == "spike":
        # A repeated burst every ten minutes. Held-out data includes one full burst.
        rate = 200 if 48 <= i % 120 < 72 else 50
    elif scenario == "ramp":
        total_points = DURATION_MINUTES * 60 // INTERVAL_SECONDS
        rate = 50 + 150 * min(i / (total_points - 1), 1)
    elif scenario == "periodic":
        rate = 50 + 60 * (1 + math.sin(2 * math.pi * i / 120)) / 2
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    return round(max(1, rate + rng.uniform(-3, 3)), 2)


def generate_workload(points=None):
    points = DURATION_MINUTES * 60 // INTERVAL_SECONDS if points is None else points
    if points <= 0:
        raise ValueError("points must be a positive integer")
    rows = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        rng = random.Random(42 + scenario_index)
        for i in range(points):
            rows.append({"scenario": scenario, "interval": i,
                         "elapsed_seconds": i * INTERVAL_SECONDS,
                         "request_rate_rps": generate_rate(scenario, i, rng)})
    return rows


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_workload()
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Dataset saved: {OUTPUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
