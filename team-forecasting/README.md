# Teammate 1: Forecasting and Simulation

This folder defines the forecasting/simulation workstream in the shared monorepo.
The executable source remains under `src/` so the original commands and imports
continue to work.

## Owned files

- `src/workload_generator.py`
- `src/forecaster.py`
- `src/evaluate.py`
- `src/scaling_policy.py`
- `tests/test_pipeline.py`
- `requirements.txt`
- `scripts/run_pipeline.sh`
- `data/` and `results/` generated outputs

## Responsibilities

- Workload generation and deterministic scenario definitions.
- Forecast feature construction, chronological train/test evaluation, and metrics.
- Fair reactive-versus-predictive scaling simulation.
- Leakage and fairness tests.
- Documentation of assumptions and offline results.

## Validation

From the repository root:

```powershell
python -m src.workload_generator
python -m src.forecaster
python -m src.evaluate
python -m unittest discover -s tests -v
```

Do not commit local virtual environments or generated files excluded by the
root `.gitignore`.
