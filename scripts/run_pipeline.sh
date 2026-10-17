#!/usr/bin/env bash
set -euo pipefail
python3 -m src.workload_generator
python3 -m src.forecaster
python3 -m src.evaluate
