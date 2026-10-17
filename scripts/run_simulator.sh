#!/usr/bin/env bash
set -euo pipefail

export APP_ROLE="${APP_ROLE:-simulator}"
export CONTROL_INTERVAL_SECONDS="${CONTROL_INTERVAL_SECONDS:-2}"
exec python -m uvicorn src.ppo_k8s_app:app --host 127.0.0.1 --port 8000
