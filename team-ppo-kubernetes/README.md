# Teammate 2: PPO and Kubernetes

This folder defines the live PPO/Kubernetes workstream in the shared monorepo.
The executable source remains under `src/` so the original commands and imports
continue to work.

## Owned files

- `src/ppo_agent.py`
- `src/ppo_env.py`
- `src/train_ppo.py`
- `src/evaluate_ppo.py`
- `src/ppo_k8s_app.py`
- `tests/test_ppo_project.py`
- `requirements-ppo.txt`
- `Dockerfile.ppo`
- `k8s/ppo-autoscaler.yaml`
- `models/ppo_autoscaler.npz`
- `models/seed_history.json`

## Responsibilities

- PPO agent, environment, training, and evaluation.
- Live gateway/worker behavior and Kubernetes scaling integration.
- RBAC, Deployment, Service, image, and LinuxONE `s390x` validation.
- Python 3.12 and architecture-specific dependency checks.
- Runtime health, readiness, and deployment documentation.

The PPO container uses Python 3.12 (`python:3.12-slim`). Install
`requirements-ppo.txt` with Python 3.12 when training or evaluating locally so
the local runtime matches the live image. The forecasting workstream keeps its
own Python and dependency requirements.

The live app exposes `/health` for process liveness and `/ready` for Kubernetes
readiness. The gateway is ready only while its worker Service responds to a
health request; worker pods report ready when their app is running.

## Validation

From the repository root with Python 3.12 active:

```powershell
python -m unittest tests.test_ppo_project -v
python -m src.evaluate_ppo
```

For LinuxONE validation, build and run the image for `linux/s390x` on a
configured multi-platform builder, then verify the Kubernetes deployment on an
actual `s390x` cluster.

Do not commit local virtual environments or generated files excluded by the
root `.gitignore`.
