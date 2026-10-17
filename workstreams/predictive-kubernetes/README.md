# Predictive Kubernetes Workstream

This directory documents the PPO and Kubernetes delivery workstream inside the
shared repository. Its code, tests, Kubernetes manifest, dependencies, trained
model, and workload input are kept together for ownership and review. The
canonical runtime implementation remains in the repository-root `src/`,
`tests/`, `k8s/`, and `scripts/` locations.

## Project files

- `src/`: PPO agent, environment, training, evaluation, and live gateway/worker.
- `tests/test_ppo_project.py`: PPO behavior and training-input tests.
- `requirements-ppo.txt`: Python dependencies for training and the live app.
- `Dockerfile.ppo`: Python 3.12 container for gateway and worker roles.
- `k8s/ppo-autoscaler.yaml`: scoped RBAC, Deployments, and Services.
- `models/`: saved policy and seed history.
- `workload.csv`: deterministic four-scenario training and evaluation trace.

## Local setup, training, and evaluation

Use Python 3.12. In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-ppo.txt
python -m unittest tests.test_ppo_project -v
python -m src.train_ppo --steps 5120
python -m src.evaluate_ppo
```

On Linux/macOS, activate `.venv` with `source .venv/bin/activate` and use the
same Python commands. Evaluation writes interval and summary CSVs under the
local `results/` directory.

## Container smoke test

Build the image from this directory. Use either Docker or Podman:

```sh
docker build -f Dockerfile.ppo -t ppo-autoscaler:local .
```

Start a worker container and check its health and request handler:

```sh
docker run --rm -d --name ppo-worker -e APP_ROLE=worker \
  -p 8001:8001 ppo-autoscaler:local
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/ready
curl -X POST http://127.0.0.1:8001/work
docker stop ppo-worker
```

For Podman, replace `docker` with `podman`. This smoke test checks the
containerized worker; testing gateway scaling requires Kubernetes.

## Live Kubernetes demo

For Kind, create the cluster, load the image, apply the manifest, and wait for
both deployments:

```sh
kind create cluster --name ppo-autoscaling
docker build -f Dockerfile.ppo -t ppo-autoscaler:local .
kind load docker-image ppo-autoscaler:local --name ppo-autoscaling
kubectl apply -f k8s/ppo-autoscaler.yaml
kubectl rollout status deployment/ppo-workers
kubectl rollout status deployment/ppo-controller
kubectl port-forward service/ppo-controller 8080:8000
```

Open `http://127.0.0.1:8080/` for the demo dashboard. The gateway forwards
`/hit` requests to worker pods and adjusts the `ppo-workers` Deployment using
RBAC scoped to that Deployment and its scale subresource. `/health` reports
process liveness. `/ready` reports readiness; the gateway checks that the
worker Service responds before becoming ready.

Image references and cluster setup may need adjustment for another cluster.

## LinuxONE (`s390x`) validation

Use a configured multi-platform builder and IBM-provided registry and cluster
credentials. From this directory:

```sh
docker buildx build --platform linux/s390x -f Dockerfile.ppo \
  -t <registry>/<namespace>/ppo-autoscaler:<tag> --push .
docker buildx imagetools inspect <registry>/<namespace>/ppo-autoscaler:<tag>
```

Set both deployment image references in `k8s/ppo-autoscaler.yaml` to the pushed
image, then deploy on an actual `s390x` cluster and verify worker readiness,
gateway forwarding, and scale changes. A successful build on another
architecture does not validate runtime behavior on LinuxONE.

Local environments, Python cache files, and evaluation outputs are excluded by
this directory's `.gitignore`.
