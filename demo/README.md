# Predictive Autoscaling Demo Kit

This folder contains the material for the final demonstration video. The
application code remains in the repository root; this kit provides the
repeatable run order and ready-to-use commands.

## Demo endpoints

Open these browser tabs before recording:

```text
http://148.100.79.148:8000/
http://148.100.79.148:8000/details
```

The first page controls the live workload. The second page shows request
counts, worker P50/P95/P99/max latency, forecast demand, capacity, overload,
and Kubernetes replica state.

## Terminal layout

Use three terminals:

### Terminal 1: Kubernetes deployment

```sh
ssh linuxone
sudo microk8s kubectl get deployment ppo-workers -w
```

### Terminal 2: Worker pods

```sh
ssh linuxone
sudo microk8s kubectl get pods -l app=ppo-worker -w
```

### Terminal 3: Windows traffic and metrics

From the repository root:

```powershell
.\demo\check-live.ps1
.\demo\run-live-ramp.ps1
```

## Recording sequence

1. Show the repository and this demo guide.
2. Open `/details` and show one ready worker with zero traffic.
3. Show the LinuxONE node and `ppo-workers` at `1/1`.
4. Open `/` and set the target to `1000 RPS` with a `180 second` ramp.
5. Click **Start gradual ramp**.
6. Show `forecast_rps`, `desired_replicas`, `pending_replicas`, and
   `ready_replicas` increasing.
7. Show Kubernetes moving from `1/1` to multiple ready pods.
8. Show P95/P99 latency and capacity on `/details`.
9. Stop the workload and capture scale-down.

## Offline evidence

From the repository root:

```powershell
python -m src.workload_generator
python -m src.forecaster
python -m src.evaluate
python -m unittest discover -s tests -v
```

The generated results are written to `results/`, which is intentionally
ignored by Git. The canonical implementation is under `src/`, `tests/`,
`scripts/`, and `k8s/`.

## What to say

> This demo sends real HTTP requests from the workstation to an IBM LinuxONE
> gateway running on `s390x`. The controller uses recent demand and its
> forecast to request worker replicas before capacity is exhausted. Kubernetes
> readiness determines when capacity becomes available, so pending and ready
> replicas are shown separately. When traffic stops, the deployment scales
> back down.
