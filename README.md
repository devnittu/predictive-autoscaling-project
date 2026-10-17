# Predictive Autoscaling Platform

An end-to-end predictive autoscaling project for the IBM Z Datathon. The
repository combines a deterministic forecasting experiment, a PPO policy, and
a real Kubernetes gateway/worker demo validated on IBM LinuxONE (`s390x`).

The project has two complementary outcomes:

1. **Offline evidence:** reproducible workload generation, chronological
   forecasting, and a fair predictive-versus-reactive comparison.
2. **Live evidence:** real HTTP traffic, gateway latency percentiles, forecast
   telemetry, and genuine Kubernetes replica provisioning and scale-down.

## Repository layout

```text
.
├── src/                         # Canonical application and experiment code
├── tests/                       # End-to-end regression and fairness tests
├── scripts/                    # Reproducible local and remote runners
├── k8s/                        # Kubernetes manifests for the live demo
├── models/                     # Trained PPO policy and seed metadata
├── data/                       # Generated workload input
├── workstreams/
│   ├── forecasting-simulation/ # Forecasting and offline evaluation workspace
│   └── predictive-kubernetes/  # PPO and Kubernetes delivery workspace
├── Dockerfile.ppo              # Gateway/worker image
└── requirements*.txt           # Runtime and experiment dependencies
```

The root `src/`, `tests/`, `k8s/`, and `scripts/` directories are canonical.
The two `workstreams/` directories are clearly named ownership workspaces for
parallel development and documentation; they do not introduce alternate
runtime packages.

## Problem and motivation

Reactive autoscalers act after demand appears in sampled metrics, and new replicas take time to become ready. Short, repeating workload changes may therefore create temporary overload before capacity catches up. This project asks a narrow, testable question: when a one-step workload forecast is available using prior observations, can an otherwise identical replica rule reduce estimated overload under the same startup delay and replica budget?

## Architecture

```text
workload_generator -> workload.csv -> forecaster -> forecast CSV + accuracy plot
                                             |
                                             v
                           evaluator -> per-interval simulation + scaling summary/plot
```

The original `src.app` FastAPI service remains a health-check example. The separate PPO demo below runs a gateway and worker Deployment; the gateway forwards real HTTP requests and patches the worker Deployment scale through Kubernetes RBAC.

## Workstream ownership

Two teammates can work in parallel in the same repository:

- [workstreams/forecasting-simulation/](./workstreams/forecasting-simulation/)
  owns workload generation, forecasting, offline evaluation, and
  reactive-versus-predictive fairness tests.
- [workstreams/predictive-kubernetes/](./workstreams/predictive-kubernetes/)
  owns PPO training/evaluation, the live Kubernetes path, RBAC/manifests, and
  LinuxONE validation.

Coordinate changes to shared root documentation, generated outputs, and
dependency files before merging.

## Setup and run

Use Python 3.11 or newer. In Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.workload_generator
python -m src.forecaster
python -m src.evaluate
python -m unittest discover -s tests -v
```

On Linux, run the pipeline with `bash scripts/run_pipeline.sh`, then run tests with `python3 -m unittest discover -s tests -v`.

To reproduce from clean generated outputs, remove only the generated files and rerun the pipeline and tests. In Windows PowerShell:

```powershell
$generated = @('data/workload.csv','results/forecast_metrics.csv','results/forecast_plot.png','results/forecast_results.csv','results/scaling_comparison.csv','results/scaling_comparison.png','results/scaling_summary.csv','results/summary.txt')
Remove-Item -LiteralPath $generated -ErrorAction SilentlyContinue
python -m src.workload_generator
python -m src.forecaster
python -m src.evaluate
python -m unittest discover -s tests -v
```

On Linux:

```sh
rm -f data/workload.csv results/forecast_metrics.csv results/forecast_plot.png results/forecast_results.csv results/scaling_comparison.csv results/scaling_comparison.png results/scaling_summary.csv results/summary.txt
bash scripts/run_pipeline.sh
python3 -m unittest discover -s tests -v
```

## Forecast method and experiment

The generator creates four 40-minute scenarios at 5-second intervals: steady demand, repeated 2-minute bursts every 10 minutes, a linear ramp, and a periodic trace. A local fixed seed per scenario makes generated rows deterministic. Each scenario is split chronologically: intervals 0–359 are the pre-test period (the model fit uses target intervals 120–359 because the seasonal lag needs 120 points), and intervals 360–479 are held out.

For target interval *t*, the feature vector is `[d(t-1), ..., d(t-6), d(t-120)]`. A scikit-learn ordinary least squares regression predicts `d(t)`. The seasonal lag provides a simple periodic baseline for the 10-minute periodic and burst patterns. It never includes `d(t)` or a later value. Test forecasts use a rolling one-step setup: once interval *t* has passed, its observed demand can be used to forecast *t+1*. This is not a 120-step forecast made at the train/test boundary. Consequently, the known repeated spike timing is represented in the held-out history; this experiment does not claim to predict a novel, unannounced spike.

Forecast quality is measured against held-out actuals using:

- `MAE = mean(abs(actual - forecast))`
- `RMSE = sqrt(mean((actual - forecast)^2))`

Both are in requests per second (RPS), and are reported per scenario in `forecast_metrics.csv`.

## Scaling simulation

The evaluator joins forecasts to the workload by `(scenario, interval)` and simulates the same 120 held-out target intervals for each policy. It verifies that each forecast's origin is `target interval - 1` and that the saved actual value matches the workload.

At the start of interval *t*:

1. Any scale-up replicas whose startup delay has elapsed become active.
2. Reactive observes demand from *t - 1* (one 5-second interval of observation delay). Predictive uses the forecast for *t*, issued from data through *t - 1*.
3. The policy recommends `clip(ceil(signal_rps / 60), 1, 10)` replicas. Each decision can change the provisioned target by at most one replica. The default cooldown is zero intervals.
4. Scale-ups become active at the start of *t + 1* (5-second startup delay). Scale-downs are immediate; pending starts are cancelled before active replicas are retired.
5. Actual demand for *t* is served by active replicas. Both policies start each scenario with one active replica and use 60 RPS per replica, minimum 1, maximum 10, identical scaling step, observation timing, startup timing, and cooldown settings.

For actual rate `d_t` and active replicas `r_t`, estimated overload rate is `max(0, d_t - 60*r_t)` RPS. The summary separates:

- `overload_intervals`: number of sampled 5-second intervals with positive overload.
- `overload_duration_seconds`: overload interval count times 5 seconds.
- `sum_overload_rps_intervals`: the sum of sampled excess rates, in RPS-intervals.
- `estimated_excess_requests`: the RPS-interval sum times 5 seconds, assuming the sampled rate remains constant for that interval. This is an estimate, not a request counter.
- `mean_overload_recovery_seconds`: mean onset-to-first-non-overload time for episodes that recover within the held-out window. Episodes still active at the window end are censored and excluded from this mean.
- Replica intervals, mean replicas, scaling action counts, utilization, peak overload, and episode count are additional simulation metrics.

Missing forecasts hold the predictive target unchanged; actual demand is never used as a forecast fallback. The simulation does not model queueing, request latency, pod scheduling, resource contention, changing per-replica capacity, or measured Kubernetes control-plane timing. Results are estimates, not Prometheus or production measurements. Predictive scaling is not expected to win on every workload.

## Latest local results

This deterministic run used seed 42 per scenario, 120 held-out intervals, and the assumptions above. Forecast metrics and scaling metrics are separate:

| Scenario | Forecast MAE (RPS) | Forecast RMSE (RPS) |
|---|---:|---:|
| Steady | 1.386 | 1.634 |
| Periodic | 1.775 | 2.174 |
| Ramp | 1.614 | 1.884 |
| Spike | 1.977 | 2.459 |

| Scenario | Policy | Overload intervals / duration | Estimated excess requests | Mean active replicas |
|---|---|---:|---:|---:|
| Periodic | Predictive | 4 / 20 s | 113.45 | 1.717 |
|  | Reactive | 5 / 25 s | 168.85 | 1.717 |
| Ramp | Predictive | 5 / 25 s | 760.95 | 3.475 |
|  | Reactive | 8 / 40 s | 792.50 | 3.442 |
| Spike | Predictive | 3 / 15 s | 1,187.80 | 1.575 |
|  | Reactive | 4 / 20 s | 1,877.95 | 1.575 |
| Steady | Predictive | 0 / 0 s | 0 | 1.000 |
|  | Reactive | 0 / 0 s | 0 | 1.000 |

The result favors predictive scaling on overload in these generated traces, with similar replica use. It is a single synthetic experiment and does not establish production performance.

## Outputs

- `data/workload.csv`: generated scenarios with `scenario`, `interval`, `elapsed_seconds`, `request_rate_rps`.
- `results/forecast_results.csv`: held-out actual/forecast rates, target interval, and forecast origin.
- `results/forecast_metrics.csv`: per-scenario MAE, RMSE, test size, and chronological split boundaries.
- `results/scaling_comparison.csv`: policy decisions and estimated state for every held-out interval.
- `results/scaling_summary.csv`: machine-readable aggregate outcomes with units in metric names.
- `results/summary.txt`: assumptions and a readable summary.
- `results/forecast_plot.png`: forecast traces; `results/scaling_comparison.png`: actual workload against each policy's assumed capacity, alongside active replica counts.

The documented pipeline regenerates each output. The workload generator overwrites only its workload CSV; subsequent stages overwrite their corresponding result files.

## 4-minute Datathon demo

**0:00–0:40 — Problem and scope.** “Reactive scaling waits for demand metrics and replica startup. This prototype tests whether a short workload forecast can make the same simple scaling rule act earlier. It is an offline simulation; it does not control a cluster.”

**0:40–1:25 — Workloads and forecast.** Show the four scenarios in `forecast_plot.png`. Explain the chronological boundary at interval 360, the six recent lags plus the 120-interval seasonal lag, and one-step rolling evaluation. Point to the spike burst: it repeats every 120 intervals and the prior burst is in the training history, so the model learns repetition rather than predicting a novel spike.

**1:25–2:35 — Fair controller comparison.** Open `scaling_comparison.png` and `scaling_comparison.csv`. Compare actual workload with the capacity line for each policy, then pick the same scenario and interval. Show that reactive observes interval *t−1*, predictive uses the forecast for *t* issued from data through *t−1*, and scale-ups start one interval later for both. The policies use the same held-out rows, initial replica count, capacity, bounds, scale step and cooldown.

**2:35–3:25 — Results and limitations.** Show `forecast_metrics.csv` separately from `scaling_summary.csv`. State that estimated overload was lower for the forecast policy on these periodic, ramp and repeated-spike traces, while replica usage was similar and the comparison is limited to these synthetic traces. Explain that excess requests assume sampled overload remains constant for five seconds. These are estimates, not observed service metrics.

**3:25–4:00 — Demo boundary.** Optionally show `/health` in the API container. Explain that this service and the optional CPU HPA are separate from the forecast simulation. For an IBM LinuxONE follow-up, build and run on `s390x`, verify dependencies, and connect only after cluster credentials and Metrics Server are provided.

### Likely judge questions

- **Did the forecast see the test target?** No. Features for target *t* end at *t−1*, and the saved origin is checked against that interval. The rolling test uses observed earlier test points to make later one-step forecasts, as a live system could.
- **Why does the forecast follow the spike edge so closely?** The edge is repeated every 120 intervals. The 120-interval lag points to the previous cycle's matching position. A new, unannounced spike is not predicted by this result.
- **Does predictive scaling always win?** No. It reduces estimated overload in these traces, with very similar replica usage; the ramp case uses slightly more replica intervals. Different noise, capacity, startup time or novel demand can change the outcome.
- **Are excess requests measured?** No. They are estimated as sampled overload RPS times five seconds per interval. No request counter, latency, pod scheduling time or Prometheus data is collected.
- **Is this running on Kubernetes or LinuxONE?** Not in this experiment. The API image has a published `s390x` base and its dependencies resolve for that architecture, but the complete scientific pipeline needs compatible wheels or native builds. A cluster and IBM credentials are required for deployment.

## Docker and Kubernetes example

Build and serve the separate health-check API locally:

```sh
docker build -t autoscaling-demo:latest .
docker run --rm -p 8000:8000 autoscaling-demo:latest
```

The API container installs only `requirements-api.txt`; the forecasting dependencies in `requirements.txt` are not needed for the API. The API responds at `/` and `/health`. The Kubernetes manifests deploy that API and a reactive CPU HPA. They require a cluster with Metrics Server, permission to create Deployment/Service/HPA resources, and an image available to cluster nodes:

```sh
kubectl apply -f k8s/demo-app.yaml
kubectl apply -f k8s/hpa.yaml
kubectl get deployment,service,hpa
```

Before deployment, set `image` in `k8s/demo-app.yaml` to the pushed image reference. The [Docker Official Images Python catalog](https://github.com/docker-library/official-images/blob/master/library/python) currently lists `s390x` for the Python 3.11 slim tag family; still verify the exact pulled tag and resulting application image manifest. With a configured multi-platform builder and registry, the Linux build/push command is `docker buildx build --platform linux/s390x -t <registry>/<namespace>/autoscaling-demo:<tag> --push .`; verify it with `docker buildx imagetools inspect <registry>/<namespace>/autoscaling-demo:<tag>`. Then configure the IBM cluster context (`kubectl config current-context`), update the manifest image, and run the apply/get commands above. This requires IBM-provided registry credentials, cluster credentials and RBAC, and a cluster with Metrics Server. Run the image on an `s390x` host to verify runtime behavior. A successful local x86 build does not prove `s390x` compatibility. The manifests do not deploy the forecasting policy, and this project has not claimed a live cluster deployment.

### Running the forecasting pipeline on LinuxONE

The API container and the scientific Python pipeline have different dependency needs. The Python 3.11 slim manifest currently publishes `linux/s390x`, and all API dependency wheels resolved for CPython 3.11/s390x in a binary-only pip check. The full pipeline's binary-only check stopped at NumPy: expect to build scientific packages from source unless the IBM environment provides compatible wheels or packages. NumPy source builds need C/C++ compilers, Python headers, BLAS/LAPACK and `pkg-config`; SciPy source builds also need a Fortran compiler. Install equivalent packages from the LinuxONE distribution repositories. On Debian-based LinuxONE, the starting package set is:

```sh
sudo apt-get update
sudo apt-get install -y build-essential gfortran pkg-config python3-dev libopenblas-dev liblapack-dev cmake ninja-build
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.workload_generator
python -m src.forecaster
python -m src.evaluate
python -m unittest discover -s tests -v
```

The exact system package names and availability depend on the IBM-provided distribution. If wheels are required, prepare a LinuxONE-compatible wheelhouse rather than downloading x86 wheels. NumPy's [source-build guide](https://numpy.org/doc/stable/building/index.html) documents its native build prerequisites. This environment did not include an IBM LinuxONE host, so the full scientific stack was not compiled or run on `s390x` here.

## Live PPO and Kubernetes replica demo

This is a second, live path alongside the linear-forecast simulator above. A NumPy PPO actor observes six completed request-rate windows plus desired and pending replicas, then chooses scale down, hold, or scale up. PPO is initialized by behavior cloning an observation-only capacity target rule and fine-tuned with clipped policy updates. The training split ends at interval 359; evaluation uses 360–479. The policy never receives the target interval's request rate. A 5-second control window and 5-second replica startup are used in simulation; Kubernetes startup time is measured through ready-replica status and may differ in practice.

The real request path is `load generator → gateway Service → gateway pod → worker Service → worker pod`. Every accepted `/hit` request is forwarded to an actual worker pod. The controller samples gateway hits over five seconds, runs the saved PPO policy, and patches the `ppo-workers` Deployment scale (one replica per action, between 1 and 10). RBAC is scoped to that Deployment and its scale subresource. The policy assumes the same nominal 60 RPS per replica used in training; that is not a measured capacity for this lightweight worker. This is a local demonstration controller, not a production autoscaler: it uses request arrival counts rather than Prometheus metrics, processes one control loop, and does not include durable telemetry, authentication, or disruption budgets.

### Train and evaluate the PPO model

From the repository root, in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ppo.txt
.\.venv\Scripts\python.exe -m src.workload_generator
.\.venv\Scripts\python.exe -m src.train_ppo --steps 5120
.\.venv\Scripts\python.exe -m src.evaluate_ppo
```

On Linux/macOS, use `python3 -m ...` instead. The saved model is `models/ppo_autoscaler.npz`; the generated held-out traces are `results/ppo_scaling_intervals.csv` and `results/ppo_scaling_comparison.csv`. The simulator uses identical held-out intervals, 60 RPS per active replica, one initial replica, one-interval startup, and 1–10 replica bounds. Estimated excess requests equal overload RPS times five seconds; these are simulated estimates, not counts from a service.

The current seeded 5,120-step PPO run produced:

| Held-out scenario | Overload intervals / seconds | Excess-rate sum (RPS-intervals) | Estimated excess requests | Mean active replicas | Peak replicas |
| --- | ---: | ---: | ---: | ---: | ---: |
| Steady | 0 / 0 | 0 | 0 | 1.00 | 1 |
| Spike | 4 / 20 | 375.59 | 1,877.95 | 1.58 | 4 |
| Ramp | 17 / 85 | 206.51 | 1,032.55 | 3.36 | 4 |
| Periodic | 4 / 20 | 25.08 | 125.40 | 2.03 | 3 |

These are a single deterministic synthetic split. PPO performs better than the one-replica policy on simulated overload, at higher average replica use in non-steady cases. The results are not production evidence. In the separate linear-forecast versus reactive experiment, the metrics in `results/scaling_summary.csv` remain the relevant comparison; do not mix them with PPO results.

### Run the real Kubernetes demo on Docker Desktop with Kind

Prerequisites: Docker Desktop with its Linux container engine running, `kubectl`, and a Windows x64 host for the commands below. Kind creates a local Kubernetes cluster inside Docker; this demonstrates Kubernetes scheduling and replica changes but is not IBM LinuxONE.

1. Train the model using the commands above. Install Kind if needed (official release binary, Windows x64):

```powershell
$kindPath = Join-Path $env:TEMP 'kind.exe'
Invoke-WebRequest https://kind.sigs.k8s.io/dl/v0.33.0/kind-windows-amd64 -OutFile $kindPath
```

2. Create cluster, build and load the app image, then deploy the controller and worker:

```powershell
& $kindPath create cluster --name ppo-autoscaling
docker build -f Dockerfile.ppo -t ppo-autoscaler:local .
& $kindPath load docker-image ppo-autoscaler:local --name ppo-autoscaling
kubectl apply -f k8s/ppo-autoscaler.yaml
kubectl rollout status deployment/ppo-controller --timeout=180s
kubectl rollout status deployment/ppo-workers --timeout=180s
kubectl get deployments,pods,services
```

3. In one PowerShell terminal, expose the gateway; in another, open the dashboard and send a 30-second burst. Watch the worker replica count in a third terminal:

```powershell
kubectl port-forward service/ppo-controller 8080:8000
```

```powershell
Start-Process 'http://localhost:8080/'
.\.venv\Scripts\python.exe scripts/send_load.py --rps 120 --duration 30
```

The dashboard buttons send 400 or 700 requests through the gateway; the CLI load sender is paced and reports achieved throughput and errors.

```powershell
kubectl get deployment ppo-workers --watch
kubectl get pods -l app=ppo-worker --watch
```

The browser dashboard shows the gateway's measured requests per second, the controller's last action, and desired/ready worker replicas. Confirm the scale change with `kubectl get deployment ppo-workers -o=jsonpath='{.spec.replicas} {.status.readyReplicas}'`. To remove the local demonstration cluster, run `& $kindPath delete cluster --name ppo-autoscaling`.

For Linux/macOS, install the matching Kind binary from the [official Kind releases](https://kind.sigs.k8s.io/docs/user/quick-start/), then run `kind create cluster --name ppo-autoscaling`, `docker build -f Dockerfile.ppo -t ppo-autoscaler:local .`, `kind load docker-image ppo-autoscaler:local --name ppo-autoscaling`, and the same `kubectl apply` / rollout commands.

### IBM LinuxONE / external Kubernetes requirements

No IBM LinuxONE cluster deployment is claimed. To deploy there, obtain an IBM-provided Kubernetes/LinuxONE cluster context and namespace, confirm `kubectl config current-context`, obtain permission to create Deployments, Services, ServiceAccounts, Roles, and RoleBindings, and have an image registry reachable by the cluster. Build the controller image for the cluster's node architecture, push it, change both image fields in `k8s/ppo-autoscaler.yaml` to that registry reference, then apply the manifest and check both rollouts. Validate that Python, NumPy, FastAPI, httpx, and the Kubernetes client install and run on the target `s390x` image. Finally send test requests and verify `kubectl get deployments,pods` changes ready replicas. The local x86 Docker/Kind run does not verify architecture compatibility, credentials, scheduling, or network access on LinuxONE.

### 4-minute live demonstration

1. **Problem and design (45 sec):** show dashboard and explain that the controller uses completed 5-second request windows and a PPO actor.
2. **Generate load (45 sec):** run the paced load command and show requests being routed to worker pod responses.
3. **Show Kubernetes scaling (90 sec):** keep `kubectl get deployment ppo-workers --watch` visible; point to policy actions, desired replicas, and readiness catching up after startup.
4. **Explain limits and results (60 sec):** show `results/ppo_scaling_comparison.csv`, separate simulated RPS assumptions from actual HTTP requests, and state the local Kind run is not a LinuxONE deployment.

Likely questions: **Is PPO seeing future traffic?** No; each state includes only completed windows. **Are replicas real?** In the local Kind demo, yes: the gateway patches a Kubernetes Deployment and `kubectl` shows the resulting pods. **Is this production ready?** No; one controller, synthetic training traces, coarse request counting, and no production telemetry/security/availability controls. **Was IBM LinuxONE tested?** No; access to the target cluster and `s390x` runtime remains required.
