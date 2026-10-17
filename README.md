# LinuxONE Predictive Autoscaling

An IBM Z Datathon project that compares delayed reactive scaling with
forecast-assisted scaling and runs a PPO controller against a Kubernetes worker
Deployment. The project is designed to run on Python 3.12 and IBM LinuxONE
`s390x`; scientific dependencies must be installed from wheels or built with a
native LinuxONE builder.

## Repository layout

```text
data/       deterministic workload traces
k8s/        Kubernetes RBAC, worker, controller, and service manifests
models/     saved PPO policy
results/    generated evaluation output (ignored by Git)
scripts/    Linux pipeline and load helpers
src/        forecasting, simulation, PPO, and FastAPI gateway
tests/      forecasting and PPO regression tests
```

There is one canonical source tree. The former teammate folders were removed
because they duplicated source, manifests, models, and tests.

## LinuxONE setup

Use a native LinuxONE `s390x` host or a configured multi-platform builder.
Avoid copying x86 virtual environments or compiled wheels from another machine.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-ppo.txt
```

Run the complete offline pipeline and tests:

```sh
bash scripts/run_pipeline.sh
python -m unittest discover -s tests -v
python -m src.train_ppo --steps 5120
python -m src.evaluate_ppo
```

The forecasting pipeline writes forecast and reactive/predictive comparison
files to `results/`. PPO evaluation writes
`ppo_scaling_intervals.csv` and `ppo_scaling_comparison.csv`.

## Kubernetes demo and live dashboard

Build the image on LinuxONE or with a LinuxONE-capable builder:

```sh
docker buildx build --platform linux/s390x \
  -f Dockerfile.ppo -t <registry>/<namespace>/predictive-autoscaler:<tag> \
  --push .
```

Set the image in both Deployments in `k8s/ppo-autoscaler.yaml`, then deploy:

```sh
kubectl apply -f k8s/ppo-autoscaler.yaml
kubectl rollout status deployment/ppo-workers
kubectl rollout status deployment/ppo-controller
kubectl port-forward service/ppo-controller 8080:8000
```

Open <http://127.0.0.1:8080/>. The integrated FastAPI dashboard shows:

- live request rate and estimated ready capacity;
- desired, ready, and pending worker replicas;
- PPO scale decisions and controller errors;
- traffic/capacity and replica-history charts;
- recent scaling events;
- saved PPO evaluation results;
- buttons for controlled request loads and bursts.

### LinuxONE server + local client experiment

For a real experiment, run the server and Kubernetes control plane on the
LinuxONE host. Run the load generator from a separate client machine so its
latency includes the network path.

On the LinuxONE host:

```sh
docker buildx build --platform linux/s390x \
  -f Dockerfile.ppo -t <registry>/<namespace>/predictive-autoscaler:<tag> \
  --push .
# Set that image in k8s/ppo-autoscaler.yaml before applying it.
kubectl apply -f k8s/ppo-autoscaler.yaml
kubectl rollout status deployment/ppo-workers --timeout=180s
kubectl rollout status deployment/ppo-controller --timeout=180s
kubectl port-forward --address 0.0.0.0 service/ppo-controller 8000:8000
```

Use a secure tunnel or private network for the port-forward; do not expose
the unauthenticated demo service directly to the public internet. Verify from
the client machine:

```powershell
Invoke-RestMethod http://<linuxone-host>:8000/health
Invoke-RestMethod http://<linuxone-host>:8000/ready
```

From this Windows workstation, run a real HTTP experiment:

```powershell
.\scripts\run_remote_experiment.ps1 `
  -BaseUrl http://<linuxone-host>:8000 `
  -Requests 10000 `
  -Concurrency 100
```

The report includes client-observed throughput, status counts, P50/P95/P99
latency, controller state, and gateway-to-worker latency from `/api/metrics`.
The client latency includes network and gateway time; gateway metrics isolate
the gateway-to-worker portion. Capture `/api/state` and `/api/history` before,
during, and after each load phase. Run a baseline first, then a sustained
load, then a step spike, and compare error rate, client P95/P99, desired versus
ready replicas, pending replicas, overload, and recovery time.

The equivalent Linux client command is:

```sh
bash scripts/run_remote_experiment.sh http://<linuxone-host>:8000 10000 100
```

For a continuous experiment that stays visible on the dashboard, use the
fourth argument as a duration in seconds. The request count is then unlimited
until the duration expires:

```powershell
.\scripts\run_remote_experiment.ps1 `
  -BaseUrl http://<linuxone-host>:8000 `
  -Concurrency 100 `
  -DurationSeconds 60
```

Equivalent Linux command:

```sh
bash scripts/run_remote_experiment.sh http://<linuxone-host>:8000 0 100 60
```

For a controlled, non-spike ramp that gradually reaches 1000 RPS, use ramp
mode. It sends requests at an increasing target rate instead of saturating the
client immediately:

```powershell
.\scripts\run_remote_experiment.ps1 `
  -BaseUrl http://<linuxone-host>:8000 `
  -Concurrency 100 `
  -RampToRps 1000 `
  -RampDurationSeconds 180
```

Equivalent Linux command:

```sh
bash scripts/run_remote_experiment.sh http://<linuxone-host>:8000 0 100 0 1000 180
```

The gateway forecasts the next demand trend and reports `forecast_rps` and
`forecast_target_replicas` separately from measured `rps_last_window`. During
the ramp, the expected sequence is `forecast_scale_up`, desired replicas
increasing, pending replicas while pods start, and then ready replicas catching
up. The LinuxONE manifest permits 20 workers (1200 RPS of configured capacity),
so the full 1000-RPS ramp can become ready without relying on emergency scaling.
Watch both `/details` and Kubernetes directly:

```sh
sudo microk8s kubectl get deployment ppo-workers -w
sudo microk8s kubectl get pods -l app=ppo-worker -w
```

The LinuxONE `/details` page also has browser controls for the demo. Use
**Start gradual ramp** to begin a browser-generated ramp, **Increase +100** or
**Decrease -100** to change the live target, **Pause/Resume** to hold traffic,
and **Stop** to end it. These controls send real `POST /hit` requests through
the LinuxONE gateway; they are not simulator-only counters. The quick-test
buttons remain available for small fixed request batches.

Use `-Concurrency 100` for a continuous spike. Use `-Concurrency 10` or `20`
for a continuous normal-rate load. Keep `/api/state` open in another terminal
while the run is active; the five-second controller window will then visibly
show actual RPS, `emergency_scale_up`, desired replicas, pending replicas, and
ready replicas. Stop the run with `Ctrl+C`; the script prints its report when
the configured duration ends.

For the real LinuxONE demo, open the LinuxONE URL, not the local simulator URL:

```text
http://<linuxone-host>:8000/
http://<linuxone-host>:8000/details
```

`http://127.0.0.1:8000/` is the local continuous simulator and does not show
the LinuxONE Kubernetes pods. The continuous runner can take longer than its
nominal duration to finish reporting if requests are still waiting for their
HTTP timeout; the controller decisions are still visible during the active
load window.

This experiment measures real HTTP behavior. It does not claim that a
Kubernetes replica is ready instantly: the Kubernetes readiness probe and
actual pod startup determine when capacity becomes available. The gateway
also applies an emergency scale-up when measured RPS exceeds ready capacity.

The gateway also exposes:

```text
GET  /health       process liveness
GET  /ready        worker-backed readiness
GET  /api/state    current controller state
GET  /api/history  recent telemetry and events
GET  /api/results  saved PPO evaluation summary
POST /hit          forward one request to a worker
```

## Continuous laptop simulator

Run the same prediction-and-scaling loop locally without Kubernetes:

```sh
APP_ROLE=simulator CONTROL_INTERVAL_SECONDS=2 \
  python -m uvicorn src.ppo_k8s_app:app --host 127.0.0.1 --port 8000
```

Or use `bash scripts/run_simulator.sh`. On Windows PowerShell, run
`.\scripts\run_simulator.ps1`.

Open <http://127.0.0.1:8000/>. The simulator continuously replays the
deterministic workload, trains a NumPy linear lag model from prior observations,
predicts the next interval, applies bounded replica changes with a configured
startup delay, and refreshes the dashboard every second. It shows actual demand,
forecast demand, capacity, ready/desired replicas, overload, actions, and scale
events. Set `CONTROL_INTERVAL_SECONDS` to a smaller value for a faster demo.
`STARTUP_DELAY_SECONDS` defaults to 10 seconds; use it to model realistic
provisioning rather than making replicas ready in one control tick.

### Manual spike controller

The simulator dashboard includes a bounded manual load controller. Use it to
inject a constant synthetic demand level without generating external traffic:

1. Start the simulator with `.\scripts\run_simulator.ps1`.
2. Open <http://127.0.0.1:8000/>.
3. Enter an RPS value from `0` to `600` and a duration from `1` to `300`
   seconds.
4. Select **Spike** or **Steady**, then click **Start load**.
5. Watch actual demand, forecast, capacity, overload, replica counts, charts,
   and scaling events update live.
6. Click **Stop load** to return to the deterministic workload trace.

The same controls are available through the simulator API:

```powershell
$body = @{ mode = "spike"; rate_rps = 300; duration_seconds = 20 } |
  ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/load -Method Post `
  -ContentType "application/json" -Body $body

Invoke-RestMethod http://127.0.0.1:8000/api/load/stop -Method Post
```

The simulator reports estimated overload as `max(actual_rps - capacity_rps, 0)`.
It does not create real user requests or measure application P95 latency. For
real HTTP latency, run a separate load generator against the gateway and
measure successful request durations independently.

The `/api/state` response exposes the latest model decision, including forecast
RPS, target replicas, previous and new desired replicas, the decision reason,
utilization, forecast error, pending replicas, and configured startup delay.
Each `/api/history` point contains the same capacity and timing fields for
post-test analysis.

The live forecaster uses two layers: the trained seasonal lag model and an
online correction layer. After each completed interval it records the forecast
error and compares the trained prediction with the most recent observed RPS.
For a sustained 300 RPS spike, the first interval may still reflect the old
forecast, but the next interval should adapt toward 300 RPS, request up to the
replica limit one step at a time, and show those replicas as pending during
startup. This is online forecast adaptation, not a claim that the NumPy model
weights were retrained during the dashboard session.

Unknown sudden spikes also use a reactive safety path. If observed demand
exceeds capacity from ready replicas, the controller immediately requests the
replicas required for that observed rate and labels the decision
`emergency_scale_up`. The replicas still remain pending until
`STARTUP_DELAY_SECONDS` elapses, so the dashboard shows the real overload and
provisioning gap instead of pretending the spike was predicted or instantly
fixed.

This mode is a local simulation: it does not change Kubernetes and it does not
claim to measure production latency or real request counters.

The dashboard is same-origin and has no Node.js, CDN, or browser build
dependency. It is therefore suitable for a LinuxONE deployment with only the
Python container.

## Experiment assumptions

The synthetic traces use four scenarios with five-second intervals. Forecasting
uses only completed observations: six short lags and a 120-interval seasonal
lag. The held-out window is evaluated chronologically with rolling one-step
forecasts. Reactive and predictive policies use the same capacity, replica
bounds, startup delay, cooldown, initial replicas, and held-out intervals.

Overload and excess-request values are estimates from sampled RPS, not measured
latency or production request counters. The experiment does not model pod
scheduling time, queueing, resource contention, or a novel unseen spike.
Successful local or `amd64` execution does not certify `s390x`; validate the
image and runtime on an actual LinuxONE environment.

## Validation

```sh
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

Generated outputs, caches, and virtual environments are excluded by
`.gitignore`.

## Submission runbook: deploy, test, and record the demo

This section is the reproducible runbook used for the single-node LinuxONE
experiment. Replace `148.100.79.148` with the address of your own host.
The LinuxONE host must be reachable by SSH as `linux1`, and the Windows
client must be able to reach TCP port `8000`.

### 1. Build the LinuxONE image

Build on LinuxONE so every Python dependency and native binary matches
`s390x`:

```sh
cd /home/linux1/shared/predictive-autoscaling-project
sudo docker build -f Dockerfile.ppo -t predictive-autoscaler:linuxone .
sudo docker inspect predictive-autoscaler:linuxone \
  --format '{{.Architecture}} {{.Id}}'
```

The architecture must be `s390x`. Do not copy a Windows or `amd64` virtual
environment into the image.

### 2. Load the image into MicroK8s

Docker and MicroK8s use separate image stores. Import the image explicitly:

```sh
sudo docker tag predictive-autoscaler:linuxone ppo-autoscaler:local
sudo docker save ppo-autoscaler:local | sudo microk8s ctr images import -
sudo microk8s ctr images ls | grep ppo-autoscaler
```

The manifest uses `imagePullPolicy: IfNotPresent`, so this local image is used
without a registry or internet image pull.

### 3. Prepare single-node MicroK8s networking

The application manifest is already configured for this LinuxONE host:

- the gateway uses host networking because this single-node MicroK8s setup
  does not route the Kubernetes Service CIDR from host-networked processes;
- the gateway uses the worker ClusterIP directly;
- the gateway Kubernetes client uses the local MicroK8s API endpoint
  `127.0.0.1:16443`;
- TCP port `8000` is allowed through the host firewall.

If MicroK8s was freshly installed and Calico is not ready, run this recovery
procedure once. It repairs only the single-node Calico state; it does not
stop the Jupyter container:

```sh
sudo mkdir -p \
  /var/snap/microk8s/current/var/lib/calico \
  /var/snap/microk8s/current/var/run/calico \
  /var/snap/microk8s/common/var/log/calico/cni
printf 'ppo-autoscaler' | sudo tee \
  /var/snap/microk8s/current/var/lib/calico/nodename >/dev/null
sudo cp /var/snap/microk8s/current/credentials/client.config \
  /var/snap/microk8s/current/args/cni-network/calico-kubeconfig
sudo chown root:microk8s \
  /var/snap/microk8s/current/args/cni-network/calico-kubeconfig
sudo chmod 640 \
  /var/snap/microk8s/current/args/cni-network/calico-kubeconfig
```

If the Calico custom resources are missing, create the initial cluster
information and IP pool:

```sh
cat <<'YAML' | sudo microk8s kubectl apply -f -
apiVersion: crd.projectcalico.org/v1
kind: ClusterInformation
metadata:
  name: default
spec:
  clusterGUID: linuxone-single-node
  clusterType: k8s,bgp
  calicoVersion: v3.29.3
  datastoreReady: true
---
apiVersion: crd.projectcalico.org/v1
kind: IPPool
metadata:
  name: default-ipv4-ippool
spec:
  cidr: 10.1.0.0/16
  blockSize: 26
  ipipMode: Never
  vxlanMode: Always
  natOutgoing: true
  disabled: false
YAML
```

For this host-networked, single-node installation, Calico and its controllers
must use the local API endpoint instead of `10.152.183.1:443`. Calico already
uses host networking in this MicroK8s installation. Apply the API endpoint
override to both Calico workloads:

```sh
sudo microk8s kubectl set env -n kube-system \
  daemonset/calico-node KUBERNETES_SERVICE_HOST=127.0.0.1 \
  KUBERNETES_SERVICE_PORT=16443
sudo microk8s kubectl set env -n kube-system \
  deployment/calico-kube-controllers KUBERNETES_SERVICE_HOST=127.0.0.1 \
  KUBERNETES_SERVICE_PORT=16443
sudo microk8s kubectl rollout status -n kube-system \
  daemonset/calico-node --timeout=180s
sudo microk8s kubectl rollout status -n kube-system \
  deployment/calico-kube-controllers --timeout=180s
```

The controller must also use host networking on this single-node host so it
can reach the local API endpoint. This is already present in
`k8s/ppo-autoscaler.yaml`. Leave CoreDNS on normal pod networking; it is not
needed by the application because the gateway uses the worker ClusterIP
explicitly. Do not put CoreDNS on host networking because it can conflict
with the host's DNS listener on port 53.

If the Calico pods were created before the environment change, delete only
those pods so their controllers recreate them:

```sh
sudo microk8s kubectl delete pod -n kube-system \
  -l k8s-app=calico-node --ignore-not-found
sudo microk8s kubectl delete pod -n kube-system \
  -l k8s-app=calico-kube-controllers --ignore-not-found
sudo microk8s kubectl delete pod -n kube-system \
  -l k8s-app=kube-dns --ignore-not-found
```

Confirm the cluster before deploying the application:

```sh
sudo microk8s kubectl get nodes
sudo microk8s kubectl get pods -A
```

The node and Calico components should be `Ready`/`1/1 Running`. If an
application pod says `cannot find a qualified ippool`, create the IP pool
above before retrying.

### 4. Deploy the real gateway and worker

```sh
cd /home/linux1/shared/predictive-autoscaling-project
sudo iptables -C INPUT -p tcp --dport 8000 -j ACCEPT 2>/dev/null || \
  sudo iptables -I INPUT 1 -p tcp --dport 8000 -j ACCEPT
sudo microk8s kubectl apply -f k8s/ppo-autoscaler.yaml
sudo microk8s kubectl rollout status deployment/ppo-workers --timeout=180s
sudo microk8s kubectl rollout status deployment/ppo-controller --timeout=180s
sudo microk8s kubectl get pods -o wide
```

Do not run the Docker simulator on port `8000` at the same time:

```sh
sudo docker stop predictive-autoscaler-sim 2>/dev/null || true
```

The Jupyter container is unrelated and should remain running.

### 5. Verify from LinuxONE and from the client workstation

On LinuxONE:

```sh
curl -i http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
curl -X POST -i http://127.0.0.1:8000/hit
curl http://127.0.0.1:8000/api/state
curl http://127.0.0.1:8000/api/metrics
```

From Windows:

```powershell
$base = "http://148.100.79.148:8000"
curl.exe "$base/health"
curl.exe "$base/ready"
curl.exe -X POST "$base/hit"
curl.exe "$base/api/state"
curl.exe "$base/api/metrics"
```

`GET /hit` is intentionally invalid and returns `405`; the endpoint is
`POST /hit`.

### 6. Run the video experiment

Record the dashboard at `http://148.100.79.148:8000/` and the detailed
decision view at `http://148.100.79.148:8000/details`. Keep a terminal with
the following commands visible for evidence:

```powershell
curl.exe "$base/api/state"
curl.exe "$base/api/metrics"
```

Use these phases in the recording:

1. **Baseline, 30 seconds:** send 1 request every few seconds and show one
   ready worker, low RPS, zero errors, and baseline P95.
2. **Sustained load:** run
   `.\scripts\run_remote_experiment.ps1 -BaseUrl $base -Requests 3000
   -Concurrency 25`. Show throughput, client P95/P99, gateway P95/P99,
   desired replicas, ready replicas, and controller actions.
3. **Sudden spike:** run the same script with `-Concurrency 100`. Explain
   that an unseen spike cannot be forecast before it happens; the
   `emergency_scale_up` safety path reacts to measured overload.
4. **Provisioning delay:** keep the dashboard visible while desired replicas
   increases. Explain that desired replicas are requests, pending replicas
   are provisioning, and only ready replicas add capacity.
5. **Recovery:** stop the load, wait for the controller interval, and capture
   the return to low RPS, stable ready replicas, and zero request errors.

For a repeatable report:

```powershell
.\scripts\run_remote_experiment.ps1 `
  -BaseUrl $base `
  -Requests 10000 `
  -Concurrency 100
```

The script measures client-observed latency. `/api/metrics` measures only
gateway-to-worker latency. Report both; they answer different questions.

### 7. Metrics to show in the submission

| Metric | Meaning |
|---|---|
| Client P50/P95/P99 | End-to-end latency from the workstation |
| Gateway P50/P95/P99 | Gateway-to-worker HTTP latency |
| Throughput | Completed requests divided by test duration |
| HTTP status counts | Successes and errors during the phase |
| Actual RPS | Requests observed by the gateway control loop |
| Forecast RPS | PPO/forecast estimate for the next interval |
| Desired replicas | Controller target |
| Ready replicas | Pods currently serving capacity |
| Pending replicas | Desired minus ready replicas |
| Capacity RPS | `ready replicas * 60` |
| Overload RPS | `max(actual RPS - capacity RPS, 0)` |
| Action | PPO action or `emergency_scale_up` |
| Recovery time | Time from load stop until overload clears |

Never describe `desired_replicas` as running capacity. Only
`ready_replicas` contributes to capacity.

### 8. Final evidence checklist

Before submitting the video or report, capture:

```sh
sudo microk8s kubectl get nodes -o wide
sudo microk8s kubectl get pods -A -o wide
sudo microk8s kubectl get deployments
sudo microk8s kubectl get svc
```

Also capture:

- the `s390x` image architecture;
- `/health` and `/ready` returning HTTP 200;
- at least one successful `POST /hit`;
- the baseline and spike P95/P99 results;
- `/api/state` showing desired, ready, pending, action, and overload;
- `/api/metrics` showing gateway P95/P99 and zero errors;
- the dashboard decision timeline during scale-up and recovery.

The honest claim is: predictive scaling handles patterns, online correction
adapts to sustained changes, and emergency scaling reacts to genuinely unseen
spikes. A sudden unseen spike is not retroactively predicted.
