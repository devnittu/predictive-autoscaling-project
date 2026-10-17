param(
    [Parameter(Mandatory=$true)][string]$BaseUrl,
    [int]$Requests = 1000,
    [int]$Concurrency = 50,
    [int]$PauseMilliseconds = 0,
    [int]$DurationSeconds = 0,
    [int]$RampToRps = 0,
    [int]$RampDurationSeconds = 0
)

$env:BASE_URL = $BaseUrl.TrimEnd("/")
$env:REQUESTS = $Requests
$env:CONCURRENCY = $Concurrency
$env:PAUSE_MS = $PauseMilliseconds
$env:DURATION_SECONDS = $DurationSeconds
$env:RAMP_TO_RPS = $RampToRps
$env:RAMP_DURATION_SECONDS = $RampDurationSeconds

@'
import concurrent.futures
import json
import os
import statistics
import time
import urllib.request

base = os.environ["BASE_URL"]
total = int(os.environ["REQUESTS"])
concurrency = int(os.environ["CONCURRENCY"])
pause = int(os.environ["PAUSE_MS"]) / 1000
duration = int(os.environ["DURATION_SECONDS"])
ramp_to = int(os.environ["RAMP_TO_RPS"])
ramp_duration = int(os.environ["RAMP_DURATION_SECONDS"])
if ramp_to > 0 and ramp_duration <= 0:
    raise ValueError("RampDurationSeconds must be positive when RampToRps is set")

def hit():
    started = time.perf_counter()
    try:
        request = urllib.request.Request(base + "/hit", method="POST")
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
            return (time.perf_counter() - started) * 1000, response.status
    except Exception as exc:
        return (time.perf_counter() - started) * 1000, str(exc)

def run_worker(deadline):
    worker_results = []
    while time.perf_counter() < deadline:
        worker_results.append(hit())
        if pause:
            time.sleep(pause)
    return worker_results

def run_ramp():
    results = []
    ramp_start = time.perf_counter()
    deadline = ramp_start + ramp_duration
    pending = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ramp_pool:
        while time.perf_counter() < deadline:
            elapsed = time.perf_counter() - ramp_start
            target_rps = max(1.0, ramp_to * min(1.0, elapsed / ramp_duration))
            interval = 1.0 / target_rps
            pending.append(ramp_pool.submit(hit))
            if len(pending) >= concurrency:
                pending.pop(0).result()
            time.sleep(interval)
            while pending and pending[0].done():
                results.append(pending.pop(0).result())
        results.extend(future.result() for future in pending)
    return results

started = time.perf_counter()
if ramp_to > 0:
    results = run_ramp()
    total = len(results)
else:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
      if duration > 0:
        deadline = started + duration
        worker_results = pool.map(run_worker, [deadline] * concurrency)
        results = [item for worker in worker_results for item in worker]
        total = len(results)
      else:
        results = []
        for _ in range(total):
            results.append(pool.submit(hit))
            if pause:
                time.sleep(pause)
        results = [future.result() for future in results]
elapsed = time.perf_counter() - started
latencies = sorted(value for value, status in results if isinstance(status, int))
statuses = {}
for _, status in results:
    statuses[str(status)] = statuses.get(str(status), 0) + 1
def pct(p):
    if not latencies:
        return None
    return round(latencies[min(len(latencies) - 1, int(p * len(latencies)) - 1)], 3)
report = {
    "base_url": base,
    "requests": len(results),
    "concurrency": concurrency,
    "duration_seconds": duration,
    "ramp_to_rps": ramp_to,
    "ramp_duration_seconds": ramp_duration,
    "elapsed_seconds": round(elapsed, 3),
    "throughput_rps": round(len(results) / elapsed, 3) if elapsed else None,
    "status_counts": statuses,
    "successful_requests": len(latencies),
    "client_latency_ms": {
        "p50": pct(.50), "p95": pct(.95), "p99": pct(.99),
        "max": round(max(latencies), 3) if latencies else None,
        "mean": round(statistics.fmean(latencies), 3) if latencies else None,
    },
}
print(json.dumps(report, indent=2))
try:
    with urllib.request.urlopen(base + "/api/state", timeout=5) as response:
        print(json.dumps({"controller_state": json.load(response)}, indent=2))
    with urllib.request.urlopen(base + "/api/metrics", timeout=5) as response:
        print(json.dumps({"gateway_metrics": json.load(response)}, indent=2))
except Exception as exc:
    print(json.dumps({"telemetry_error": str(exc)}))
'@ | & .\.venv\Scripts\python.exe -
