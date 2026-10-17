#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:?Usage: $0 BASE_URL [REQUESTS] [CONCURRENCY] [DURATION_SECONDS]}"
REQUESTS="${2:-1000}"
CONCURRENCY="${3:-50}"
DURATION_SECONDS="${4:-0}"
RAMP_TO_RPS="${5:-0}"
RAMP_DURATION_SECONDS="${6:-0}"

BASE_URL="${BASE_URL%/}" REQUESTS="$REQUESTS" CONCURRENCY="$CONCURRENCY" DURATION_SECONDS="$DURATION_SECONDS" RAMP_TO_RPS="$RAMP_TO_RPS" RAMP_DURATION_SECONDS="$RAMP_DURATION_SECONDS" \
python3 - <<'PY'
import concurrent.futures, json, os, statistics, time, urllib.request
base=os.environ["BASE_URL"]; total=int(os.environ["REQUESTS"]); concurrency=int(os.environ["CONCURRENCY"])
duration=int(os.environ["DURATION_SECONDS"])
ramp_to=int(os.environ["RAMP_TO_RPS"]); ramp_duration=int(os.environ["RAMP_DURATION_SECONDS"])
if ramp_to > 0 and ramp_duration <= 0: raise ValueError("ramp duration must be positive when ramp target is set")
def hit(_):
    start=time.perf_counter()
    try:
        request=urllib.request.Request(base+"/hit", method="POST")
        with urllib.request.urlopen(request, timeout=15) as r:
            r.read(); return (time.perf_counter()-start)*1000, r.status
    except Exception as exc:
        return (time.perf_counter()-start)*1000, str(exc)
def run_worker(deadline):
    results=[]
    while time.perf_counter() < deadline: results.append(hit(None))
    return results
def run_ramp():
    results=[]; ramp_start=time.perf_counter(); deadline=ramp_start+ramp_duration; pending=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        while time.perf_counter() < deadline:
            elapsed=time.perf_counter()-ramp_start
            target=max(1.0, ramp_to*min(1.0, elapsed/ramp_duration))
            pending.append(pool.submit(hit, None))
            if len(pending) >= concurrency: pending.pop(0).result()
            time.sleep(1.0/target)
            while pending and pending[0].done(): results.append(pending.pop(0).result())
        results.extend(f.result() for f in pending)
    return results
start=time.perf_counter()
if ramp_to > 0:
    results=run_ramp()
else:
  with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
    if duration > 0:
        deadline=start+duration
        workers=pool.map(run_worker, [deadline]*concurrency)
        results=[item for worker in workers for item in worker]
    else:
        results=list(pool.map(hit, range(total)))
elapsed=time.perf_counter()-start
lat=sorted(x for x,s in results if isinstance(s,int)); statuses={}
for _,s in results: statuses[str(s)]=statuses.get(str(s),0)+1
def pct(p): return round(lat[min(len(lat)-1,int(p*len(lat))-1)],3) if lat else None
total=len(results)
print(json.dumps({"base_url":base,"requests":total,"concurrency":concurrency,
 "duration_seconds":duration,"ramp_to_rps":ramp_to,"ramp_duration_seconds":ramp_duration,"elapsed_seconds":round(elapsed,3),"throughput_rps":round(total/elapsed,3),
 "status_counts":statuses,"successful_requests":len(lat),
 "client_latency_ms":{"p50":pct(.5),"p95":pct(.95),"p99":pct(.99),
 "max":round(max(lat),3) if lat else None,"mean":round(statistics.fmean(lat),3) if lat else None}}, indent=2))
for path,key in (("/api/state","controller_state"),("/api/metrics","gateway_metrics")):
    try:
        with urllib.request.urlopen(base+path,timeout=5) as r: print(json.dumps({key:json.load(r)},indent=2))
    except Exception as exc: print(json.dumps({"telemetry_error":str(exc)}))
PY
