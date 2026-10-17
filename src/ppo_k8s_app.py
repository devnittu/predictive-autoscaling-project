"""HTTP workload gateway and worker used by the live Kubernetes PPO demo."""
import asyncio
import csv
from collections import deque
import json
import math
import os
from pathlib import Path
import socket
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .ppo_agent import PPOAgent
from .ppo_env import ACTION_NAMES, MAX_REPLICAS, MIN_REPLICAS, make_observation
from .live_simulator import ContinuousSimulator

ROOT = Path(__file__).resolve().parents[1]
ROLE = os.getenv("APP_ROLE", "gateway")
MODEL = os.getenv("PPO_MODEL", str(ROOT / "models" / "ppo_autoscaler.npz"))
WINDOW = float(os.getenv("CONTROL_INTERVAL_SECONDS", "5"))
STARTUP_DELAY = float(os.getenv("STARTUP_DELAY_SECONDS", "10"))
MAX_WORKER_REPLICAS = int(os.getenv("MAX_REPLICAS", str(MAX_REPLICAS)))
WORKER_URL = os.getenv("WORKER_URL", "http://ppo-workers:8001/work")
WORKER_HEALTH_URL = os.getenv("WORKER_HEALTH_URL", "http://ppo-workers:8001/health")
DEPLOYMENT = os.getenv("WORKER_DEPLOYMENT", "ppo-workers")
NAMESPACE = os.getenv("POD_NAMESPACE", "default")
POLICY = PPOAgent.load(MODEL) if ROLE == "gateway" else None
history = deque([50.0] * 6, maxlen=6)
hits = 0
desired = 1
ready = 1
latest_rps = 0.0
forecast_rps = 0.0
latest_action = "hold"
last_error = None
state_history = deque(maxlen=120)
event_history = deque(maxlen=40)
apps_api = None
control_task = None
worker_client = None
latency_samples_ms = deque(maxlen=10000)
request_count = 0
request_errors = 0
simulator = (ContinuousSimulator(
                 ROOT / "data" / "workload.csv", WINDOW, STARTUP_DELAY)
             if ROLE == "simulator" else None)


async def control_loop():
    global hits, desired, ready, latest_rps, forecast_rps, latest_action, last_error
    while True:
        await asyncio.sleep(WINDOW)
        count, hits = hits, 0
        latest_rps = count / WINDOW
        previous_rps = history[-1] if history else latest_rps
        history.append(latest_rps)
        horizon = max(1, math.ceil(STARTUP_DELAY / WINDOW))
        trend = latest_rps - previous_rps
        forecast_rps = max(0.0, latest_rps + trend * horizon)
        try:
            state = await asyncio.to_thread(
                apps_api.read_namespaced_deployment, DEPLOYMENT, NAMESPACE)
            desired = int(state.spec.replicas or 1)
            ready = int(state.status.ready_replicas or 0)
            pending = max(0, desired - ready)
            obs = make_observation(history, desired, pending)
            action, _, _ = POLICY.act(obs, deterministic=True)
            latest_action = ACTION_NAMES[action]
            target = desired + (1 if action == 2 else -1 if action == 0 else 0)
            target = max(MIN_REPLICAS, min(MAX_WORKER_REPLICAS, target))
            forecast_target = max(
                MIN_REPLICAS,
                min(MAX_WORKER_REPLICAS, math.ceil(forecast_rps / 60)),
            )
            if forecast_target > target:
                target = forecast_target
                latest_action = "forecast_scale_up"
            emergency = latest_rps > ready * 60
            if emergency:
                target = max(target, min(MAX_WORKER_REPLICAS, max(
                    MIN_REPLICAS, math.ceil(latest_rps / 60))))
                latest_action = "emergency_scale_up"
            if target != desired:
                await asyncio.to_thread(
                    apps_api.patch_namespaced_deployment_scale,
                    DEPLOYMENT, NAMESPACE, {"spec": {"replicas": target}})
                desired = target
            pending = max(0, desired - ready)
            last_error = None
            state_history.append({
                "timestamp": time.time(),
                "rps": latest_rps,
                "desired_replicas": desired,
                "ready_replicas": ready,
                "pending_replicas": pending,
                "capacity": ready * 60,
                "forecast_rps": forecast_rps,
                "forecast_target_replicas": forecast_target,
                "action": latest_action,
                "emergency": emergency,
                "error": None,
            })
            event_history.appendleft({
                "timestamp": time.time(),
                "type": "scale_decision",
                "message": f"{latest_action} at {latest_rps:.1f} RPS",
            })
            print(json.dumps({"rps": latest_rps, "action": latest_action,
                "desired_replicas": desired, "ready_replicas": ready,
                "pending_replicas": pending}), flush=True)
        except Exception as exc:
            last_error = str(exc)
            event_history.appendleft({
                "timestamp": time.time(),
                "type": "error",
                "message": last_error,
            })
            print(f"PPO control loop error: {exc}", flush=True)


@asynccontextmanager
async def lifespan(_app):
    global apps_api, control_task, worker_client
    if ROLE == "simulator":
        simulator.start()
        yield
        simulator.stop()
        return
    if ROLE == "gateway":
        from kubernetes import client, config
        config.load_incluster_config()
        apps_api = client.AppsV1Api()
        worker_client = httpx.AsyncClient(timeout=4, limits=httpx.Limits(max_connections=200, max_keepalive_connections=100))
        control_task = asyncio.create_task(control_loop())
    yield
    if control_task:
        control_task.cancel()
    if worker_client:
        await worker_client.aclose()


app = FastAPI(title="PPO Kubernetes Autoscaler Demo", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "role": ROLE}


@app.get("/ready")
async def readiness():
    """Report ready only when the gateway can reach a worker pod."""
    if ROLE == "gateway":
        if worker_client is None:
            raise HTTPException(503, "worker client is not initialized")
        try:
            response = await worker_client.get(WORKER_HEALTH_URL)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(503, f"worker service is not ready: {exc}") from exc
    return {"status": "ready", "role": ROLE}


@app.get("/api/state")
def get_state():
    if ROLE == "simulator":
        point = simulator.snapshot()["last"] or {}
        snapshot = simulator.snapshot()
        return {
            "role": ROLE, "running": simulator.snapshot()["running"],
            "rps_last_window": point.get("actual_rps", 0.0),
            "predicted_rps": point.get("predicted_rps", 0.0),
            "desired_replicas": point.get("desired_replicas", 1),
            "ready_replicas": point.get("ready_replicas", 1),
            "pending_replicas": point.get("pending_replicas", 0),
            "capacity_rps": point.get("capacity_rps", 60.0),
            "action": point.get("action", "hold"),
            "overload_rps": point.get("overload_rps", 0.0),
            "scenario": point.get("scenario", simulator.scenario),
            "load_control": snapshot["load_control"],
            "startup_delay_seconds": snapshot["startup_delay_seconds"],
            "startup_delay_steps": snapshot["startup_delay_steps"],
            "decision": point.get("decision", {}),
            "utilization": point.get("utilization", 0.0),
            "forecast_error_rps": point.get("forecast_error_rps", 0.0),
            "model_prediction_rps": point.get("model_prediction_rps", 0.0),
            "recent_observation_prediction_rps": point.get(
                "recent_observation_prediction_rps", 0.0),
            "online_residual_rps": point.get("online_residual_rps", 0.0),
            "interval_seconds": WINDOW, "timestamp": time.time(),
        }
    metrics = get_metrics()
    return {"role": ROLE, "rps_last_window": latest_rps,
        "forecast_rps": forecast_rps,
        "desired_replicas": desired, "ready_replicas": ready,
        "action": latest_action, "last_error": last_error,
        "pending_replicas": max(0, desired - ready),
        "capacity_rps": ready * 60,
        "overload_rps": max(0.0, latest_rps - ready * 60),
        "utilization": latest_rps / max(1, ready * 60),
        "request_count": metrics["requests_completed"],
        "request_errors": metrics["request_errors"],
        "latency_ms": metrics["latency_ms"],
        "interval_seconds": WINDOW,
        "timestamp": time.time()}


@app.get("/api/history")
def get_history():
    if ROLE == "simulator":
        snapshot = simulator.snapshot()
        return {"points": snapshot["points"], "events": snapshot["events"]}
    return {"points": list(state_history), "events": list(event_history)}


@app.post("/api/load")
def start_manual_load(payload: dict = Body(...)):
    if ROLE != "simulator":
        raise HTTPException(409, "manual load control is available only in simulator mode")
    try:
        return {
            "status": "started",
            "load_control": simulator.set_load(
                payload.get("mode", "spike"),
                payload.get("rate_rps"),
                payload.get("duration_seconds"),
            ),
        }
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.post("/api/load/stop")
def stop_manual_load():
    if ROLE != "simulator":
        raise HTTPException(409, "manual load control is available only in simulator mode")
    return {"status": "stopped", "load_control": simulator.clear_load()}


@app.get("/api/results")
def get_results():
    path = ROOT / "results" / "ppo_scaling_comparison.csv"
    if not path.exists():
        return {"available": False, "rows": []}
    with path.open(newline="", encoding="utf-8") as stream:
        return {"available": True, "rows": list(csv.DictReader(stream))}


@app.get("/details", response_class=HTMLResponse)
def detailed_dashboard():
    if ROLE != "simulator":
        return """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>LinuxONE Autoscaler Metrics</title><style>
:root{color-scheme:dark;--bg:#08111f;--panel:#14233a;--line:#2a4262;--muted:#9db0cb;--blue:#70a9ff;--green:#4ce0a5;--amber:#ffc565;--red:#ff7777}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#eef6ff;font:14px system-ui}main{max-width:1450px;margin:auto;padding:24px 18px}header{display:flex;justify-content:space-between;align-items:start;gap:20px}h1{margin:0;font-size:clamp(1.5rem,3vw,2.4rem)}h2{font-size:1rem;margin:0 0 12px}.muted{color:var(--muted)}.pill{border:1px solid var(--line);border-radius:99px;padding:7px 12px;color:var(--green)}.grid{display:grid;grid-template-columns:repeat(8,1fr);gap:10px;margin:20px 0 12px}.card,section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}.label{font-size:.68rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}.value{font-size:1.3rem;font-weight:700;margin-top:7px}.layout{display:grid;grid-template-columns:1.1fr .9fr;gap:12px}section{margin-top:12px}pre{white-space:pre-wrap;word-break:break-word;background:#0c1829;border-radius:8px;padding:12px;margin:0;color:#d7e7ff;line-height:1.45}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid var(--line);padding:7px}th{color:var(--muted);font-size:.68rem;text-transform:uppercase}.scroll{max-height:360px;overflow:auto}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}@media(max-width:1050px){.grid{grid-template-columns:repeat(4,1fr)}}@media(max-width:650px){.grid{grid-template-columns:repeat(2,1fr)}.layout{display:block}}
</style></head><body><main><header><div><h1>LinuxONE Autoscaler Metrics</h1><div class=muted>Live Kubernetes gateway · client traffic · worker latency · controller decisions</div></div><div id=status class=pill>Connecting…</div></header>
<div class=grid><div class=card><div class=label>Actual RPS</div><div id=actual class=value>—</div><div id=window class=muted>window —</div></div><div class=card><div class=label>Client requests</div><div id=requests class=value>—</div><div id=errorRate class=muted>error rate —</div></div><div class=card><div class=label>Worker P50</div><div id=p50 class=value>—</div><div class=muted>ms</div></div><div class=card><div class=label>Worker P95</div><div id=p95 class=value>—</div><div class=muted>ms</div></div><div class=card><div class=label>Worker P99</div><div id=p99 class=value>—</div><div class=muted>ms</div></div><div class=card><div class=label>Worker max</div><div id=maxlat class=value>—</div><div class=muted>ms</div></div><div class=card><div class=label>Ready / Desired</div><div id=replicas class=value>—</div><div id=pending class=muted>pending —</div></div><div class=card><div class=label>Overload</div><div id=overload class=value>—</div><div id=util class=muted>utilization —</div></div></div>
<div class=layout><div><section><h2>Live traffic and capacity</h2><canvas id=chart height=260></canvas></section><section><h2>Live controller state</h2><pre id=state>Waiting…</pre></section><section><h2>Decision timeline</h2><div class=scroll><table><thead><tr><th>Time</th><th>RPS</th><th>Desired</th><th>Ready</th><th>Pending</th><th>Capacity</th><th>Action</th><th>Error</th></tr></thead><tbody id=rows></tbody></table></div></section></div><div><section><h2>Live gateway metrics</h2><pre id=metrics>Waiting…</pre></section><section><h2>Model and capacity detail</h2><pre id=detail>Waiting…</pre></section><section><h2>Scaling events</h2><div id=events class=scroll>Waiting…</div></section><section><h2>How to drive traffic</h2><div class=muted>Run <code>scripts/run_remote_experiment.ps1 -BaseUrl http://&lt;linuxone&gt;:8000 -Concurrency 100 -DurationSeconds 60</code> from the client machine. Client P95/P99 are printed by the runner; worker P50/P95/P99 are shown here.</div></section></div></div>
<script>
const $=id=>document.getElementById(id),fmt=x=>JSON.stringify(x,null,2);
async function get(path){let r=await fetch(path,{cache:'no-store'});let x=await r.json();if(!r.ok)throw Error(x.detail||path);return x}
function draw(points){let c=$('chart'),d=devicePixelRatio||1,w=c.clientWidth,h=260;c.width=w*d;c.height=h*d;let x=c.getContext('2d');x.scale(d,d);x.clearRect(0,0,w,h);if(!points.length)return;let max=Math.max(1,...points.flatMap(p=>[Number(p.rps)||0,Number(p.capacity)||0]))*1.15,px=i=>10+i*(w-20)/Math.max(1,points.length-1),py=v=>h-25-v/max*(h-45);x.strokeStyle='#2a4262';for(let i=0;i<4;i++){x.beginPath();x.moveTo(0,py(max*i/3));x.lineTo(w,py(max*i/3));x.stroke()}[['rps','#70a9ff','actual RPS'],['capacity','#4ce0a5','ready capacity']].forEach(([k,col,name])=>{x.strokeStyle=col;x.lineWidth=2;x.beginPath();points.forEach((p,i)=>i?x.lineTo(px(i),py(Number(p[k])||0)):x.moveTo(px(i),py(Number(p[k])||0)));x.stroke();x.fillStyle=col;x.font='11px system-ui';x.fillText(name,12+(['rps','capacity'].indexOf(k)*110),15)})}
async function refresh(){try{let[s,h,m]=await Promise.all([get('/api/state'),get('/api/history'),get('/api/metrics')]);let p=h.points||[];$('status').textContent='● Live · '+new Date().toLocaleTimeString();$('actual').textContent=Number(s.rps_last_window).toFixed(1);$('window').textContent='window '+Number(s.interval_seconds).toFixed(1)+'s';$('requests').textContent=m.requests_completed;$('errorRate').textContent='errors '+m.request_errors+' · samples '+m.latency_samples;$('p50').textContent=m.latency_ms.p50??'—';$('p95').textContent=m.latency_ms.p95??'—';$('p99').textContent=m.latency_ms.p99??'—';$('maxlat').textContent=m.latency_ms.max??'—';$('replicas').textContent=s.ready_replicas+' / '+s.desired_replicas;$('pending').textContent='pending '+s.pending_replicas;$('overload').textContent=Number(s.overload_rps).toFixed(1);$('util').textContent='utilization '+(Number(s.utilization)*100).toFixed(1)+'%';$('state').textContent=fmt(s);$('metrics').textContent=fmt(m);$('detail').textContent=fmt({action:s.action,forecast_rps:s.forecast_rps,observed_rps:s.rps_last_window,forecast_error_rps:Number(s.rps_last_window)-Number(s.forecast_rps),capacity_rps:s.capacity_rps,headroom_rps:Math.max(0,s.capacity_rps-s.rps_last_window),desired_replicas:s.desired_replicas,ready_replicas:s.ready_replicas,pending_replicas:s.pending_replicas,last_error:s.last_error,updated_at:new Date(s.timestamp*1000).toISOString()});draw(p);$('rows').innerHTML=p.slice(-40).reverse().map(x=>'<tr><td>'+new Date(x.timestamp*1000).toLocaleTimeString()+'</td><td>'+Number(x.rps||0).toFixed(1)+'</td><td>'+x.desired_replicas+'</td><td>'+x.ready_replicas+'</td><td>'+Math.max(0,(x.desired_replicas||0)-(x.ready_replicas||0))+'</td><td>'+Number(x.capacity||0).toFixed(0)+'</td><td>'+x.action+'</td><td>'+((x.error||''))+'</td></tr>').join('');$('events').innerHTML=(h.events||[]).map(x=>'<div class=muted>'+new Date(x.timestamp*1000).toLocaleTimeString()+' · '+x.type+' · '+x.message+'</div>').join('')||'No events'}catch(e){$('status').textContent='● Disconnected';$('status').className='pill bad';$('state').textContent=e.message}}
refresh();setInterval(refresh,1000);
</script></main></body></html>"""
    return """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Autoscaler Decision Lab</title><style>
:root{color-scheme:dark;--bg:#08111f;--panel:#14233a;--line:#2a4262;--muted:#9db0cb;--blue:#70a9ff;--green:#4ce0a5;--amber:#ffc565;--red:#ff7777}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#eef6ff;font:14px system-ui}main{max-width:1400px;margin:auto;padding:24px 18px}header{display:flex;justify-content:space-between;gap:20px;align-items:start}h1{margin:0;font-size:clamp(1.5rem,3vw,2.4rem)}h2{font-size:1rem;margin:0 0 14px}.muted{color:var(--muted)}.pill{border:1px solid var(--line);border-radius:99px;padding:7px 12px;color:var(--green)}.grid{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;margin:20px 0 12px}.card,section{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px}.label{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}.value{font-size:1.35rem;font-weight:700;margin-top:7px}.layout{display:grid;grid-template-columns:1.1fr .9fr;gap:12px}section{margin-top:12px}pre{white-space:pre-wrap;word-break:break-word;background:#0c1829;border-radius:8px;padding:12px;margin:0;color:#d7e7ff;line-height:1.45}table{width:100%;border-collapse:collapse}td,th{text-align:left;border-bottom:1px solid var(--line);padding:8px}th{color:var(--muted);font-size:.7rem;text-transform:uppercase}button,input,select{border:1px solid var(--line);border-radius:7px;padding:8px;background:#0c1829;color:#eef6ff;margin:0 6px 6px 0}button{background:var(--blue);color:#08111f;font-weight:700;cursor:pointer}.danger{background:#573044;color:#fff}.scroll{max-height:300px;overflow:auto}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}@media(max-width:900px){.grid{grid-template-columns:repeat(3,1fr)}.layout{display:block}}@media(max-width:520px){.grid{grid-template-columns:repeat(2,1fr)}}
</style></head><body><main><header><div><h1>Autoscaler Decision Lab</h1><div class=muted>Detailed control-loop telemetry · forecast → target → provisioning → ready capacity</div></div><div id=status class=pill>Connecting…</div></header>
<div class=grid><div class=card><div class=label>Actual demand</div><div id=actual class=value>—</div><div class=muted>RPS</div></div><div class=card><div class=label>Forecast</div><div id=forecast class=value>—</div><div class=muted>next interval RPS</div></div><div class=card><div class=label>Desired</div><div id=desired class=value>—</div><div class=muted>controller target</div></div><div class=card><div class=label>Ready</div><div id=ready class=value>—</div><div class=muted>serving capacity</div></div><div class=card><div class=label>Provisioning</div><div id=pending class=value>—</div><div class=muted>not ready yet</div></div><div class=card><div class=label>Overload</div><div id=overload class=value>—</div><div class=muted>excess RPS</div></div></div>
<section><h2>Manual load controller</h2><span class=muted>Bounded synthetic demand, not real client traffic. </span><select id=mode><option value=spike>Spike</option><option value=steady>Steady</option></select><input id=rate type=number value=300 min=0 max=600 step=10 aria-label="RPS"><input id=duration type=number value=20 min=1 max=300 aria-label="duration seconds"><button id=start>Start load</button><button id=stop class=danger>Stop load</button><span id=loadStatus class=muted></span></section>
<div class=layout><div><section><h2>Model decision explanation</h2><pre id=decision>Waiting…</pre></section><section><h2>Latest control-loop record</h2><pre id=latest>Waiting…</pre></section><section><h2>Decision timeline</h2><div class=scroll><table><thead><tr><th>Time</th><th>Actual</th><th>Forecast</th><th>Desired</th><th>Ready</th><th>Pending</th><th>Capacity</th><th>Action</th></tr></thead><tbody id=rows></tbody></table></div></section></div><div><section><h2>Capacity and timing</h2><pre id=capacity>Waiting…</pre></section><section><h2>Scaling events</h2><div id=events class=scroll>Waiting…</div></section><section><h2>What the numbers mean</h2><div class=muted>Desired replicas are the controller’s requested count. Provisioning replicas are created but not ready. Ready replicas alone add capacity. Startup delay defaults to 10 seconds, so a spike can produce overload before capacity catches up. Forecast error is actual minus predicted RPS.</div></section></div></div>
<script>
const $=id=>document.getElementById(id),fmt=x=>JSON.stringify(x,null,2);
async function load(path,options){let r=await fetch(path,options);let x=await r.json();if(!r.ok)throw Error(x.detail||'request failed');return x}
async function refresh(){try{let[s,h]=await Promise.all([load('/api/state'),load('/api/history')]);let p=h.points||[],last=p[p.length-1]||{};$('status').textContent=s.running?'● Running':'● Stopped';$('actual').textContent=Number(s.rps_last_window).toFixed(1);$('forecast').textContent=Number(s.predicted_rps).toFixed(1);$('desired').textContent=s.desired_replicas;$('ready').textContent=s.ready_replicas;$('pending').textContent=s.pending_replicas;$('overload').textContent=Number(s.overload_rps).toFixed(1);$('decision').textContent=fmt({forecast_rps:s.predicted_rps,target_replicas:s.decision.target_replicas,previous_desired_replicas:s.decision.previous_desired_replicas,new_desired_replicas:s.decision.new_desired_replicas,reason:s.decision.reason,action:s.action});$('capacity').textContent=fmt({capacity_rps:s.capacity_rps,utilization:s.utilization,forecast_error_rps:s.forecast_error_rps,startup_delay_seconds:s.startup_delay_seconds,startup_delay_steps:s.startup_delay_steps,interval_seconds:s.interval_seconds,load_control:s.load_control});$('latest').textContent=fmt(last);$('rows').innerHTML=p.slice(-30).reverse().map(x=>'<tr><td>'+new Date(x.timestamp*1000).toLocaleTimeString()+'</td><td>'+Number(x.actual_rps).toFixed(1)+'</td><td>'+Number(x.predicted_rps).toFixed(1)+'</td><td>'+x.desired_replicas+'</td><td>'+x.ready_replicas+'</td><td>'+x.pending_replicas+'</td><td>'+Number(x.capacity_rps).toFixed(0)+'</td><td>'+x.action+'</td></tr>').join('');$('events').innerHTML=(h.events||[]).map(x=>'<div class=muted>'+new Date(x.timestamp*1000).toLocaleTimeString()+' · '+x.type+' · '+x.message+'</div>').join('')||'No events'}catch(e){$('status').textContent='Disconnected';$('latest').textContent=e.message}}
$('start').onclick=async()=>{try{let x=await load('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:$('mode').value,rate_rps:Number($('rate').value),duration_seconds:Number($('duration').value)})});$('loadStatus').textContent=' Running '+x.load_control.rate_rps+' RPS'}catch(e){$('loadStatus').textContent=' Error: '+e.message}};
$('stop').onclick=async()=>{let x=await load('/api/load/stop',{method:'POST'});$('loadStatus').textContent=' '+x.load_control.mode+' mode'};refresh();setInterval(refresh,1000);
</script></main></body></html>"""


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if ROLE == "simulator":
        return """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>Continuous Autoscaling Simulator</title>
<style>:root{color-scheme:dark;--bg:#09111f;--panel:#152238;--line:#2b405e;--muted:#9db0cb;--green:#4ce0a5;--blue:#70a9ff;--amber:#ffc565;--red:#ff7777}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:#eff6ff;font:15px system-ui}main{max-width:1150px;margin:auto;padding:26px 18px}header{display:flex;justify-content:space-between;align-items:start}h1{margin:0 0 5px;font-size:clamp(1.5rem,3vw,2.3rem)}.muted{color:var(--muted)}.pill{border:1px solid var(--line);border-radius:99px;padding:7px 12px;color:var(--green)}.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin:22px 0 12px}.card,section{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:15px}.label{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}.value{font-size:1.45rem;font-weight:700;margin-top:8px}.charts{display:grid;grid-template-columns:1.5fr 1fr;gap:12px}canvas{width:100%;height:270px;background:#101b2d;border-radius:9px;margin-top:10px}section{margin-top:12px}button{border:0;border-radius:8px;padding:9px 12px;margin:0 8px 8px 0;background:var(--blue);color:#07101e;font-weight:700;cursor:pointer}button.secondary{background:#293953;color:#eaf2ff}input,select{background:#0d1728;color:#eff6ff;border:1px solid var(--line);border-radius:7px;padding:8px;margin:0 6px 8px 0;width:110px}td,th{text-align:left;padding:8px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:.75rem;text-transform:uppercase}@media(max-width:800px){.grid,.charts{grid-template-columns:1fr 1fr}.charts{display:block}}@media(max-width:500px){.grid{grid-template-columns:1fr 1fr}header{display:block}.pill{display:inline-block;margin-top:12px}}</style></head><body><main><header><div><h1>Continuous Predictive Autoscaling</h1><div class=muted>Standalone laptop simulation · live forecast → replica decision → capacity</div></div><div id=status class=pill>Starting…</div></header><div class=grid><div class=card><div class=label>Actual demand</div><div id=actual class=value>—</div><div class=muted>RPS</div></div><div class=card><div class=label>Prediction</div><div id=prediction class=value>—</div><div class=muted>next interval RPS</div></div><div class=card><div class=label>Replicas</div><div id=replicas class=value>—</div><div class=muted>ready / desired</div></div><div class=card><div class=label>Capacity</div><div id=capacity class=value>—</div><div class=muted>ready RPS</div></div><div class=card><div class=label>Action</div><div id=action class=value>—</div><div id=scenario class=muted>—</div></div></div><section><div class=label>Manual load controller</div><div class=muted>Override demand for a bounded test window. Maximum 600 RPS and 300 seconds.</div><p><select id=loadMode><option value=spike>Spike</option><option value=steady>Steady</option></select><input id=loadRate type=number min=0 max=600 value=250 step=10 aria-label="RPS"><input id=loadDuration type=number min=1 max=300 value=20 step=1 aria-label="duration seconds"><button id=startLoadButton>Start load</button><button id=stopLoadButton class=secondary>Stop load</button><span id=loadStatus class=muted></span></p></section><div class=charts><section><div class=label>Actual demand and forecast</div><canvas id=traffic></canvas></section><section><div class=label>Replica scaling</div><canvas id=replicasChart></canvas></section></div><section><div class=label>Scaling events</div><div id=events class=muted>Waiting for simulator…</div></section></main><script>
const $=id=>document.getElementById(id);function draw(id,ps,ss){let c=$(id),d=devicePixelRatio||1,w=c.clientWidth,h=270;c.width=w*d;c.height=h*d;let x=c.getContext('2d');x.scale(d,d);x.clearRect(0,0,w,h);if(!ps.length)return;let max=Math.max(1,...ss.flatMap(s=>ps.map(p=>Number(p[s.k])||0)))*1.15,px=i=>10+i*(w-20)/Math.max(1,ps.length-1),py=v=>h-20-v/max*(h-38);x.strokeStyle='#2b405e';for(let i=0;i<4;i++){x.beginPath();x.moveTo(0,py(max*i/3));x.lineTo(w,py(max*i/3));x.stroke()}ss.forEach((s,j)=>{x.strokeStyle=s.c;x.lineWidth=2;x.beginPath();ps.forEach((p,i)=>i?x.lineTo(px(i),py(Number(p[s.k])||0)):x.moveTo(px(i),py(Number(p[s.k])||0)));x.stroke();x.fillStyle=s.c;x.font='11px system-ui';x.fillText(s.n,12+j*105,15)})}async function startLoad(){let rate=Number($('loadRate').value),duration=Number($('loadDuration').value);$('loadStatus').textContent=' Starting…';try{let r=await fetch('/api/load',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode:$('loadMode').value,rate_rps:rate,duration_seconds:duration})});let data=await r.json();if(!r.ok)throw Error(data.detail||'load rejected');$('loadStatus').textContent=' Running '+rate+' RPS for '+duration+'s'}catch(e){$('loadStatus').textContent=' Error: '+e.message}}async function stopLoad(){let r=await fetch('/api/load/stop',{method:'POST'});let data=await r.json();$('loadStatus').textContent=' '+data.load_control.mode+' mode'}async function refresh(){try{let[sr,hr]=await Promise.all([fetch('/api/state'),fetch('/api/history')]);let s=await sr.json(),h=await hr.json();$('status').textContent=s.running?'● Running':'● Stopped';$('actual').textContent=Number(s.rps_last_window).toFixed(1);$('prediction').textContent=Number(s.predicted_rps).toFixed(1);$('replicas').textContent=s.ready_replicas+' / '+s.desired_replicas;$('capacity').textContent=Number(s.capacity_rps).toFixed(0);$('action').textContent=s.action.replace('_',' ');$('scenario').textContent=s.scenario+' · overload '+Number(s.overload_rps).toFixed(1)+' RPS';draw('traffic',h.points,[{k:'actual_rps',n:'actual',c:'#70a9ff'},{k:'predicted_rps',n:'forecast',c:'#ffc565'},{k:'capacity_rps',n:'capacity',c:'#4ce0a5'}]);draw('replicasChart',h.points,[{k:'desired_replicas',n:'desired',c:'#ffc565'},{k:'ready_replicas',n:'ready',c:'#4ce0a5'}]);$('events').innerHTML=h.events.length?h.events.map(e=>'<div>'+new Date(e.timestamp*1000).toLocaleTimeString()+' · '+e.type+' · '+e.message+'</div>').join(''):'No scale events yet'}catch(e){$('status').textContent='Disconnected'}}$('startLoadButton').addEventListener('click',startLoad);$('stopLoadButton').addEventListener('click',stopLoad);refresh();setInterval(refresh,1000);addEventListener('resize',refresh)
</script></body></html>"""
    if ROLE != "gateway": return {"role": ROLE, "pod": socket.gethostname()}
    return """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"><title>LinuxONE Predictive Autoscaler</title>
<style>
:root{color-scheme:dark;--bg:#0b1220;--panel:#151f32;--muted:#91a2bb;--line:#2b3a52;--green:#49d69b;--blue:#6ca8ff;--amber:#f4bd62;--red:#ff7272}
*{box-sizing:border-box}body{font:15px system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:#edf4ff;margin:0}main{max-width:1180px;margin:auto;padding:28px 18px}header{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;margin-bottom:22px}h1{font-size:clamp(1.5rem,3vw,2.3rem);margin:0 0 6px}.sub,.muted{color:var(--muted)}.pill{border:1px solid var(--line);border-radius:999px;padding:7px 11px;color:var(--green);white-space:nowrap}.grid{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:12px}.card,section{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:16px}.label{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.08em}.value{font-size:1.65rem;font-weight:700;margin-top:8px}.good{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}.charts{display:grid;grid-template-columns:1.4fr 1fr;gap:12px}canvas{width:100%;height:250px;background:#101a2b;border-radius:9px;margin-top:12px}section{margin-top:12px}button{border:0;border-radius:8px;padding:10px 14px;margin:0 8px 8px 0;background:var(--blue);color:#07101e;font-weight:700;cursor:pointer}button.secondary{background:#293953;color:#eaf2ff}button:disabled{opacity:.55;cursor:wait}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:9px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-size:.78rem;text-transform:uppercase}#events{max-height:180px;overflow:auto}.event{padding:7px 0;border-bottom:1px solid var(--line)}@media(max-width:800px){.grid,.charts{grid-template-columns:1fr 1fr}.charts{display:block}}@media(max-width:520px){.grid{grid-template-columns:1fr 1fr}header{display:block}.pill{display:inline-block;margin-top:12px}}
</style></head><body><main><header><div><h1>LinuxONE Predictive Autoscaler</h1><div class=sub>Live PPO gateway, worker capacity, and scaling telemetry</div></div><div id=connection class=pill>Connecting…</div></header>
<div class=grid><div class=card><div class=label>Traffic</div><div id=rps class=value>—</div><div class=muted>requests / second</div></div><div class=card><div class=label>Workers</div><div id=workers class=value>—</div><div class=muted>ready / desired</div></div><div class=card><div class=label>Capacity</div><div id=capacity class=value>—</div><div class=muted>ready RPS</div></div><div class=card><div class=label>Decision</div><div id=action class=value>—</div><div class=muted>latest PPO action</div></div></div>
<div class=charts><section><div class=label>Traffic and capacity</div><canvas id=traffic></canvas></section><section><div class=label>Replica history</div><canvas id=replicas></canvas></section></div>
<section><div class=label>Live workload controls</div><p class=muted>Drive real HTTP traffic from this browser. Start a gradual ramp, then adjust the target while watching forecast, ready pods, capacity, and latency.</p><p><label>Target RPS <input id=targetRps type=number min=0 max=1000 value=1000 step=50></label><label>Ramp seconds <input id=rampSeconds type=number min=10 max=900 value=180 step=10></label></p><button onclick="startRamp()">Start gradual ramp</button><button onclick="changeTarget(100)" class=secondary>Increase +100</button><button onclick="changeTarget(-100)" class=secondary>Decrease -100</button><button onclick="toggleTraffic()" class=secondary id=pauseButton>Pause</button><button onclick="stopTraffic()" class=secondary>Stop</button><div id=trafficStatus class=muted>Idle</div><hr><p class=muted>Quick tests</p><button onclick="load(100)">Send 100 requests</button><button onclick="load(400)">Send 400 requests</button><button onclick="load(700)" class=secondary>Run burst</button><span id=loadStatus class=muted></span></section>
<section><div class=label>Scaling events</div><div id=events class=muted>Waiting for the first control interval…</div></section>
<section><div class=label>Offline PPO evaluation</div><div id=results class=muted>Loading results…</div></section>
</main><script>
const $=id=>document.getElementById(id), fmtTime=t=>new Date(t*1000).toLocaleTimeString();
function draw(id,points,series){const c=$(id),d=devicePixelRatio||1,w=c.clientWidth,h=250;c.width=w*d;c.height=h*d;const x=c.getContext('2d');x.scale(d,d);x.clearRect(0,0,w,h);if(!points.length)return;const vals=series.flatMap(s=>points.map(p=>Number(p[s.key])||0)),max=Math.max(1,...vals)*1.15;const px=i=>10+i*(w-20)/Math.max(1,points.length-1),py=v=>h-20-(v/max)*(h-35);x.strokeStyle='#2b3a52';x.lineWidth=1;for(let i=0;i<4;i++){let y=py(max*i/3);x.beginPath();x.moveTo(0,y);x.lineTo(w,y);x.stroke()}series.forEach(s=>{x.strokeStyle=s.color;x.lineWidth=2;x.beginPath();points.forEach((p,i)=>{let y=py(Number(p[s.key])||0);i?x.lineTo(px(i),y):x.moveTo(px(i),y)});x.stroke()});x.font='11px system-ui';series.forEach((s,i)=>{x.fillStyle=s.color;x.fillText(s.name,12+i*105,15)})}
async function refresh(){try{let [sr,hr,rr]=await Promise.all([fetch('/api/state'),fetch('/api/history'),fetch('/api/results')]);if(!sr.ok||!hr.ok)throw Error('gateway unavailable');let s=await sr.json(),h=await hr.json();$('connection').textContent='● Connected';$('connection').className='pill';$('rps').textContent=s.rps_last_window.toFixed(1)+' RPS';$('workers').textContent=s.ready_replicas+' / '+s.desired_replicas;$('capacity').textContent=s.capacity_rps.toFixed(0)+' RPS';$('action').textContent=s.action.replace('_',' ');$('action').className='value '+(s.action==='scale_up'?'warn':s.action==='scale_down'?'good':'');draw('traffic',h.points,[{key:'rps',name:'traffic',color:'#6ca8ff'},{key:'capacity',name:'capacity',color:'#49d69b'}]);draw('replicas',h.points,[{key:'desired_replicas',name:'desired',color:'#f4bd62'},{key:'ready_replicas',name:'ready',color:'#49d69b'}]);$('events').innerHTML=h.events.length?h.events.map(e=>`<div class=event><span class=muted>${fmtTime(e.timestamp)}</span> · ${e.type} · ${e.message}</div>`).join(''):'No events yet';if(rr.ok)renderResults(await rr.json())}catch(e){$('connection').textContent='● Disconnected';$('connection').className='pill bad';$('connection').title=e.message}}
function renderResults(x){if(!x.available){$('results').textContent='No ppo_scaling_comparison.csv found. Run python -m src.evaluate_ppo first.';return}let rows=x.rows;$('results').innerHTML='<table><tr><th>Scenario</th><th>Overload intervals</th><th>Excess requests</th><th>Mean replicas</th></tr>'+rows.map(r=>`<tr><td>${r.scenario}</td><td>${r.overload_intervals}</td><td>${Number(r.estimated_excess_requests).toFixed(1)}</td><td>${Number(r.mean_active_replicas).toFixed(2)}</td></tr>`).join('')+'</table>'}
async function load(n){let b=[...document.querySelectorAll('button')];b.forEach(x=>x.disabled=true);$('loadStatus').textContent=' Sending '+n+' requests…';await Promise.all(Array.from({length:n},()=>fetch('/hit',{method:'POST'})));$('loadStatus').textContent=' Complete';b.forEach(x=>x.disabled=false);refresh()}
let trafficTimer=null,trafficRunning=false,trafficPaused=false,trafficStarted=0,trafficStartRps=0,trafficTarget=1000,trafficRampSeconds=180,trafficInFlight=0;
function setTrafficStatus(text){$('trafficStatus').textContent=text+' · target '+Math.round(trafficTarget)+' RPS · in flight '+trafficInFlight}
function launchTraffic(){if(!trafficRunning||trafficPaused)return;let elapsed=(performance.now()-trafficStarted)/1000;let progress=Math.min(1,elapsed/trafficRampSeconds);let rate=trafficStartRps+(trafficTarget-trafficStartRps)*progress;let count=Math.max(1,Math.round(rate/10));for(let i=0;i<count;i++){trafficInFlight++;fetch('/hit',{method:'POST'}).catch(()=>{}).finally(()=>{trafficInFlight=Math.max(0,trafficInFlight-1)});}setTrafficStatus(progress<1?'Ramping':'Running')}
function startRamp(){stopTraffic();trafficTarget=Math.min(1000,Math.max(0,Number($('targetRps').value)||0));trafficRampSeconds=Math.max(10,Number($('rampSeconds').value)||180);trafficStartRps=0;trafficStarted=performance.now();trafficRunning=true;trafficPaused=false;$('pauseButton').textContent='Pause';trafficTimer=setInterval(launchTraffic,100);setTrafficStatus('Ramping')}
function changeTarget(delta){trafficTarget=Math.min(1000,Math.max(0,trafficTarget+delta));$('targetRps').value=trafficTarget;if(!trafficRunning)startRamp();else setTrafficStatus(trafficPaused?'Paused':'Target changed')}
function toggleTraffic(){if(!trafficRunning)return;trafficPaused=!trafficPaused;$('pauseButton').textContent=trafficPaused?'Resume':'Pause';setTrafficStatus(trafficPaused?'Paused':'Running')}
function stopTraffic(){if(trafficTimer){clearInterval(trafficTimer);trafficTimer=null}trafficRunning=false;trafficPaused=false;trafficInFlight=0;if($('trafficStatus'))$('trafficStatus').textContent='Stopped'}
addEventListener('resize',refresh);refresh();setInterval(refresh,2000);
</script></body></html>"""


@app.post("/hit")
async def hit():
    global hits, request_count, request_errors
    if ROLE != "gateway": raise HTTPException(404)
    started = time.perf_counter()
    hits += 1
    try:
        response = await worker_client.post(WORKER_URL)
        response.raise_for_status()
        latency_samples_ms.append((time.perf_counter() - started) * 1000)
        request_count += 1
        return response.json()
    except Exception as exc:
        latency_samples_ms.append((time.perf_counter() - started) * 1000)
        request_errors += 1
        raise HTTPException(502, f"worker request failed: {exc}") from exc


@app.get("/api/metrics")
def get_metrics():
    samples = sorted(latency_samples_ms)
    def percentile(p):
        if not samples:
            return None
        index = min(len(samples) - 1, math.ceil(p * len(samples)) - 1)
        return round(samples[index], 3)
    return {
        "requests_completed": request_count,
        "request_errors": request_errors,
        "latency_samples": len(samples),
        "latency_ms": {
            "p50": percentile(0.50),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": round(samples[-1], 3) if samples else None,
        },
        "note": "gateway-to-worker HTTP latency, not client network latency",
    }


@app.post("/work")
def work():
    # Deliberate lightweight request handler so pod identity is visible in responses.
    return {"worker_pod": socket.gethostname(), "served_at": time.time()}
