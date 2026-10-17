"""HTTP workload gateway and worker used by the live Kubernetes PPO demo."""
import asyncio
from collections import deque
import json
import os
import socket
import time
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from .ppo_agent import PPOAgent
from .ppo_env import ACTION_NAMES, MAX_REPLICAS, MIN_REPLICAS, make_observation

ROLE = os.getenv("APP_ROLE", "gateway")
MODEL = os.getenv("PPO_MODEL", "/app/models/ppo_autoscaler.npz")
WINDOW = float(os.getenv("CONTROL_INTERVAL_SECONDS", "5"))
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
latest_action = "hold"
last_error = None
apps_api = None
control_task = None
worker_client = None


async def control_loop():
    global hits, desired, ready, latest_rps, latest_action, last_error
    while True:
        await asyncio.sleep(WINDOW)
        count, hits = hits, 0
        latest_rps = count / WINDOW
        history.append(latest_rps)
        try:
            state = apps_api.read_namespaced_deployment(DEPLOYMENT, NAMESPACE)
            desired = int(state.spec.replicas or 1)
            ready = int(state.status.ready_replicas or 0)
            pending = max(0, desired - ready)
            obs = make_observation(history, desired, pending)
            action, _, _ = POLICY.act(obs, deterministic=True)
            latest_action = ACTION_NAMES[action]
            target = desired + (1 if action == 2 else -1 if action == 0 else 0)
            target = max(MIN_REPLICAS, min(MAX_REPLICAS, target))
            if target != desired:
                apps_api.patch_namespaced_deployment_scale(
                    DEPLOYMENT, NAMESPACE, {"spec": {"replicas": target}})
                desired = target
            last_error = None
            print(json.dumps({"rps": latest_rps, "action": latest_action,
                "desired_replicas": desired, "ready_replicas": ready,
                "pending_replicas": pending}), flush=True)
        except Exception as exc:
            last_error = str(exc)
            print(f"PPO control loop error: {exc}", flush=True)


@asynccontextmanager
async def lifespan(_app):
    global apps_api, control_task, worker_client
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
    return {"role": ROLE, "rps_last_window": latest_rps,
        "desired_replicas": desired, "ready_replicas": ready,
        "action": latest_action, "last_error": last_error,
        "interval_seconds": WINDOW}


@app.get("/", response_class=HTMLResponse)
def dashboard():
    if ROLE != "gateway": return {"role": ROLE, "pod": socket.gethostname()}
    return """<!doctype html><html><head><meta charset=utf-8><title>PPO Autoscaler</title>
<style>body{font:16px system-ui;background:#101827;color:#e5edf7;margin:3rem}main{max-width:780px;margin:auto}section{background:#1d2a3e;padding:1.2rem;margin:1rem 0;border-radius:12px}.n{font-size:2rem;color:#54d6a0}button{padding:.8rem 1.2rem;background:#4ade80;border:0;border-radius:8px;cursor:pointer}pre{white-space:pre-wrap}</style></head>
<body><main><h1>PPO Kubernetes Autoscaler</h1><p>Live request traffic is forwarded to worker pods. PPO changes the worker Deployment through the Kubernetes API.</p>
<section><button onclick="send()">Send 400 requests</button><button onclick="burst()">Run 700-request burst</button><p id=s>Loading…</p><pre id=log></pre></section></main>
<script>async function refresh(){let r=await fetch('/api/state');let x=await r.json();document.querySelector('#s').innerHTML=`<span class=n>${x.rps_last_window.toFixed(1)} RPS</span> · ${x.ready_replicas}/${x.desired_replicas} ready/desired · ${x.action}`;document.querySelector('#log').textContent=x.last_error||'Controller connected'}
async function send(){for(let i=0;i<400;i++)fetch('/hit',{method:'POST'});refresh()}async function burst(){for(let i=0;i<700;i++)fetch('/hit',{method:'POST'});setTimeout(refresh,5500)}setInterval(refresh,1500);refresh()</script></body></html>"""


@app.post("/hit")
async def hit():
    global hits
    if ROLE != "gateway": raise HTTPException(404)
    hits += 1
    try:
        response = await worker_client.post(WORKER_URL)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        raise HTTPException(502, f"worker request failed: {exc}") from exc


@app.post("/work")
def work():
    # Deliberate lightweight request handler so pod identity is visible in responses.
    return {"worker_pod": socket.gethostname(), "served_at": time.time()}
