"""Tiny HTTP service for optional Docker connectivity checks."""
from datetime import datetime, timezone
from fastapi import FastAPI

app = FastAPI(title="Predictive Autoscaling Demo", version="0.1.0")

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}

@app.get("/")
def root():
    return {"project": "workload-aware-predictive-autoscaling", "mode": "prototype"}
