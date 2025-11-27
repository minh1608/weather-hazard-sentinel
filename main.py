
# ===== main.py (FastAPI wrapper for Cloud Run) =====
"""
HTTP wrapper around orchestrator_cloud_run_cycle().

- GET "/"   → healthcheck
- POST "/run" → run one monitoring cycle (for Cloud Scheduler)
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from agents import orchestrator_cloud_run_cycle

app = FastAPI(
    title="Weather & Hazard Sentinel",
    description="Red Cross Weather & Hazard Monitoring Orchestrator",
    version="1.0.0",
)

@app.get("/")
def healthcheck():
    return {"status": "ok", "service": "weather-hazard-sentinel"}

@app.post("/run")
def run_cycle():
    try:
        result = orchestrator_cloud_run_cycle()
        return JSONResponse(status_code=200, content={"status": "ok", "result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
