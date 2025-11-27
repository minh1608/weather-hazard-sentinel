# Weather & Hazard Sentinel

Multi-agent weather & hazard monitoring for real-time emergency readiness.

This project builds an end-to-end “Weather & Hazard Sentinel” for the American Red Cross style workflow:
AI agents ingest live weather data, assess hazards, and generate daily readiness briefings for regional leadership.

## Features

- **Data Ingestion Agent** – pulls live weather data from OpenWeather for key subregions.
- **Hazard Interpretation Agent** – applies rule-based logic, with optional Gemini support for richer context.
- **Trigger Evaluation Agent** – compares hazards against threshold rules (e.g., flash flood “Enhanced Monitoring” / “Response Consideration”).
- **Briefing Agent** – generates a concise leadership brief (LLM if available, rule-based fallback otherwise).
- **Memory & Logging Agent** – logs runs, risks, triggers, and briefs for dashboards and audits.
- **Orchestrator** – coordinates the full monitoring cycle.

## Files

- `agents.py` – core multi-agent orchestration and logic.
- `main.py` – FastAPI wrapper exposing:
  - `GET /` – healthcheck
  - `POST /run` – run one monitoring cycle
- `Dockerfile` – container for Cloud Run deployment.

## Deployment (Cloud Run)

1. Build and push the image:

```bash
gcloud builds submit --tag \
  us-central1-docker.pkg.dev/PROJECT_ID/sentinel-repo/sentinel:latest
