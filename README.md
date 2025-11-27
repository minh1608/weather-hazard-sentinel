# Weather & Hazard Sentinel
### Multi-Agent Monitoring for Real-Time Emergency Readiness

Weather & Hazard Sentinel is a multi-agent system that ingests live weather data, interprets hazards, evaluates readiness triggers, and generates clear daily briefings for emergency management teams.

## Why this project exists
Emergency readiness often fails due to delayed or inconsistent situational awareness. This project automates the entire monitoring workflow to reduce cognitive load and ensure nothing is missed.

## System Overview
Agents:
- Data Ingestion Agent
- Hazard Interpretation Agent
- Trigger Evaluation Agent
- Briefing Agent (LLM optional)
- Memory & Logging Agent
- Orchestrator

## Repository Structure
agents.py  
main.py  
Dockerfile  
README.md  

## Run locally
pip install -r requirements.txt  
python main.py

## Deploy to Cloud Run
gcloud builds submit --tag us-central1-docker.pkg.dev/PROJECT_ID/sentinel-repo/sentinel:latest  
gcloud run deploy sentinel-service --image us-central1-docker.pkg.dev/PROJECT_ID/sentinel-repo/sentinel:latest --region us-central1 --platform managed --allow-unauthenticated --port 8080

## Test
curl https://YOUR_SERVICE_URL/  
curl -X POST https://YOUR_SERVICE_URL/run  
