
# Exported from notebook:
# Copy all code from Cells 1–15 here manually for Cloud Run.
# (Notebook users see this as reference; GitHub repo will contain full file.)

# ===== agents.py (deployment module) =====
"""
Core Weather & Hazard Sentinel logic for deployment.

This file is a refactored version of the Kaggle notebook code:
- Uses os.getenv(...) instead of kaggle_secrets.
- Omits visualization / display calls.
- Exposes `orchestrator_cloud_run_cycle()` for Cloud Run / FastAPI.
"""

import os
import json
import uuid
import datetime as dt
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import logging
import requests
import pandas as pd

# Optional: GCS storage for checkpoint
try:
    from google.cloud import storage
except Exception:
    storage = None

logging.basicConfig(level=logging.INFO)

# -------------------------------------------------------
# CONFIG (env-driven for deployment)
# -------------------------------------------------------

REGION_NAME = os.getenv("REGION_NAME", "American Red Cross | Texas Gulf Coast Region")

REGION_AREAS = [
    "Coastal Bend",
    "Houston Metro",
    "Golden Triangle",
]

REGION_AREA_COORDS = {
    "Coastal Bend": {"lat": 27.8, "lon": -97.4},
    "Houston Metro": {"lat": 29.76, "lon": -95.37},
    "Golden Triangle": {"lat": 30.08, "lon": -94.13},
}

HAZARD_TYPES = [
    "Heavy Rain & Flooding",
    "Severe Storms",
    "Excessive Heat",
    "Wildfire",
]

TRIGGERS = [
    {
        "id": "flood_enhanced_monitoring",
        "hazard": "Heavy Rain & Flooding",
        "min_likelihood": "Medium",
        "min_impact": "Disruptive",
        "recommended_posture": "Enhanced Monitoring",
        "name": "Heavy Rain – Flash Flood Watch",
        "note": "Consider readiness actions for flood-prone areas.",
    },
    {
        "id": "flood_response_consideration",
        "hazard": "Heavy Rain & Flooding",
        "min_likelihood": "High",
        "min_impact": "Dangerous",
        "recommended_posture": "Response Consideration",
        "name": "Heavy Rain – Possible Flash Flooding",
        "note": "Discuss shelter readiness and resource pre-positioning.",
    },
]

LIKELIHOOD_ORDER = ["Low", "Medium", "High"]
IMPACT_ORDER = ["Nuisance", "Disruptive", "Dangerous"]

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")
WEATHER_API_URL = os.getenv("WEATHER_API_URL", "https://api.openweathermap.org/data/2.5/weather")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
CHECKPOINT_BLOB_NAME = "weather_hazard_sentinel/checkpoint.json"

# -------------------------------------------------------
# Gemini client (optional)
# -------------------------------------------------------

try:
    import google.generativeai as genai
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        logging.info("Gemini client configured in agents.py.")
    else:
        genai = None
        logging.info("GEMINI_API_KEY not set; Gemini disabled in agents.py.")
except Exception as e:
    genai = None
    logging.warning(f"Could not import google.generativeai in agents.py: {e}")


def call_gemini(prompt: str,
                model_name: str = GEMINI_MODEL,
                temperature: float = 0.2,
                max_output_tokens: int = 512) -> str:
    if not genai or not GEMINI_API_KEY:
        return "[Gemini disabled] " + prompt[:200]

    try:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        try:
            if getattr(response, "candidates", None):
                parts = response.candidates[0].content.parts
                text = "".join(getattr(p, "text", "") for p in parts)
                if text.strip():
                    return text
        except Exception as inner:
            logging.warning(f"Could not extract Gemini text: {inner}")
        return "[Gemini raw response] " + str(response)[:400]
    except Exception as e:
        logging.error(f"Gemini call failed: {e}")
        return f"[Gemini error: {e}]"

# -------------------------------------------------------
# Dataclasses (A2A schemas)
# -------------------------------------------------------

@dataclass
class HazardInputsMessage:
    run_id: str
    as_of: dt.datetime
    areas: List[str]
    forecasts: List[Dict[str, Any]]
    bulletins: List[str]


@dataclass
class HazardRisk:
    area: str
    hazard: str
    timeframe: str
    likelihood: str
    impact: str
    rationale: str
    supporting_evidence: List[str]


@dataclass
class HazardRisksMessage:
    run_id: str
    as_of: dt.datetime
    risks: List[HazardRisk]


@dataclass
class AreaTriggerSummary:
    name: str
    posture: str
    fired_triggers: List[Dict[str, Any]]


@dataclass
class TriggerResultsMessage:
    run_id: str
    as_of: dt.datetime
    areas: List[AreaTriggerSummary]


@dataclass
class BriefPacketMessage:
    run_id: str
    as_of: dt.datetime
    markdown_brief: str
    text_brief: str
    posture_overview: Dict[str, str]


@dataclass
class CheckpointState:
    last_run_time: Optional[dt.datetime]
    last_posture_by_area: Dict[str, str]
    last_run_id: Optional[str]
    operational_period_label: str

# -------------------------------------------------------
# Weather helper
# -------------------------------------------------------

def fetch_weather_raw(lat: float,
                      lon: float,
                      units: str = "metric") -> Dict[str, Any]:
    if not WEATHER_API_KEY:
        logging.warning("No WEATHER_API_KEY set; returning empty weather.")
        return {}

    params = {"lat": lat, "lon": lon, "units": units, "appid": WEATHER_API_KEY}
    try:
        resp = requests.get(WEATHER_API_URL, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"Weather API call failed: {e}")
        return {}

# -------------------------------------------------------
# Agents: Ingestion, Hazard Interpretation, Trigger Evaluation, Briefing, Memory
# (same logic as notebook, but without Kaggle-specific bits)
# -------------------------------------------------------

class DataIngestionAgent:
    def __init__(self, region_areas: List[str]):
        self.region_areas = region_areas

    def ingest(self,
               time_window: str = "next_24_hours",
               products: Optional[List[str]] = None,
               demo_force_hazard: bool = False) -> HazardInputsMessage:
        if products is None:
            products = ["openweather_current"]

        run_id = str(uuid.uuid4())
        now = dt.datetime.utcnow()
        forecasts: List[Dict[str, Any]] = []
        bulletins: List[str] = []

        for area in self.region_areas:
            coords = REGION_AREA_COORDS.get(area)
            if not coords:
                continue
            raw = fetch_weather_raw(coords["lat"], coords["lon"])
            if not raw:
                forecasts.append({
                    "area": area,
                    "hazard": "Unknown",
                    "timeframe": time_window,
                    "products": products,
                })
                bulletins.append(f"{area}: Unable to retrieve external weather data.")
                continue

            main = raw.get("main", {})
            temp = float(main.get("temp", 0.0))
            feels_like = float(main.get("feels_like", temp))
            rain_mm = 0.0
            if "rain" in raw:
                rain = raw["rain"]
                rain_mm = float(rain.get("1h", rain.get("3h", 0.0)))
            qpf_inches_24h = rain_mm / 25.4 if rain_mm else 0.0

            hazards_here = []
            if qpf_inches_24h >= 0.1:
                hazards_here.append({
                    "area": area,
                    "hazard": "Heavy Rain & Flooding",
                    "qpf_inches_24h": qpf_inches_24h,
                    "timeframe": "Next 6–24 hours",
                    "products": products,
                })

            if feels_like >= 35.0:
                hi = feels_like * 1.1
                hazards_here.append({
                    "area": area,
                    "hazard": "Excessive Heat",
                    "heat_index": hi,
                    "timeframe": "Afternoon",
                    "products": products,
                })

            if not hazards_here:
                hazards_here.append({
                    "area": area,
                    "hazard": "No Significant Hazard",
                    "timeframe": time_window,
                    "products": products,
                })

            if demo_force_hazard and area == "Coastal Bend":
                hazards_here = [{
                    "area": area,
                    "hazard": "Heavy Rain & Flooding",
                    "qpf_inches_24h": 3.2,
                    "timeframe": "Next 24 hours",
                    "products": products + ["demo_override"],
                }]

            forecasts.extend(hazards_here)

            summary_text = (
                f"Current temp {temp:.1f}°C (feels like {feels_like:.1f}°C), "
                f"rain last hour {rain_mm:.1f} mm. "
            )
            if hazards_here and hazards_here[0]["hazard"] != "No Significant Hazard":
                summary_text += "Potential operational impacts due to highlighted hazards."
            else:
                summary_text += "No significant hazards detected at this time."
            bulletins.append(f"{area}: {summary_text}")

        return HazardInputsMessage(
            run_id=run_id,
            as_of=now,
            areas=self.region_areas,
            forecasts=forecasts,
            bulletins=bulletins,
        )

class HazardInterpretationAgent:
    def __init__(self, use_gemini: bool = False):
        self.use_gemini = use_gemini

    def _rule_based_seed(self, fc: Dict[str, Any]) -> Dict[str, str]:
        hazard = fc.get("hazard", "Unknown")
        if hazard == "Heavy Rain & Flooding":
            qpf = float(fc.get("qpf_inches_24h", 0.0))
            if qpf >= 3.0:
                return {"likelihood": "High", "impact": "Dangerous", "rationale": f"QPF={qpf:.2f} in/24h."}
            elif qpf >= 1.5:
                return {"likelihood": "Medium", "impact": "Disruptive", "rationale": f"QPF={qpf:.2f} in/24h."}
            elif qpf >= 0.5:
                return {"likelihood": "Low", "impact": "Nuisance", "rationale": f"QPF={qpf:.2f} in/24h."}
            else:
                return {"likelihood": "Low", "impact": "Nuisance", "rationale": "Minimal QPF."}
        if hazard == "Excessive Heat":
            hi = float(fc.get("heat_index", 0.0))
            if hi >= 108:
                return {"likelihood": "High", "impact": "Dangerous", "rationale": f"Heat index={hi:.1f}."}
            elif hi >= 103:
                return {"likelihood": "Medium", "impact": "Disruptive", "rationale": f"Heat index={hi:.1f}."}
            elif hi >= 95:
                return {"likelihood": "Low", "impact": "Nuisance", "rationale": f"Heat index={hi:.1f}."}
            else:
                return {"likelihood": "Low", "impact": "Nuisance", "rationale": "Heat not critical."}
        if hazard == "No Significant Hazard":
            return {"likelihood": "Low", "impact": "Nuisance", "rationale": "No significant hazard indicated."}
        return {"likelihood": "Low", "impact": "Nuisance", "rationale": "Default / unknown hazard."}

    def assess(self, inputs_msg: HazardInputsMessage) -> HazardRisksMessage:
        risks: List[HazardRisk] = []
        for fc in inputs_msg.forecasts:
            area = fc.get("area", "Unknown")
            hazard = fc.get("hazard", "Unknown")
            timeframe = fc.get("timeframe", "Next 24 hours")
            seed = self._rule_based_seed(fc)
            risks.append(
                HazardRisk(
                    area=area,
                    hazard=hazard,
                    timeframe=timeframe,
                    likelihood=seed["likelihood"],
                    impact=seed["impact"],
                    rationale=seed["rationale"],
                    supporting_evidence=fc.get("products", []),
                )
            )
        return HazardRisksMessage(
            run_id=inputs_msg.run_id,
            as_of=inputs_msg.as_of,
            risks=risks,
        )

class TriggerEvaluationAgent:
    def __init__(self, triggers: List[Dict[str, Any]]):
        self.triggers = triggers
        self.posture_rank = ["Normal", "Enhanced Monitoring", "Response Consideration"]

    def _meets_trigger(self, risk: HazardRisk, trig: Dict[str, Any]) -> bool:
        if risk.hazard != trig["hazard"]:
            return False
        if LIKELIHOOD_ORDER.index(risk.likelihood) < LIKELIHOOD_ORDER.index(trig["min_likelihood"]):
            return False
        if IMPACT_ORDER.index(risk.impact) < IMPACT_ORDER.index(trig["min_impact"]):
            return False
        return True

    def evaluate(self, risk_msg: HazardRisksMessage) -> TriggerResultsMessage:
        areas_dict: Dict[str, AreaTriggerSummary] = {}
        for r in risk_msg.risks:
            if r.area not in areas_dict:
                areas_dict[r.area] = AreaTriggerSummary(name=r.area, posture="Normal", fired_triggers=[])
        for r in risk_msg.risks:
            for trig in self.triggers:
                if self._meets_trigger(r, trig):
                    summary = areas_dict[r.area]
                    new_posture = trig["recommended_posture"]
                    if self.posture_rank.index(new_posture) > self.posture_rank.index(summary.posture):
                        summary.posture = new_posture
                    summary.fired_triggers.append(
                        {
                            "trigger_id": trig["id"],
                            "name": trig["name"],
                            "rationale": f"{r.likelihood} likelihood, {r.impact} impact; {r.rationale}",
                        }
                    )
        return TriggerResultsMessage(
            run_id=risk_msg.run_id,
            as_of=risk_msg.as_of,
            areas=list(areas_dict.values()),
        )

class BriefingAgent:
    def __init__(self, region_name: str, use_gemini: bool = True):
        self.region_name = region_name
        self.use_gemini = use_gemini

    def _fallback_brief_text(self, risks_msg: HazardRisksMessage, triggers_msg: TriggerResultsMessage) -> str:
        run_time = risks_msg.as_of
        lines: List[str] = []
        lines.append(f"Weather & Hazard Brief for {self.region_name}")
        lines.append(f"As of {run_time.isoformat()} UTC\n")
        risks = risks_msg.risks
        if not risks:
            lines.append("Overall: No significant hazards identified for the monitored period.")
        else:
            lines.append("Key Hazards:")
            for r in risks:
                lines.append(
                    f"- {r.area}: {r.hazard} "
                    f"({r.likelihood} likelihood, {r.impact} impact) – {r.timeframe}. "
                    f"{r.rationale}"
                )
        if triggers_msg.areas:
            lines.append("\nRecommended Readiness Posture:")
            for a in triggers_msg.areas:
                if not a.fired_triggers:
                    lines.append(f"- {a.name}: {a.posture} (no triggers fired).")
                else:
                    reasons = "; ".join(f"{t['name']} ({t['rationale']})" for t in a.fired_triggers)
                    lines.append(f"- {a.name}: {a.posture} due to {reasons}.")
        else:
            lines.append("\nRecommended Readiness Posture: Normal operations for all monitored areas.")
        return "\n".join(lines)

    def generate(self, risks_msg: HazardRisksMessage, triggers_msg: TriggerResultsMessage) -> BriefPacketMessage:
        posture_overview = {a.name: a.posture for a in triggers_msg.areas}
        if not self.use_gemini or not GEMINI_API_KEY:
            brief_text = self._fallback_brief_text(risks_msg, triggers_msg)
        else:
            structured = {
                "region_name": self.region_name,
                "as_of": risks_msg.as_of.isoformat(),
                "risks": [asdict(r) for r in risks_msg.risks],
                "areas": [
                    {"name": a.name, "posture": a.posture, "fired_triggers": a.fired_triggers}
                    for a in triggers_msg.areas
                ],
            }
            prompt = (
                "You are generating an internal weather & hazard brief for the American Red Cross. "
                "Write a concise brief for regional leadership, with sections:\n"
                "1) Overview\n2) Key Hazards by Area\n3) Recommended Readiness Posture\n\n"
                "Focus on timing, likelihood, impact, and operational implications. "
                "Avoid overly technical meteorological jargon. "
                "Input (JSON):\n\n"
                f"{json.dumps(structured)[:4000]}"
            )
            brief_text = call_gemini(prompt)
            if brief_text.startswith("[Gemini error") or brief_text.startswith("[Gemini raw"):
                brief_text = self._fallback_brief_text(risks_msg, triggers_msg)
        return BriefPacketMessage(
            run_id=risks_msg.run_id,
            as_of=risks_msg.as_of,
            markdown_brief=brief_text,
            text_brief=brief_text,
            posture_overview=posture_overview,
        )

class MemoryLoggingAgent:
    def __init__(self):
        self.runs: List[Dict[str, Any]] = []
        self.risks_log: List[Dict[str, Any]] = []
        self.triggers_log: List[Dict[str, Any]] = []
        self.briefs_log: List[Dict[str, Any]] = []
        self.checkpoint = CheckpointState(
            last_run_time=None,
            last_posture_by_area={},
            last_run_id=None,
            operational_period_label="Initial",
        )

    def log_cycle(self,
                  hazard_inputs: HazardInputsMessage,
                  risks_msg: HazardRisksMessage,
                  trig_msg: TriggerResultsMessage,
                  brief_msg: BriefPacketMessage):
        self.runs.append({
            "run_id": hazard_inputs.run_id,
            "as_of": hazard_inputs.as_of,
            "areas": ",".join(hazard_inputs.areas),
            "n_forecasts": len(hazard_inputs.forecasts),
            "n_bulletins": len(hazard_inputs.bulletins),
            "n_risks": len(risks_msg.risks),
            "n_trigger_areas": len(trig_msg.areas),
        })
        for r in risks_msg.risks:
            row = asdict(r)
            row["run_id"] = risks_msg.run_id
            row["as_of"] = risks_msg.as_of
            self.risks_log.append(row)
        for a in trig_msg.areas:
            for t in a.fired_triggers:
                self.triggers_log.append({
                    "run_id": trig_msg.run_id,
                    "as_of": trig_msg.as_of,
                    "area": a.name,
                    "posture": a.posture,
                    "trigger_id": t["trigger_id"],
                    "trigger_name": t["name"],
                    "rationale": t["rationale"],
                })
        self.briefs_log.append({
            "run_id": brief_msg.run_id,
            "as_of": brief_msg.as_of,
            "brief": brief_msg.text_brief,
        })
        self.checkpoint = CheckpointState(
            last_run_time=hazard_inputs.as_of,
            last_posture_by_area=brief_msg.posture_overview,
            last_run_id=hazard_inputs.run_id,
            operational_period_label="Ongoing",
        )

# -------------------------------------------------------
# GCS checkpoint helpers
# -------------------------------------------------------

def save_checkpoint_to_gcs(checkpoint: CheckpointState):
    if not storage or not GCS_BUCKET:
        logging.warning("GCS not configured; skipping checkpoint save.")
        return
    client = storage.Client(project=GCP_PROJECT_ID or None)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(CHECKPOINT_BLOB_NAME)
    payload = {
        "last_run_time": checkpoint.last_run_time.isoformat() if checkpoint.last_run_time else None,
        "last_posture_by_area": checkpoint.last_posture_by_area,
        "last_run_id": checkpoint.last_run_id,
        "operational_period_label": checkpoint.operational_period_label,
    }
    blob.upload_from_string(json.dumps(payload), content_type="application/json")
    logging.info("Checkpoint saved to GCS.")

def load_checkpoint_from_gcs() -> Optional[CheckpointState]:
    if not storage or not GCS_BUCKET:
        logging.warning("GCS not configured; skipping checkpoint load.")
        return None
    client = storage.Client(project=GCP_PROJECT_ID or None)
    bucket = client.bucket(GCS_BUCKET)
    blob = bucket.blob(CHECKPOINT_BLOB_NAME)
    if not blob.exists():
        logging.info("No checkpoint in GCS.")
        return None
    data = json.loads(blob.download_as_text())
    last_run_time = dt.datetime.fromisoformat(data["last_run_time"]) if data["last_run_time"] else None
    return CheckpointState(
        last_run_time=last_run_time,
        last_posture_by_area=data.get("last_posture_by_area", {}),
        last_run_id=data.get("last_run_id"),
        operational_period_label=data.get("operational_period_label", "Ongoing"),
    )

# -------------------------------------------------------
# Orchestrator + exported cycle
# -------------------------------------------------------

class OrchestratorScheduler:
    def __init__(self,
                 ingestion: DataIngestionAgent,
                 hazard_int: HazardInterpretationAgent,
                 trigger_eval: TriggerEvaluationAgent,
                 briefing: BriefingAgent,
                 memory: MemoryLoggingAgent):
        self.ingestion = ingestion
        self.hazard_int = hazard_int
        self.trigger_eval = trigger_eval
        self.briefing = briefing
        self.memory = memory
        self.paused = False

    def run_cycle(self,
                  time_window: str = "next_72_hours",
                  products: Optional[List[str]] = None,
                  demo_force_hazard: bool = False):
        if self.paused:
            logging.info("Orchestrator paused; skipping cycle.")
            return None
        hazard_inputs_msg = self.ingestion.ingest(
            time_window=time_window,
            products=products,
            demo_force_hazard=demo_force_hazard,
        )
        risks_msg = self.hazard_int.assess(hazard_inputs_msg)
        trig_msg = self.trigger_eval.evaluate(risks_msg)
        brief_msg = self.briefing.generate(risks_msg, trig_msg)
        self.memory.log_cycle(hazard_inputs_msg, risks_msg, trig_msg, brief_msg)
        return hazard_inputs_msg, risks_msg, trig_msg, brief_msg


# Instantiate global agents + orchestrator for deployment
ingestion_agent = DataIngestionAgent(REGION_AREAS)
hazard_agent = HazardInterpretationAgent(use_gemini=False)
trigger_agent = TriggerEvaluationAgent(TRIGGERS)
briefing_agent = BriefingAgent(REGION_NAME, use_gemini=True)
memory_agent = MemoryLoggingAgent()

orchestrator = OrchestratorScheduler(
    ingestion=ingestion_agent,
    hazard_int=hazard_agent,
    trigger_eval=trigger_agent,
    briefing=briefing_agent,
    memory=memory_agent,
)

def orchestrator_cloud_run_cycle():
    """
    Core monitoring cycle for Cloud Run / Scheduler:
    - Load checkpoint
    - Run one cycle
    - Save checkpoint
    """
    cp = load_checkpoint_from_gcs()
    if cp:
        memory_agent.checkpoint = cp
    result = orchestrator.run_cycle()
    if result is None:
        return "SKIPPED"
    save_checkpoint_to_gcs(memory_agent.checkpoint)
    return "OK"
