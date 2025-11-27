# sentinel_ui.py
"""
Simple Streamlit UI for Weather & Hazard Sentinel.

- Runs one monitoring cycle using agents.py
- Shows posture by area + brief text
- Great for screenshots / demo video
"""

import os

import streamlit as st
import pandas as pd
import requests

from agents import orchestrator, memory_agent

AREAS_METADATA = {
    "Coastal Bend": {"lat": 27.9, "lon": -97.3},
    "Houston Metro": {"lat": 29.76, "lon": -95.37},
    "Golden Triangle": {"lat": 30.09, "lon": -93.74},
}
ALL_AREAS = list(AREAS_METADATA.keys())

st.set_page_config(
    page_title="Weather & Hazard Sentinel",
    page_icon="🌩️",
    layout="centered",
)

# --- Custom styling ---
CUSTOM_CSS = """
<style>
/* Overall app width + padding */
main .block-container {
    max-width: 1100px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Background */
.stApp {
    background: radial-gradient(circle at top left, #1e293b 0, #020617 50%, #020617 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #020617;
    border-right: 1px solid rgba(148, 163, 184, 0.3);
}
[data-testid="stSidebar"] h2, 
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] label {
    color: #e5e7eb;
}

/* Hero area */
.hero {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 1.5rem 1.75rem;
    border-radius: 18px;
    background: linear-gradient(135deg, #0f172a, #1d2a3b);
    box-shadow: 0 18px 35px rgba(15, 23, 42, 0.7);
    margin-bottom: 1.5rem;
}
.hero-emoji {
    font-size: 2.8rem;
}
.hero-text h1 {
    font-size: 2.2rem;
    margin: 0;
    color: #f9fafb;
    letter-spacing: 0.03em;
}
.hero-text p {
    margin: 0.25rem 0 0;
    color: #cbd5f5;
    font-size: 0.98rem;
}

/* Section titles */
.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin: 1.5rem 0 0.5rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #e5e7eb;
}
.section-title span.icon {
    font-size: 1.3rem;
}

/* Success banner */
.success-banner {
    padding: 0.75rem 1rem;
    border-radius: 12px;
    background: rgba(22, 163, 74, 0.15);
    border: 1px solid rgba(34, 197, 94, 0.6);
    color: #bbf7d0;
    font-size: 0.92rem;
    margin-top: 0.75rem;
}

/* Data cards (tables + text boxes) */
.data-card {
    padding: 0.9rem 1rem;
    border-radius: 14px;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.35);
    margin-bottom: 0.75rem;
}

/* Dataframes */
.data-card .stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}

/* Brief text box */
.brief-box {
    white-space: pre-wrap;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
                 "Segoe UI", sans-serif;
    font-size: 0.94rem;
    line-height: 1.45;
    color: #e5e7eb;
}

/* Run button */
button[kind="primary"] {
    border-radius: 999px !important;
    padding: 0.5rem 1.25rem !important;
    font-weight: 600 !important;
}
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="hero">
      <div class="hero-emoji">⛈️</div>
      <div class="hero-text">
        <h1>Weather &amp; Hazard Sentinel</h1>
        <p>Multi-Agent Monitoring for Real-Time Readiness</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    This UI runs the same multi-agent pipeline from the notebook:
    
    1. Ingest weather data (demo mode can force a flood scenario)  
    2. Assess hazards by area  
    3. Evaluate readiness triggers  
    4. Generate a leadership brief  
    """,
)

# --- Controls ---
st.sidebar.header("Run Settings")

demo_force = st.sidebar.checkbox(
    "Demo: force Coastal Bend flood scenario",
    value=True,
    help="Overrides live weather and injects a heavy rain / flood case."
)

time_window = st.sidebar.selectbox(
    "Time window",
    ["next_24_hours", "next_72_hours"],
    index=1,
)

# Filter area focus
selected_area = st.sidebar.selectbox(
    "Focus area",
    ["All areas"] + ALL_AREAS,
    index=0,
)

st.sidebar.markdown("---")

# Live API test for OpenWeather
st.sidebar.subheader("Live API test")

if st.sidebar.button("🔌 Test OpenWeather API"):
    api_key = os.environ.get("OPENWEATHER_API_KEY")
    if not api_key:
        st.sidebar.error("Set OPENWEATHER_API_KEY env var to run a live test.")
    else:
        lat, lon = 29.76, -95.37  # Houston test
        url = (
            "https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={api_key}&units=imperial"
        )
        try:
            resp = requests.get(url, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                summary = {
                    "location": data.get("name", "Houston test"),
                    "temp_F": data.get("main", {}).get("temp"),
                    "conditions": data.get("weather", [{}])[0].get("description"),
                }
                st.sidebar.success("OpenWeather API live test OK ✅")
                st.sidebar.json(summary)
            else:
                st.sidebar.error(f"API error {resp.status_code}: {resp.text[:120]}...")
        except Exception as e:
            st.sidebar.error(f"Request failed: {e}")

st.sidebar.markdown("When ready, click **Run Monitoring Cycle** below.")

run_clicked = st.button("🚀 Run Monitoring Cycle")

if run_clicked:
    with st.spinner("Running agents (ingest → hazards → triggers → brief)..."):
        result = orchestrator.run_cycle(
            time_window=time_window,
            products=["openweather_current"],
            demo_force_hazard=demo_force,
        )

    if result is None:
        st.error("Orchestrator is paused or returned no result.")
    else:
        hazard_inputs, risks_msg, trig_msg, brief_msg = result

        # Success banner
        st.markdown(
            '<div class="success-banner">Monitoring cycle completed.</div>',
            unsafe_allow_html=True,
        )

        # -----------------------------
        # Regional snapshot (map + chips)
        # -----------------------------
        st.markdown(
            '<div class="section-title"><span class="icon">🗺️</span>'
            'Regional Snapshot</div>',
            unsafe_allow_html=True,
        )

        # Build DF for map
        map_rows = []
        for area_name in hazard_inputs.areas:
            meta = AREAS_METADATA.get(area_name)
            if not meta:
                continue
            area_posture = next(
                (a.posture for a in trig_msg.areas if a.name == area_name),
                "Unknown",
            )
            map_rows.append(
                {
                    "area": area_name,
                    "lat": meta["lat"],
                    "lon": meta["lon"],
                    "posture": area_posture,
                }
            )

        if map_rows:
            df_map = pd.DataFrame(map_rows)
            c1, c2 = st.columns([2, 3])

            with c1:
                st.map(df_map[["lat", "lon"]], zoom=6)

            with c2:
                st.markdown("**Posture by region**")
                for row in map_rows:
                    icon = "🟢"
                    if row["posture"].startswith("Response"):
                        icon = "🟠"
                    elif row["posture"].startswith("Alert"):
                        icon = "🔴"
                    st.markdown(f"{icon} **{row['area']}** — {row['posture']}")

        # -------------------------
        # Readiness posture by area
        # -------------------------
        st.markdown(
            '<div class="section-title"><span class="icon">🛡️</span>'
            'Readiness Posture by Area</div>',
            unsafe_allow_html=True,
        )

        posture_rows = [
            {
                "Area": a.name,
                "Posture": a.posture,
                "Triggers Fired": len(a.fired_triggers),
            }
            for a in trig_msg.areas
        ]
        df_posture = pd.DataFrame(posture_rows)

        if selected_area != "All areas":
            df_posture = df_posture[df_posture["Area"] == selected_area]

        with st.container():
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.dataframe(df_posture, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # -------------
        # Key Hazards
        # -------------
        st.markdown(
            '<div class="section-title"><span class="icon">⚠️</span>'
            "Key Hazards</div>",
            unsafe_allow_html=True,
        )

        df_risks = pd.DataFrame(
            [
                {
                    "Area": r.area,
                    "Hazard": r.hazard,
                    "Timeframe": r.timeframe,
                    "Likelihood": r.likelihood,
                    "Impact": r.impact,
                    "Rationale": r.rationale,
                }
                for r in risks_msg.risks
            ]
        )

        if selected_area != "All areas":
            df_risks = df_risks[df_risks["Area"] == selected_area]

        with st.container():
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.dataframe(df_risks, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ------------------------
        # Leadership brief (text)
        # ------------------------
        st.markdown(
            '<div class="section-title"><span class="icon">📝</span>'
            "Leadership Brief (Text)</div>",
            unsafe_allow_html=True,
        )

        with st.container():
            st.markdown('<div class="data-card brief-box">', unsafe_allow_html=True)
            st.text_area(
                "Brief text",
                value=brief_msg.text_brief,
                height=260,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------
        # Run metadata
        # ---------------
        st.markdown(
            '<div class="section-title"><span class="icon">📊</span>'
            "Run Metadata</div>",
            unsafe_allow_html=True,
        )

        meta = {
            "run_id": hazard_inputs.run_id,
            "as_of_utc": hazard_inputs.as_of.isoformat(),
            "areas": hazard_inputs.areas,
        }

        with st.container():
            st.markdown('<div class="data-card">', unsafe_allow_html=True)
            st.json(meta)
            st.markdown("</div>", unsafe_allow_html=True)

        # ---------------
        # Run history
        # ---------------
        if memory_agent.runs:
            st.markdown(
                '<div class="section-title"><span class="icon">📜</span>'
                "Run History (this session)</div>",
                unsafe_allow_html=True,
            )

            df_runs = pd.DataFrame(memory_agent.runs)

            with st.container():
                st.markdown('<div class="data-card">', unsafe_allow_html=True)
                st.dataframe(df_runs, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

else:
    st.info("Click **Run Monitoring Cycle** to generate a new brief.")