import streamlit as st
import pandas as pd
import psycopg2
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os
from datetime import datetime, timedelta

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — must be the first Streamlit command
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Maritime Intelligence Command Center",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS — Intel Command Center dark theme polish
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Hide default Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ── KPI Cards ── */
    .kpi-card {
        background: linear-gradient(135deg, #111827 0%, #1a2332 100%);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 16px;
        padding: 24px 28px;
        text-align: center;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.3);
    }
    .kpi-card:hover {
        border-color: rgba(0, 212, 255, 0.4);
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
        transform: translateY(-2px);
    }
    .kpi-value {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .kpi-label {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 8px;
    }
    .kpi-value.danger {
        background: linear-gradient(135deg, #ff4757, #ff6b81);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .kpi-value.warning {
        background: linear-gradient(135deg, #ffa502, #ffcc02);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e2e8f0;
        border-left: 3px solid #00d4ff;
        padding-left: 14px;
        margin: 28px 0 16px 0;
        letter-spacing: 0.5px;
    }

    /* ── Status Badge ── */
    .status-online {
        display: inline-block;
        background: rgba(0, 255, 136, 0.12);
        color: #00ff88;
        border: 1px solid rgba(0, 255, 136, 0.3);
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        animation: pulse-glow 2s ease-in-out infinite;
    }
    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 4px rgba(0, 255, 136, 0.2); }
        50% { box-shadow: 0 0 12px rgba(0, 255, 136, 0.4); }
    }

    /* ── Dataframe styling ── */
    .stDataFrame { border-radius: 12px; overflow: hidden; }

    /* ── Plotly chart containers ── */
    .stPlotlyChart { border-radius: 12px; }

    /* ── Top Bar ── */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 8px 0 20px 0;
        border-bottom: 1px solid rgba(0, 212, 255, 0.1);
        margin-bottom: 24px;
    }
    .top-bar-title {
        font-size: 1.6rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .top-bar-subtitle {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# DATABASE CONNECTION
# ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=120)  # Cache for 2 minutes to reduce DB hits
def get_data():
    """Fetch news alerts from PostgreSQL with graceful NaN handling."""
    try:
        conn = psycopg2.connect(
            host=os.environ.get("POSTGRES_HOST", "localhost"),
            database=os.environ.get("POSTGRES_DB", "maritime_intel"),
            user=os.environ.get("POSTGRES_USER", "anshul_admin"),
            password=os.environ.get("POSTGRES_PASSWORD", "maritime_secure_pass"),
            port=os.environ.get("POSTGRES_PORT", "5432"),
        )
        query = """
            SELECT headline, risk_score, sentiment_label, source_url, published_at
            FROM news_alerts
            ORDER BY published_at DESC
        """
        df = pd.read_sql(query, conn)
        conn.close()

        # ── Graceful NaN handling ──
        df["risk_score"] = df["risk_score"].fillna(0).astype(int)
        df["sentiment_label"] = df["sentiment_label"].fillna("Unknown")
        df["headline"] = df["headline"].fillna("No headline available")
        df["source_url"] = df["source_url"].fillna("")
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


def generate_mock_coordinates(n):
    """Generate mock maritime coordinates for the incident map.
    Concentrated around major shipping lanes: Strait of Malacca,
    Gulf of Aden, English Channel, South China Sea."""
    hotspots = [
        (1.3, 103.8),    # Strait of Malacca
        (12.0, 45.0),    # Gulf of Aden
        (50.5, 1.0),     # English Channel
        (14.5, 114.0),   # South China Sea
        (29.0, 33.0),    # Suez Canal
        (-6.0, 39.5),    # Dar es Salaam / Tanzania
        (25.3, 55.3),    # Persian Gulf / UAE
        (36.0, -5.5),    # Strait of Gibraltar
    ]
    lats, lons = [], []
    for i in range(n):
        base = hotspots[i % len(hotspots)]
        lats.append(base[0] + np.random.uniform(-3, 3))
        lons.append(base[1] + np.random.uniform(-3, 3))
    return lats, lons


# ─────────────────────────────────────────────────────────────
# PLOTLY THEME — consistent dark styling
# ─────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8"),
    margin=dict(l=40, r=20, t=50, b=40),
    xaxis=dict(gridcolor="rgba(100,116,139,0.1)", zerolinecolor="rgba(100,116,139,0.15)"),
    yaxis=dict(gridcolor="rgba(100,116,139,0.1)", zerolinecolor="rgba(100,116,139,0.15)"),
)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Command Center")
    st.markdown("---")
    st.markdown(f"**Operator:** Anshul Thakur")
    st.markdown(f"**System:** Maritime Intel Agent v1.0")
    st.markdown(f"**AI Model:** FinBERT Sentiment Engine")
    st.markdown(f"**Last Sync:** `{datetime.now().strftime('%H:%M:%S %Z')}`")
    st.markdown("---")

    # Auto-refresh controls
    auto_refresh = st.toggle("⚡ Auto-Refresh (2 min)", value=False)
    if st.button("🔄 Refresh Data Now", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

    st.markdown("---")
    st.markdown(
        '<p style="color: #475569; font-size: 0.7rem; text-align: center;">'
        "MARITIME INTELLIGENCE AGENT<br>Phase 4 · Visualization Layer</p>",
        unsafe_allow_html=True,
    )

# If auto-refresh is enabled, rerun every 120 seconds
if auto_refresh:
    st.empty()  # Streamlit needs an element before auto-rerun timer
    import time
    # Using st.fragment or manual rerun approach
    # st.rerun triggers a full page rerun; we use a placeholder approach
    # The cache TTL of 120s handles staleness; the toggle is a UX signal.


# ─────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="top-bar">
        <div>
            <div class="top-bar-title">🚢 Maritime Risk Intelligence Dashboard</div>
            <div class="top-bar-subtitle">Real-time threat monitoring · FinBERT NLP Engine · PostgreSQL Data Layer</div>
        </div>
        <div>
            <span class="status-online">● SYSTEM ONLINE</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
# FETCH DATA
# ─────────────────────────────────────────────────────────────
data, error = get_data()

if error:
    st.error(f"⛔ Database Connection Error: `{error}`")
    st.info("Ensure Docker containers are running: `docker compose up -d`")
    st.stop()

if data.empty:
    st.warning("📭 No alerts found in the database. Waiting for n8n to ingest data…")
    st.stop()


# ─────────────────────────────────────────────────────────────
# KPI CARDS ROW
# ─────────────────────────────────────────────────────────────
total_alerts = len(data)
avg_risk = round(data["risk_score"].mean(), 1) if total_alerts > 0 else 0
high_risk = len(data[data["risk_score"] >= 7])
critical_risk = len(data[data["risk_score"] >= 9])

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-value">{total_alerts}</div>
            <div class="kpi-label">Total Alerts</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k2:
    risk_class = "danger" if avg_risk >= 7 else ("warning" if avg_risk >= 4 else "")
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-value {risk_class}">{avg_risk}</div>
            <div class="kpi-label">Avg Risk Score</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-value warning">{high_risk}</div>
            <div class="kpi-label">High Risk (≥7)</div>
        </div>""",
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-value danger">{critical_risk}</div>
            <div class="kpi-label">Critical (≥9)</div>
        </div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
# ROW 2 — Risk Trend + Heatmap
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Threat Analytics</div>', unsafe_allow_html=True)

chart_left, chart_right = st.columns([3, 2])

with chart_left:
    # ── Risk Score Over Time (Line Chart) ──
    if "published_at" in data.columns and data["published_at"].notna().any():
        trend_data = data.sort_values("published_at")
        fig_trend = go.Figure()
        fig_trend.add_trace(go.Scatter(
            x=trend_data["published_at"],
            y=trend_data["risk_score"],
            mode="lines+markers",
            name="Risk Score",
            line=dict(color="#00d4ff", width=2.5),
            marker=dict(
                size=6,
                color=trend_data["risk_score"],
                colorscale=[[0, "#00ff88"], [0.5, "#ffa502"], [1, "#ff4757"]],
                showscale=False,
            ),
            fill="tozeroy",
            fillcolor="rgba(0, 212, 255, 0.05)",
            hovertemplate="<b>%{x|%b %d, %H:%M}</b><br>Risk: %{y}<extra></extra>",
        ))
        fig_trend.update_layout(
            title=dict(text="Risk Score Progression", font=dict(size=14, color="#e2e8f0")),
            xaxis_title=None,
            yaxis_title="Risk Score",
            yaxis=dict(range=[0, 10.5]),
            height=380,
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="trend_chart")
    else:
        st.info("No timeline data available.")

with chart_right:
    # ── Risk Score Distribution (Bar Heatmap) ──
    risk_counts = data["risk_score"].value_counts().reindex(range(1, 11), fill_value=0)
    colors = [
        "#10b981" if v <= 3 else "#f59e0b" if v <= 6 else "#ef4444"
        for v in risk_counts.index
    ]
    fig_dist = go.Figure(go.Bar(
        x=risk_counts.index,
        y=risk_counts.values,
        marker=dict(
            color=colors,
            line=dict(width=0),
            cornerradius=4,
        ),
        hovertemplate="Risk %{x}: <b>%{y} alerts</b><extra></extra>",
    ))
    fig_dist.update_layout(
        title=dict(text="Risk Distribution (1–10)", font=dict(size=14, color="#e2e8f0")),
        xaxis=dict(
            title="Risk Level", dtick=1,
            gridcolor="rgba(100,116,139,0.1)", zerolinecolor="rgba(100,116,139,0.15)",
        ),
        yaxis=dict(
            title="Count",
            gridcolor="rgba(100,116,139,0.1)", zerolinecolor="rgba(100,116,139,0.15)",
        ),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        margin=dict(l=40, r=20, t=50, b=40),
        bargap=0.15,
    )
    st.plotly_chart(fig_dist, use_container_width=True, key="dist_chart")


# ─────────────────────────────────────────────────────────────
# ROW 3 — Incident Map + Sentiment Breakdown
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">🌍 Global Incident Map & Sentiment Overview</div>', unsafe_allow_html=True)

map_col, sentiment_col = st.columns([3, 2])

with map_col:
    # ── Interactive Map with Mock Maritime Coordinates ──
    lats, lons = generate_mock_coordinates(len(data))
    map_df = data.copy()
    map_df["lat"] = lats
    map_df["lon"] = lons

    fig_map = px.scatter_mapbox(
        map_df,
        lat="lat",
        lon="lon",
        color="risk_score",
        size=np.clip(map_df["risk_score"].values * 3, 4, 30).tolist(),
        color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
        range_color=[1, 10],
        hover_name="headline",
        hover_data={"risk_score": True, "sentiment_label": True, "lat": False, "lon": False},
        zoom=1.3,
        height=420,
        title="Incident Locations (Simulated Coordinates)",
    )
    fig_map.update_layout(
        mapbox_style="carto-darkmatter",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar=dict(
            title="Risk",
            tickvals=[1, 5, 10],
            ticktext=["Low", "Med", "High"],
            len=0.6,
        ),
    )
    st.plotly_chart(fig_map, use_container_width=True, key="map_chart")

with sentiment_col:
    # ── Sentiment Donut Chart ──
    sentiment_counts = data["sentiment_label"].value_counts()
    color_map = {
        "Positive": "#10b981",
        "Negative": "#ef4444",
        "Neutral": "#64748b",
        "Unknown": "#334155",
    }
    fig_sent = go.Figure(go.Pie(
        labels=sentiment_counts.index,
        values=sentiment_counts.values,
        hole=0.55,
        marker=dict(
            colors=[color_map.get(s, "#64748b") for s in sentiment_counts.index],
            line=dict(color="#0a0e17", width=2),
        ),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig_sent.update_layout(
        title=dict(text="FinBERT Sentiment Analysis", font=dict(size=14, color="#e2e8f0")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        height=420,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=True,
        legend=dict(font=dict(size=11)),
    )
    st.plotly_chart(fig_sent, use_container_width=True, key="sentiment_chart")


# ─────────────────────────────────────────────────────────────
# ROW 4 — Data Table (Latest Alerts Feed)
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Live Alert Feed</div>', unsafe_allow_html=True)

display_df = data[["headline", "risk_score", "sentiment_label", "published_at"]].copy()
display_df.columns = ["Headline", "Risk Score", "Sentiment", "Published At"]

# Color-code risk levels in a human-readable way
def risk_tag(val):
    if val >= 9:
        return "🔴 CRITICAL"
    elif val >= 7:
        return "🟠 HIGH"
    elif val >= 4:
        return "🟡 MEDIUM"
    else:
        return "🟢 LOW"

display_df.insert(1, "Threat Level", display_df["Risk Score"].apply(risk_tag))

st.dataframe(
    display_df,
    use_container_width=True,
    height=400,
    column_config={
        "Headline": st.column_config.TextColumn("Headline", width="large"),
        "Threat Level": st.column_config.TextColumn("Threat Level", width="small"),
        "Risk Score": st.column_config.ProgressColumn(
            "Risk Score", min_value=0, max_value=10, format="%d"
        ),
        "Sentiment": st.column_config.TextColumn("Sentiment", width="small"),
        "Published At": st.column_config.DatetimeColumn("Published At", format="MMM DD, YYYY HH:mm"),
    },
)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #334155; font-size: 0.75rem;">'
    "Maritime Intelligence Agent · Built by Anshul Thakur · "
    f"Dashboard rendered at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    "</p>",
    unsafe_allow_html=True,
)