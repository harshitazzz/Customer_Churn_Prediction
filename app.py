import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import sys
import time
from dotenv import load_dotenv

load_dotenv()

# Add project root to path for src imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="ChurnSense — AI Retention Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# PREMIUM CSS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Target text elements specifically to avoid breaking Streamlit's internal material icons */
body, [data-testid="stAppViewBlockContainer"], p, li, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stButton > button {
    font-family: 'Inter', sans-serif !important;
}

/* ── Global ── */
.stApp {
    background: linear-gradient(160deg, #0a0e1a 0%, #0f172a 40%, #111827 100%) !important;
}
[data-testid="stHeader"],
.stDeployButton {display: none !important;}
[data-testid="stAppViewBlockContainer"] {padding-top: 1.5rem !important;}

h1 {
    font-weight: 800 !important;
    background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 50%, #F472B6 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    letter-spacing: -1.5px !important;
    font-size: 2.4rem !important;
    margin-bottom: 0.1rem !important;
}
p, .stMarkdown {
    color: #CBD5E1 !important;
    font-size: 1rem !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1a1f3a 100%) !important;
    border-right: 1px solid rgba(99, 102, 241, 0.2) !important;
}
section[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: none !important;
}

/* ── Cards ── */
.card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0, 0, 0, 0.2);
    transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
}
.card:hover {
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.1);
    transform: translateY(-3px);
}

/* ── Stat Cards ── */
.stat-card {
    background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 16px;
    padding: 28px 20px;
    text-align: center;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    transition: all 0.3s ease;
}
.stat-card:hover {
    transform: scale(1.03);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.12);
}
.stat-val {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA 0%, #818CF8 50%, #A78BFA 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
}
.stat-label {
    font-size: 0.72rem;
    font-weight: 700;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 10px;
}

/* ── Result Cards ── */
.result-stay {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.8) 0%, rgba(6, 95, 70, 0.8) 100%);
    border: 1px solid rgba(16, 185, 129, 0.5);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.12);
}
.result-churn {
    background: linear-gradient(135deg, rgba(127, 29, 29, 0.8) 0%, rgba(153, 27, 27, 0.8) 100%);
    border: 1px solid rgba(239, 68, 68, 0.5);
    border-radius: 20px;
    padding: 36px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(239, 68, 68, 0.12);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 50%, #A855F7 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.35) !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(139, 92, 246, 0.5) !important;
}

/* ── Inputs ── */
[data-testid="stNumberInput"] > div > div > input,
[data-testid="stSelectbox"] > div > div > div {
    background-color: rgba(15, 23, 42, 0.8) !important;
    border-radius: 10px !important;
    border: 1px solid rgba(99, 102, 241, 0.2) !important;
    color: #FFFFFF !important;
}

/* ── Section Titles ── */
.sec-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #E2E8F0;
    margin: 28px 0 14px 0;
    padding-bottom: 10px;
    border-bottom: 2px solid rgba(99, 102, 241, 0.25);
}

/* ── Driver Cards ── */
.driver-card {
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    transition: all 0.3s ease;
    backdrop-filter: blur(8px);
}
.driver-card:hover { transform: translateY(-2px); }
.driver-high { background: rgba(127, 29, 29, 0.7); border-left: 4px solid #F87171; }
.driver-med  { background: rgba(120, 53, 15, 0.7); border-left: 4px solid #FBBF24; }
.driver-low  { background: rgba(6, 78, 59, 0.7); border-left: 4px solid #34D399; }

/* ── Pipeline Steps ── */
.pipeline-step {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
}
.pipeline-step:hover {
    border-color: rgba(99, 102, 241, 0.4);
    transform: translateY(-2px);
}
.step-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px; height: 36px;
    border-radius: 50%;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    color: #fff;
    font-weight: 800;
    font-size: 0.9rem;
    margin-bottom: 10px;
}

/* ── Agent Report ── */
.agent-report {
    background: rgba(30, 41, 59, 0.6);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 16px;
    padding: 28px;
    margin-top: 20px;
    box-shadow: 0 8px 32px rgba(139, 92, 246, 0.08);
}
.agent-step {
    padding: 10px 16px;
    border-radius: 10px;
    background: rgba(15, 23, 42, 0.6);
    border-left: 3px solid #8B5CF6;
    margin-bottom: 8px;
    font-size: 0.9rem;
    color: #CBD5E1;
}
.agent-step-done {
    border-left-color: #10B981;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(30, 41, 59, 0.5);
    border-radius: 12px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: 600;
    font-size: 0.9rem;
    color: #94A3B8 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366F1, #8B5CF6) !important;
    color: #FFFFFF !important;
}

/* ── Logo area ── */
.sidebar-brand {
    text-align: center;
    padding: 8px 0 16px 0;
}
.sidebar-brand-title {
    font-size: 1.3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #60A5FA, #A78BFA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.sidebar-brand-sub {
    font-size: 0.7rem;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 2px;
}

</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD ML ARTIFACTS
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_ml_artifacts():
    base = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base, "models")
    try:
        return (
            joblib.load(os.path.join(models_dir, "churn_log_model.joblib")),
            joblib.load(os.path.join(models_dir, "churn_dt_model.joblib")),
            joblib.load(os.path.join(models_dir, "scaler.joblib")),
            joblib.load(os.path.join(models_dir, "feature_columns.joblib")),
            joblib.load(os.path.join(models_dir, "model_metrics.joblib")),
        )
    except FileNotFoundError:
        return None, None, None, None, None

log_model, dt_model, scaler, feature_cols, metrics = load_ml_artifacts()


# ═══════════════════════════════════════════════════════════════════════════════
# CHART HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
def gauge_chart(prob, is_churn):
    color = "#EF4444" if is_churn else "#10B981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 42, "color": color, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 10, "color": "#64748B"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "rgba(30,41,59,0.5)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "rgba(6,78,59,0.4)"},
                {"range": [35, 65], "color": "rgba(120,53,15,0.4)"},
                {"range": [65, 100], "color": "rgba(127,29,29,0.4)"},
            ],
        },
        title={"text": "Churn Probability", "font": {"size": 13, "color": "#64748B"}},
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def confusion_heatmap(cm, title):
    labels = ["Stay", "Churn"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#1E293B"], [0.5, "#6366F1"], [1, "#A78BFA"]],
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}", textfont={"size": 16, "color": "#fff"},
        showscale=False,
    ))
    fig.update_layout(title=dict(text=title, font=dict(color="#E2E8F0")),
                      height=300,
                      margin=dict(l=50, r=20, t=50, b=50),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Predicted", yaxis_title="Actual",
                      font=dict(color="#94A3B8"))
    fig.update_yaxes(autorange="reversed")
    return fig


def importance_chart(feat_imp):
    items = sorted(feat_imp.items(), key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items],
        orientation="h",
        marker=dict(
            color=[v for _, v in items],
            colorscale=[[0, "#6366F1"], [1, "#A78BFA"]],
        ),
    ))
    fig.update_layout(title=dict(text="Top Feature Importance", font=dict(color="#E2E8F0")),
                      height=380,
                      margin=dict(l=180, r=20, t=50, b=30),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Importance",
                      font=dict(color="#94A3B8"))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">🧠 ChurnSense</div>
        <div class="sidebar-brand-sub">AI Retention Platform</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    if log_model is None:
        st.error("Model files not found. Run `python3 src/train_model.py` first.")
        model_choice = "Logistic Regression"
        tenure = 12
        monthly = 50.0
        total = 600.0
        support = 1
        avg_spend = 0.0
    else:
        st.markdown('<p style="font-size:0.85rem; font-weight:700; color:#A78BFA; margin-bottom:10px; letter-spacing:1px; text-transform:uppercase;">⚙️ Configuration</p>', unsafe_allow_html=True)
        model_choice = st.selectbox("ML Model", ["Logistic Regression", "Decision Tree"])

        st.markdown('<p style="font-size:0.8rem; font-weight:600; color:#64748B; margin-top:20px; margin-bottom:10px; text-transform:uppercase; letter-spacing:1px;">Customer Profile</p>', unsafe_allow_html=True)
        tenure = st.number_input("Tenure (months)", 0, 72, 12, help="How many months has the customer been with the company")
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, format="%.2f")
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0, format="%.2f")
        support = st.number_input("Support Calls", 0, 10, 1, help="Number of times customer contacted support")

        avg_spend = total / (tenure + 1)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Avg Monthly Spend", f"${avg_spend:.2f}", help="Auto-calculated: TotalCharges / (Tenure + 1)")

        st.divider()
        st.markdown('<p style="font-size:0.7rem; color:#475569; text-align:center;">Pipeline: Feature Eng → Scaling → Inference → AI Agent</p>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# GLOBAL PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.title("ChurnSense")
st.caption("Agentic AI-powered customer churn prediction & retention strategy platform.")

global_pred, global_proba, global_is_churn = 0, 0.0, False
drivers_list = []
drivers_text = []

if log_model is not None:
    inp = {col: 0 for col in feature_cols}
    inp["tenure"] = tenure
    inp["MonthlyCharges"] = monthly
    inp["TotalCharges"] = total
    inp["SeniorCitizen"] = min(support, 1)
    inp["AvgMonthlySpend"] = avg_spend

    df = pd.DataFrame([inp])[feature_cols]
    scaled = scaler.transform(df)

    # Inference
    if log_model is not None and dt_model is not None:
        active_model = log_model if model_choice == "Logistic Regression" else dt_model
        global_pred = active_model.predict(scaled)[0]
    global_proba = active_model.predict_proba(scaled)[0][1]
    global_is_churn = global_pred == 1

    # Build drivers
    if tenure < 12:
        drivers_list.append(("high", "📅 Low Tenure", f"{tenure} months — new customers are statistically more likely to churn quickly."))
        drivers_text.append("Low Tenure (< 12 months)")
    elif tenure < 24:
        drivers_list.append(("med", "📅 Moderate Tenure", f"{tenure} months — building loyalty but still in re-engagement window."))
        drivers_text.append("Moderate Tenure (12-24 months)")
    else:
        drivers_list.append(("low", "📅 Strong Tenure", f"{tenure} months — long-term established customer."))
        drivers_text.append("Strong Tenure (24+ months)")

    if monthly > 80:
        drivers_list.append(("high", "💳 High Cost", f"${monthly:.0f}/mo — upper tier pricing creates cost-sensitivity pressure."))
        drivers_text.append("High Monthly Charges (> $80)")
    elif monthly > 50:
        drivers_list.append(("med", "💳 Moderate Cost", f"${monthly:.0f}/mo — mid-tier; competitive offers might pull."))
        drivers_text.append("Moderate Monthly Charges ($50-80)")
    else:
        drivers_list.append(("low", "💳 Low Cost", f"${monthly:.0f}/mo — highly affordable tier, minimal cost pressure."))
        drivers_text.append("Low Monthly Charges (< $50)")

    if support >= 3:
        drivers_list.append(("high", "📞 Frequent Support", f"{support} calls — signifies deep friction or unresolved issues."))
        drivers_text.append("Frequent Support Calls (3+)")
    elif support >= 1:
        drivers_list.append(("med", "📞 Some Support", f"{support} call(s) — minor friction detected."))
        drivers_text.append("Some Support Calls")
    else:
        drivers_list.append(("low", "📞 No Support", "No recent support interactions — no friction signals."))
        drivers_text.append("No Support Issues")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "🔮 Predictions", "🤖 Agentic AI", "📈 Model Metrics"])


# ─── Tab 1: Dashboard ────────────────────────────────────────────────────────
with tab1:
    st.markdown("""
    <div class="card">
    <b style="color:#A78BFA;">How it works:</b> Configure customer details in the <b>sidebar</b> →
    the pipeline instantly performs feature engineering & scaling →
    view predictions in <b>Predictions</b> tab →
    generate AI-powered retention strategies in <b>Agentic AI</b> tab.
    </div>
    """, unsafe_allow_html=True)

    if metrics:
        m = metrics["logistic_regression"]
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, val) in zip([c1, c2, c3, c4], [
            ("Accuracy", m["accuracy"]),
            ("Precision", m["precision"]),
            ("Recall", m["recall"]),
            ("F1 Score", m["f1"]),
        ]):
            with col:
                st.markdown(f"""
                <div class="stat-card">
                    <div class="stat-val">{val*100:.1f}%</div>
                    <div class="stat-label">{label}</div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown('<p class="sec-title">Pipeline Architecture</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    steps = [
        ("1", "Input", "Configure customer metrics via sidebar controls"),
        ("2", "ML Pipeline", "Feature engineering, scaling & model inference"),
        ("3", "Risk Analysis", "Churn probability & key business driver extraction"),
        ("4", "AI Agent", "LangGraph + RAG generates tailored retention strategy"),
    ]
    for col, (num, title, desc) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(f"""
            <div class="pipeline-step">
                <div class="step-num">{num}</div>
                <div style="font-weight:700; color:#E2E8F0; font-size:0.95rem; margin-bottom:6px;">{title}</div>
                <div style="font-size:0.78rem; color:#64748B; line-height:1.4;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ─── Tab 2: Predictions ──────────────────────────────────────────────────────
with tab2:
    if log_model is not None:
        st.markdown('<p class="sec-title" style="margin-top:0;">Prediction Overview</p>', unsafe_allow_html=True)
        st.caption(f"Live predictions using **{model_choice}** based on sidebar inputs.")

        r1, r2 = st.columns([1, 1], gap="large")
        with r1:
            if global_is_churn:
                st.markdown("""
                <div class="result-churn">
                    <div style="font-size:3rem; margin-bottom:12px">⚠️</div>
                    <div style="font-size:1.5rem; font-weight:800; color:#FCA5A5">High Churn Risk</div>
                    <div style="font-size:0.9rem; color:#F87171; margin-top:10px; line-height:1.5">
                        This customer profile flags high for attrition. Immediate proactive retention measures recommended.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-stay">
                    <div style="font-size:3rem; margin-bottom:12px">✅</div>
                    <div style="font-size:1.5rem; font-weight:800; color:#6EE7B7">Likely to Stay</div>
                    <div style="font-size:0.9rem; color:#34D399; margin-top:10px; line-height:1.5">
                        Customer engagement metrics appear healthy. Continue monitoring and current engagement strategies.
                    </div>
                </div>""", unsafe_allow_html=True)
        with r2:
            st.plotly_chart(gauge_chart(global_proba, global_is_churn),
                           use_container_width=True, config={"displayModeBar": False})

        st.markdown('<p class="sec-title">Key Risk Drivers</p>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (level, title, desc) in enumerate(drivers_list):
            with cols[i]:
                st.markdown(f"""
                <div class="driver-card driver-{level}">
                    <div style="font-weight:700; font-size:0.95rem; color:#FFFFFF;">{title}</div>
                    <div style="font-size:0.82rem; color:#E2E8F0; margin-top:6px; line-height:1.45;">{desc}</div>
                </div>""", unsafe_allow_html=True)
    else:
        st.error("Model files not found. Run `python3 src/train_model.py` first.")


# ─── Tab 3: Agentic AI ───────────────────────────────────────────────────────
with tab3:
    st.markdown('<p class="sec-title" style="margin-top:0;">🤖 Agentic AI Retention Strategy</p>', unsafe_allow_html=True)
    st.caption("Autonomously analyzes risk, retrieves best practices via RAG, and generates structured retention reports using LangGraph.")

    # Architecture diagram
    st.markdown("""
    <div class="card" style="padding:16px 20px;">
        <div style="font-size:0.78rem; font-weight:600; color:#A78BFA; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px;">LangGraph Workflow Pipeline</div>
        <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap;">
            <div style="background:rgba(99,102,241,0.2); border:1px solid rgba(99,102,241,0.3); border-radius:8px; padding:8px 14px; font-size:0.8rem; color:#A78BFA; font-weight:600;">① Analyze Risk</div>
            <div style="color:#475569; font-size:0.9rem;">→</div>
            <div style="background:rgba(59,130,246,0.2); border:1px solid rgba(59,130,246,0.3); border-radius:8px; padding:8px 14px; font-size:0.8rem; color:#60A5FA; font-weight:600;">② RAG Retrieve (FAISS)</div>
            <div style="color:#475569; font-size:0.9rem;">→</div>
            <div style="background:rgba(16,185,129,0.2); border:1px solid rgba(16,185,129,0.3); border-radius:8px; padding:8px 14px; font-size:0.8rem; color:#34D399; font-weight:600;">③ Generate Report</div>
            <div style="color:#475569; font-size:0.9rem;">→</div>
            <div style="background:rgba(244,114,182,0.2); border:1px solid rgba(244,114,182,0.3); border-radius:8px; padding:8px 14px; font-size:0.8rem; color:#F472B6; font-weight:600;">📄 Structured Output</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if log_model is None:
        st.error("Model artifacts not found. Please run `python3 src/train_model.py` first.")
    else:
        if st.button("🚀 Generate Retention Strategy", use_container_width=True):
            from src.agent import build_retention_agent

            agent = build_retention_agent()
            # Retrieve API key: st.secrets (Streamlit Cloud) → os.getenv (local .env)
            _api_key = None
            try:
                _api_key = st.secrets["OPENROUTER_API_KEY"]
            except (KeyError, FileNotFoundError):
                _api_key = os.getenv("OPENROUTER_API_KEY")

            initial_state = {
                "customer_profile": {
                    "tenure": tenure,
                    "monthly_charges": monthly,
                    "total_charges": total,
                    "support_calls": support,
                    "avg_monthly_spend": round(avg_spend, 2),
                },
                "churn_probability": float(global_proba),
                "churn_risk_level": "",
                "drivers": [],
                "retrieved_strategies": "",
                "report": "",
                "error": None,
                "api_key": _api_key,
            }

            # Generator for typewriter effect
            def stream_data(text, delay=0.015):
                """Simulates a live streaming output."""
                for word in text.split(" "):
                    yield word + " "
                    time.sleep(delay)

            # Interactive Pipeline Status
            with st.status("🧠 **Agentic AI Pipeline Initiated...**", expanded=True) as status:
                st.write("🕵️ Analyzing customer risk profile and key drivers...")
                time.sleep(0.6)
                
                st.write("📚 Querying FAISS vector store for top retention strategies...")
                time.sleep(0.6)
                
                st.write("✍️ Synthesizing tailored report using Claude 3 Haiku...")
                
                # Execute graph
                result = agent.invoke(initial_state)
                
                if result.get("error"):
                    status.update(label="Workflow Error encountered.", state="error", expanded=True)
                    st.error(result['error'])
                else:
                    status.update(label="Strategy generated successfully!", state="complete", expanded=False)

            # Display the interactive report
            if not result.get("error"):
                st.toast("Report generated successfully!", icon="🎉")
                st.balloons()

                # Report header styling
                st.markdown(f"""
                <div class="agent-report" style="margin-bottom: 0; border-bottom-left-radius: 0; border-bottom-right-radius: 0; box-shadow: none;">
                    <div style="font-size:1.1rem; font-weight:700; color:#A78BFA; margin-bottom:0px; display:flex; align-items:center; gap:8px;">
                        📄 Agentic Retention Report
                        <span style="font-size:0.7rem; background:rgba(139,92,246,0.2); border:1px solid rgba(139,92,246,0.3); border-radius:20px; padding:3px 10px; color:#A78BFA; font-weight:600;">LangGraph + OpenRouter</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Premium conversational streaming output
                with st.chat_message("assistant", avatar="🤖"):
                    st.write_stream(stream_data(result["report"]))


# ─── Tab 4: Model Metrics ────────────────────────────────────────────────────
with tab4:
    st.markdown('<p class="sec-title" style="margin-top:0;">Model Performance</p>', unsafe_allow_html=True)
    st.caption("Evaluation metrics and comparative analysis of trained algorithms.")

    if metrics is None:
        st.error("Run `python3 src/train_model.py` first.")
    else:
        log_m = metrics["logistic_regression"]
        dt_m = metrics["decision_tree"]

        comp = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
            "Logistic Regression": [f"{log_m[k]*100:.1f}%" for k in ["accuracy", "precision", "recall", "f1"]],
            "Decision Tree": [f"{dt_m[k]*100:.1f}%" for k in ["accuracy", "precision", "recall", "f1"]],
        })
        st.dataframe(comp, hide_index=True, use_container_width=True)

        st.markdown('<p class="sec-title">Confusion Matrices</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(confusion_heatmap(log_m["confusion_matrix"], "Logistic Regression"),
                            use_container_width=True, config={"displayModeBar": False})
        with c2:
            st.plotly_chart(confusion_heatmap(dt_m["confusion_matrix"], "Decision Tree"),
                            use_container_width=True, config={"displayModeBar": False})

        if "feature_importance" in metrics:
            st.markdown('<p class="sec-title">Feature Importance</p>', unsafe_allow_html=True)
            st.plotly_chart(importance_chart(metrics["feature_importance"]),
                            use_container_width=True, config={"displayModeBar": False})

        st.markdown('<p class="sec-title">Business Insights</p>', unsafe_allow_html=True)
        st.info("""
        **Data Patterns Recognized by Models:**
        - **Tenure** remains the strongest predictor of retention; first-year attrition is critically high.
        - **High monthly charges** act as an independent strong catalyst for churn risk.
        - **Frequent support calls** signal deep dissatisfaction preceding departure.
        - **Algorithm Details:** Logistic Regression generalises better on test data. The Decision Tree captures non-linear patterns but may overfit without depth constraints.
        """, icon="🧠")
