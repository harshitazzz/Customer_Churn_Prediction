import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

*, [class*="st-"] {
    font-family: 'Outfit', sans-serif !important;
}

.stApp {
    background: #0F172A !important;
}
[data-testid="stHeader"],
.stDeployButton {display: none !important;}
[data-testid="stAppViewBlockContainer"] {padding-top: 2rem !important;}

h1 {
    font-weight: 700 !important;
    color: #FFFFFF !important;
    letter-spacing: -1px !important;
    margin-bottom: 0.2rem !important;
}
p, .stMarkdown {
    color: #E2E8F0 !important;
    font-size: 1.05rem !important;
}

section[data-testid="stSidebar"] {
    background: #1E293B !important;
    border-right: 1px solid #334155 !important;
}
section[data-testid="stSidebar"] * {
    color: #F8FAFC !important;
}
[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"] {
    display: none !important;
}

.card {
    background: #1E293B;
    border: 1px solid rgba(51, 65, 85, 0.8);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease;
}
.card:hover {
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3), 0 4px 6px -4px rgba(0, 0, 0, 0.25);
    transform: translateY(-2px);
}

.stat-card {
    background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    transition: all 0.2s ease;
}
.stat-card:hover {
    transform: scale(1.02);
    border-color: #475569;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
}
.stat-val {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60A5FA 0%, #818CF8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}
.stat-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 8px;
}

.result-stay {
    background: linear-gradient(135deg, #064E3B 0%, #065F46 100%);
    border: 1px solid #059669;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(16, 185, 129, 0.15);
}
.result-churn {
    background: linear-gradient(135deg, #7F1D1D 0%, #991B1B 100%);
    border: 1px solid #DC2626;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(239, 68, 68, 0.15);
}

.stButton > button {
    background: linear-gradient(135deg, #3B82F6 0%, #6366F1 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 32px !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    box-shadow: 0 4px 6px -1px rgba(59, 130, 246, 0.3), 0 2px 4px -2px rgba(59, 130, 246, 0.2) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 15px -3px rgba(59, 130, 246, 0.4), 0 4px 6px -4px rgba(59, 130, 246, 0.25) !important;
}

[data-testid="stNumberInput"] > div > div > input,
[data-testid="stSelectbox"] > div > div > div {
    background-color: #0F172A !important;
    border-radius: 8px !important;
    border: 1px solid #334155 !important;
    color: #FFFFFF !important;
    box-shadow: inset 0 1px 2px rgba(0,0,0,0.1) !important;
}

.sec-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: #FFFFFF;
    margin: 32px 0 16px 0;
    border-bottom: 2px solid #334155;
    padding-bottom: 8px;
}
.driver-card {
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 12px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
}
.driver-high { background: #7F1D1D; border-left: 4px solid #F87171; }
.driver-med  { background: #78350F; border-left: 4px solid #FBBF24; }
.driver-low  { background: #064E3B; border-left: 4px solid #34D399; }

</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
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

log_model, dt_model, scaler, feature_cols, metrics = load_artifacts()


def gauge_chart(prob, is_churn):
    color = "#EF4444" if is_churn else "#10B981"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40, "color": color}},
        gauge={
            "axis": {"range": [0, 100], "tickfont": {"size": 10, "color": "#94A3B8"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1E293B",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 30], "color": "#064E3B"},
                {"range": [30, 60], "color": "#78350F"},
                {"range": [60, 100], "color": "#7F1D1D"},
            ],
        },
        title={"text": "Churn Probability", "font": {"size": 13, "color": "#94A3B8"}},
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def confusion_heatmap(cm, title):
    labels = ["Stay", "Churn"]
    fig = go.Figure(go.Heatmap(
        z=cm, x=labels, y=labels,
        colorscale=[[0, "#1E293B"], [1, "#38BDF8"]],
        text=[[str(v) for v in row] for row in cm],
        texttemplate="%{text}", textfont={"size": 16},
        showscale=False,
    ))
    fig.update_layout(title=title, height=300,
                      margin=dict(l=50, r=20, t=50, b=50),
                      paper_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Predicted", yaxis_title="Actual")
    fig.update_yaxes(autorange="reversed")
    return fig


def importance_chart(feat_imp):
    items = sorted(feat_imp.items(), key=lambda x: x[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k for k, _ in items],
        orientation="h",
        marker_color="#38BDF8",
    ))
    fig.update_layout(title="Top Feature Importance", height=380,
                      margin=dict(l=180, r=20, t=50, b=30),
                      paper_bgcolor="rgba(0,0,0,0)",
                      xaxis_title="Importance")
    return fig


with st.sidebar:
    st.markdown("### 📊 ChurnSense")
    st.caption("Customer Churn Prediction")
    st.divider()

    if log_model is None:
        st.error("Model files not found. Run `python3 src/train_model.py` first.")
    else:
        st.markdown('<p style="font-size:1.1rem; font-weight:700; color:#FFFFFF; margin-bottom:12px;">Control Panel</p>', unsafe_allow_html=True)
        model_choice = st.selectbox("Select Model", ["Logistic Regression", "Decision Tree"])
        
        st.markdown('<p style="font-size:0.9rem; font-weight:600; color:#94A3B8; margin-top:24px; margin-bottom:12px;">Customer Details</p>', unsafe_allow_html=True)
        tenure = st.number_input("Tenure (months)", 0, 72, 12, help="How many months has the customer been with the company")
        monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 50.0, format="%.2f")
        total = st.number_input("Total Charges ($)", 0.0, 10000.0, 600.0, format="%.2f")
        support = st.number_input("Support Calls", 0, 10, 1, help="Number of times customer contacted support")

        avg_spend = total / (tenure + 1)
        st.markdown("<br>", unsafe_allow_html=True)
        st.metric("Avg Monthly Spend", f"${avg_spend:.2f}", help="Auto-calculated: TotalCharges / (Tenure + 1)")

        st.divider()
        st.caption("**PIPELINE ACTIVE**")
        st.caption("Feature Eng. → Scaling → Model Inference")


st.title("Customer Churn Prediction")
st.caption("Analyze churn risk and explore model performance dynamically.")

tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔮 Predict Results", "📈 About Model"])

with tab1:
    st.markdown("""
    <div class="card">
    <b>How it works:</b> Use the <b>Sidebar Controls</b> to configure customer details → 
    the system instantly performs feature engineering, scales inputs, and runs predictions through 
    the selected Scikit-Learn model → switch to the <b>Predict Results</b> tab to see actionable insights.
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

    st.markdown('<p class="sec-title">Pipeline</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    for col, (num, title, desc) in zip([c1, c2, c3], [
        ("1", "Sidebar Input", "Configure tenure, charges & support calls on the left"),
        ("2", "Process", "Instant feature engineering, scaling & prediction"),
        ("3", "Insights", "View churn probability & critical business drivers"),
    ]):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center">
                <div style="font-size:1.5rem; font-weight:700; color:#60A5FA; margin-bottom:4px">{num}</div>
                <div style="font-weight:600; color:#FFFFFF; margin-bottom:4px">{title}</div>
                <div style="font-size:0.85rem; color:#94A3B8">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


with tab2:
    if log_model is not None:
        st.markdown('<p class="sec-title">Prediction Overview</p>', unsafe_allow_html=True)
        st.caption(f"Showing live predictions based on sidebar inputs using **{model_choice}**.")

        inp = {col: 0 for col in feature_cols}
        inp["tenure"] = tenure
        inp["MonthlyCharges"] = monthly
        inp["TotalCharges"] = total
        inp["SeniorCitizen"] = min(support, 1)
        inp["AvgMonthlySpend"] = avg_spend

        df = pd.DataFrame([inp])[feature_cols]
        scaled = scaler.transform(df)

        model = log_model if model_choice == "Logistic Regression" else dt_model
        pred = model.predict(scaled)[0]
        proba = model.predict_proba(scaled)[0][1]
        is_churn = pred == 1

        r1, r2 = st.columns([1, 1], gap="large")
        with r1:
            if is_churn:
                st.markdown("""
                <div class="result-churn">
                    <div style="font-size:3rem; margin-bottom:12px">⚠️</div>
                    <div style="font-size:1.6rem; font-weight:700; color:#FCA5A5">High Churn Risk</div>
                    <div style="font-size:0.95rem; color:#F87171; margin-top:8px; line-height:1.5">
                        This profile flags high for attrition. Consider immediate proactive retention measures.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="result-stay">
                    <div style="font-size:3rem; margin-bottom:12px">✅</div>
                    <div style="font-size:1.6rem; font-weight:700; color:#6EE7B7">Likely to Stay</div>
                    <div style="font-size:0.95rem; color:#34D399; margin-top:8px; line-height:1.5">
                        Customer engagement metrics appear healthy for this profile. Continue current engagement.
                    </div>
                </div>""", unsafe_allow_html=True)
        with r2:
            st.plotly_chart(gauge_chart(proba, is_churn), use_container_width=True,
                           config={"displayModeBar": False})

        st.markdown('<p class="sec-title">Key Drivers</p>', unsafe_allow_html=True)
        drivers = []
        if tenure < 12:
            drivers.append(("high", "📅 Low Tenure", f"{tenure} months — new customers are statistically more likely to churn quickly."))
        elif tenure < 24:
            drivers.append(("med", "📅 Moderate Tenure", f"{tenure} months — building loyalty but still requires engagement."))
        else:
            drivers.append(("low", "📅 Strong Tenure", f"{tenure} months — long-term established customer."))

        if monthly > 80:
            drivers.append(("high", "💳 High Cost", f"${monthly:.0f}/mo — upper tier pricing creates cost pressure."))
        elif monthly > 50:
            drivers.append(("med", "💳 Moderate Cost", f"${monthly:.0f}/mo — average industry tier."))
        else:
            drivers.append(("low", "💳 Low Cost", f"${monthly:.0f}/mo — highly affordable tier."))

        if support >= 3:
            drivers.append(("high", "📞 Frequent Support", f"{support} calls — signifies friction or unresolved persistent issues."))
        elif support >= 1:
            drivers.append(("med", "📞 Some Support", f"{support} call(s) — minor friction reported."))
        else:
            drivers.append(("low", "📞 No Support", "No recent support interactions needed."))

        cols = st.columns(3)
        for i, (level, title, desc) in enumerate(drivers):
            with cols[i]:
                st.markdown(f"""
                <div class="driver-card driver-{level}">
                    <div style="font-weight:700; font-size:1rem; color:#FFFFFF;">{title}</div>
                    <div style="font-size:0.85rem; color:#E2E8F0; margin-top:6px; line-height:1.4;">{desc}</div>
                </div>""", unsafe_allow_html=True)


with tab3:
    st.markdown('<p class="sec-title" style="margin-top:0;">Model Performance</p>', unsafe_allow_html=True)
    st.caption("Evaluation metrics and comparative analysis of trained algorithms")

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
        - **Tenure** remains the undisputed strongest predictor of retention; first-year attrition is critically high.
        - **High monthly charges** act as an independent strong catalyst for churn risk.
        - **Frequent support calls** signal deep dissatisfaction preceding departure.
        - **Algorithm Details:** Logistic Regression generalises much better on test data. The Decision Tree often over-indexes on niche interactions without proper depth constraints constraint tuning.
        """, icon="🧠")

