"""
Fraud Detection Dashboard — Adapted for real credit card dataset
Columns: TransactionID, TransactionDate, Amount, MerchantID,
         TransactionType, Location, IsFraud
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib, json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍", layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252b3b);
        border: 1px solid #2d3547; border-radius: 12px;
        padding: 20px; text-align: center;
    }
    .metric-val   { font-size: 2.2rem; font-weight: 700; }
    .metric-label { font-size: 0.85rem; color: #9aa0b0; margin-top: 4px; }
    .fraud-val    { color: #ff4d4d; }
    .safe-val     { color: #00cc88; }
    .neutral-val  { color: #4d9fff; }
    .warn-val     { color: #ffd700; }
    .section-title {
        font-size: 1.05rem; font-weight: 600;
        border-left: 3px solid #4d9fff;
        padding-left: 10px; margin: 18px 0 10px 0; color: #e0e6f0;
    }
</style>
""", unsafe_allow_html=True)

# ── Load assets ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("fraud_with_predictions.csv", parse_dates=["TransactionDate"])
    return df

@st.cache_resource
def load_models():
    rf     = joblib.load("fraud_model.pkl")
    iso    = joblib.load("iso_model.pkl")
    scaler = joblib.load("scaler.pkl")
    le_loc = joblib.load("le_location.pkl")
    return rf, iso, scaler, le_loc

@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)

df              = load_data()
rf, iso, scaler, le_loc = load_models()
metrics         = load_metrics()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("🔍 Fraud Shield")
    st.caption("ML-powered transaction monitoring")
    st.divider()

    st.subheader("⚙️ Filters")
    sel_loc = st.multiselect("City / Location",
        options=sorted(df["Location"].unique().tolist()),
        default=sorted(df["Location"].unique().tolist()))
    sel_type = st.multiselect("Transaction Type",
        options=df["TransactionType"].unique().tolist(),
        default=df["TransactionType"].unique().tolist())
    amount_range = st.slider("Amount Range ($)",
        float(df["Amount"].min()), float(df["Amount"].max()),
        (float(df["Amount"].min()), float(df["Amount"].max())))
    sel_month = st.multiselect("Month",
        options=list(range(1,13)),
        default=list(range(1,13)),
        format_func=lambda m: ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"][m-1])
    fraud_only = st.checkbox("Show Fraud Only")

    st.divider()
    st.subheader("📊 Model Info")
    st.info(f"**Random Forest**\n\n200 trees | Depth 14\n\nROC-AUC: **{metrics['roc_auc']}**")
    st.success(f"Train: **{metrics['total_train']:,}** | Test: **{metrics['total_test']:,}**")

# ── Apply filters ─────────────────────────────────────────────
month_col = df["TransactionDate"].dt.month
mask = (
    df["Location"].isin(sel_loc) &
    df["TransactionType"].isin(sel_type) &
    df["Amount"].between(*amount_range) &
    month_col.isin(sel_month)
)
if fraud_only:
    mask &= df["prediction"] == 1
dff = df[mask]

# ── Header ────────────────────────────────────────────────────
st.markdown("## 🔍 Credit Card Fraud Detection Dashboard")
st.caption(f"Showing **{len(dff):,}** of {len(df):,} transactions  |  "
           f"Period: {df['TransactionDate'].min().date()} → {df['TransactionDate'].max().date()}")
st.divider()

# ── KPI Cards ─────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
total      = len(dff)
fraud_n    = int(dff["prediction"].sum())
safe_n     = total - fraud_n
fraud_pct  = (fraud_n / total * 100) if total else 0
avg_amt    = dff["Amount"].mean()

def kpi(col, label, value, style, prefix="", suffix=""):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-val {style}">{prefix}{value}{suffix}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

kpi(k1, "Total Transactions", f"{total:,}",        "neutral-val")
kpi(k2, "Fraud Flagged",      f"{fraud_n:,}",       "fraud-val")
kpi(k3, "Legitimate",         f"{safe_n:,}",         "safe-val")
kpi(k4, "Fraud Rate",         f"{fraud_pct:.2f}",    "warn-val",    suffix="%")
kpi(k5, "Avg. Amount",        f"{avg_amt:.0f}",      "neutral-val", prefix="$")

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Overview", "💰 Amounts", "📅 Time Trends",
    "🧠 Model Performance", "🔎 Explorer",
    "🔮 Single Prediction", "📂 Batch Prediction"
])

COLORS = {"Legitimate": "#00cc88", "Fraud": "#ff4d4d"}
BG     = "rgba(0,0,0,0)"
FONT   = "#e0e6f0"

def base_layout(fig, height=340):
    fig.update_layout(
        paper_bgcolor=BG, plot_bgcolor=BG,
        font_color=FONT, height=height, margin=dict(t=20, b=20),
        legend=dict(font=dict(color=FONT))
    )
    return fig

# ═══════ TAB 1 — OVERVIEW ════════════════════════════════════
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Fraud vs Legitimate</div>', unsafe_allow_html=True)
        pie_df = pd.DataFrame({"Status": ["Legitimate","Fraud"], "Count": [safe_n, fraud_n]})
        fig = px.pie(pie_df, values="Count", names="Status", hole=0.55,
                     color="Status", color_discrete_map=COLORS)
        fig.add_annotation(text=f"<b>{fraud_pct:.1f}%</b><br>Fraud",
                           x=0.5, y=0.5, showarrow=False,
                           font=dict(size=15, color="#ff4d4d"))
        st.plotly_chart(base_layout(fig, 320), use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Fraud Rate by City</div>', unsafe_allow_html=True)
        city_df = (dff.groupby("Location")
                      .agg(fraud=("prediction","sum"), total=("prediction","count"))
                      .assign(rate=lambda x: x.fraud/x.total*100)
                      .sort_values("rate").reset_index())
        fig2 = px.bar(city_df, x="rate", y="Location", orientation="h",
                      color="rate", color_continuous_scale=["#00cc88","#ffd700","#ff4d4d"],
                      labels={"rate":"Fraud Rate (%)","Location":""})
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(base_layout(fig2, 320), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown('<div class="section-title">Fraud by Transaction Type</div>', unsafe_allow_html=True)
        type_df = (dff.groupby("TransactionType")
                      .agg(fraud=("prediction","sum"), total=("prediction","count"))
                      .assign(rate=lambda x: x.fraud/x.total*100)
                      .reset_index())
        fig3 = px.bar(type_df, x="TransactionType", y="rate",
                      color="TransactionType",
                      color_discrete_sequence=["#4d9fff","#ff4d4d"],
                      labels={"TransactionType":"", "rate":"Fraud Rate (%)"},
                      text=type_df["rate"].map("{:.2f}%".format))
        fig3.update_traces(textposition="outside")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(base_layout(fig3, 300), use_container_width=True)

    with c4:
        st.markdown('<div class="section-title">Fraud Probability Distribution</div>', unsafe_allow_html=True)
        fig4 = px.histogram(dff, x="fraud_prob",
                            color=dff["prediction"].map({0:"Legitimate",1:"Fraud"}),
                            nbins=40, barmode="overlay", opacity=0.72,
                            color_discrete_map=COLORS,
                            labels={"fraud_prob":"Fraud Score","color":"Label"})
        fig4.update_layout(showlegend=True)
        st.plotly_chart(base_layout(fig4, 300), use_container_width=True)

# ═══════ TAB 2 — AMOUNTS ═════════════════════════════════════
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="section-title">Amount Distribution by Label</div>', unsafe_allow_html=True)
        fig = px.box(dff, x=dff["prediction"].map({0:"Legitimate",1:"Fraud"}),
                     y="Amount",
                     color=dff["prediction"].map({0:"Legitimate",1:"Fraud"}),
                     color_discrete_map=COLORS,
                     labels={"x":"","Amount":"Transaction Amount ($)"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(base_layout(fig, 340), use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Amount vs Fraud Score</div>', unsafe_allow_html=True)
        samp = dff.sample(min(4000, len(dff)), random_state=42)
        fig2 = px.scatter(samp, x="Amount", y="fraud_prob",
                          color=samp["prediction"].map({0:"Legitimate",1:"Fraud"}),
                          color_discrete_map=COLORS, opacity=0.5,
                          labels={"Amount":"Amount ($)","fraud_prob":"Fraud Score"})
        st.plotly_chart(base_layout(fig2, 340), use_container_width=True)

    st.markdown('<div class="section-title">Fraud Rate by Amount Bucket</div>', unsafe_allow_html=True)
    bucket_labels = {0:"$0–500", 1:"$500–1k", 2:"$1k–2k", 3:"$2k–3.5k", 4:"$3.5k+"}
    bdf = (dff.groupby("amount_bucket")
              .agg(fraud=("prediction","sum"), total=("prediction","count"))
              .assign(rate=lambda x: x.fraud/x.total*100)
              .reset_index())
    bdf["Bucket"] = bdf["amount_bucket"].map(bucket_labels)
    fig3 = px.bar(bdf, x="Bucket", y="rate",
                  color="rate", color_continuous_scale=["#00cc88","#ffd700","#ff4d4d"],
                  labels={"rate":"Fraud Rate (%)","Bucket":""},
                  text=bdf["rate"].map("{:.1f}%".format))
    fig3.update_traces(textposition="outside")
    fig3.update_layout(coloraxis_showscale=False)
    st.plotly_chart(base_layout(fig3, 300), use_container_width=True)

# ═══════ TAB 3 — TIME TRENDS ═════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Monthly Fraud Trend</div>', unsafe_allow_html=True)
    dff2 = dff.copy()
    dff2["month"] = dff2["TransactionDate"].dt.month
    month_df = (dff2.groupby("month")
                    .agg(total=("prediction","count"), fraud=("prediction","sum"))
                    .assign(rate=lambda x: x.fraud/x.total*100)
                    .reset_index())
    month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    month_df["Month"] = month_df["month"].apply(lambda m: month_names[m-1])

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=month_df["Month"], y=month_df["total"],
                         name="Total Tx", marker_color="#2d3a5e", opacity=0.75))
    fig.add_trace(go.Scatter(x=month_df["Month"], y=month_df["rate"],
                             name="Fraud Rate %", mode="lines+markers",
                             line=dict(color="#ff4d4d", width=2.5),
                             marker=dict(size=8)), secondary_y=True)
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color=FONT,
                      height=340, margin=dict(t=10,b=20),
                      legend=dict(font=dict(color=FONT)))
    fig.update_yaxes(title_text="Transactions", secondary_y=False)
    fig.update_yaxes(title_text="Fraud Rate (%)", secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-title">Fraud by Day of Week</div>', unsafe_allow_html=True)
        dff2["dow"] = dff2["TransactionDate"].dt.dayofweek
        dow_df = (dff2.groupby("dow")
                      .agg(fraud=("prediction","sum"), total=("prediction","count"))
                      .assign(rate=lambda x: x.fraud/x.total*100)
                      .reset_index())
        dow_df["Day"] = dow_df["dow"].map({0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"})
        fig2 = px.bar(dow_df, x="Day", y="rate",
                      color="rate", color_continuous_scale=["#00cc88","#ffd700","#ff4d4d"],
                      labels={"rate":"Fraud Rate (%)","Day":""})
        fig2.update_layout(coloraxis_showscale=False)
        st.plotly_chart(base_layout(fig2, 300), use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Weekend vs Weekday Fraud</div>', unsafe_allow_html=True)
        wkd_df = (dff2.groupby("is_weekend")["prediction"]
                      .mean().reset_index())
        wkd_df["Period"] = wkd_df["is_weekend"].map({0:"Weekday",1:"Weekend"})
        fig3 = px.bar(wkd_df, x="Period", y="prediction",
                      color="Period",
                      color_discrete_sequence=["#4d9fff","#7b2ff7"],
                      labels={"prediction":"Fraud Rate","Period":""},
                      text=wkd_df["prediction"].map("{:.2%}".format))
        fig3.update_traces(textposition="outside")
        fig3.update_layout(showlegend=False)
        st.plotly_chart(base_layout(fig3, 300), use_container_width=True)

    st.markdown('<div class="section-title">Fraud by Quarter</div>', unsafe_allow_html=True)
    dff2["quarter"] = dff2["TransactionDate"].dt.quarter
    q_df = (dff2.groupby("quarter")
                .agg(fraud=("prediction","sum"), total=("prediction","count"))
                .assign(rate=lambda x: x.fraud/x.total*100)
                .reset_index())
    q_df["Quarter"] = q_df["quarter"].map({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
    fig4 = px.bar(q_df, x="Quarter", y="rate",
                  color="rate", color_continuous_scale=["#00cc88","#ffd700","#ff4d4d"],
                  labels={"rate":"Fraud Rate (%)","Quarter":""},
                  text=q_df["rate"].map("{:.1f}%".format))
    fig4.update_traces(textposition="outside")
    fig4.update_layout(coloraxis_showscale=False)
    st.plotly_chart(base_layout(fig4, 280), use_container_width=True)

# ═══════ TAB 4 — MODEL PERFORMANCE ══════════════════════════
with tab4:
    c1, c2 = st.columns([1,1])

    with c1:
        st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)
        cm_vals = [[metrics["tn"], metrics["fp"]],
                   [metrics["fn"], metrics["tp"]]]
        fig = px.imshow(cm_vals,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Predicted Legit","Predicted Fraud"],
            y=["Actual Legit","Actual Fraud"],
            color_continuous_scale=["#0e1117","#4d9fff","#ff4d4d"],
            text_auto=True)
        fig.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                          font_color=FONT, height=320, margin=dict(t=20,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown('<div class="section-title">Performance Metrics</div>', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("Precision", f"{metrics['precision']:.4f}")
        m2.metric("Recall",    f"{metrics['recall']:.4f}")
        m1.metric("F1 Score",  f"{metrics['f1']:.4f}")
        m2.metric("ROC-AUC",   f"{metrics['roc_auc']:.4f}")

        st.markdown('<div class="section-title">Feature Importance</div>', unsafe_allow_html=True)
        feat_df = pd.DataFrame({
            "Feature":    metrics["features"],
            "Importance": rf.feature_importances_
        }).sort_values("Importance", ascending=True)
        fig2 = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale=["#2d3a5e","#4d9fff"])
        fig2.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color=FONT,
                           coloraxis_showscale=False, height=340, margin=dict(t=10,b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-title">Isolation Forest — Anomaly Detection</div>',
                unsafe_allow_html=True)
    i1, i2, i3 = st.columns(3)
    iso_n = int(dff["iso_anomaly"].sum())
    i1.metric("Anomalies Flagged", f"{iso_n:,}")
    i2.metric("Normal",            f"{len(dff)-iso_n:,}")
    i3.metric("Anomaly Rate",      f"{iso_n/len(dff)*100:.2f}%" if len(dff) else "—")
    overlap = int(((dff["prediction"]==1) & (dff["iso_anomaly"]==1)).sum())
    st.info(f"🔗 **{overlap:,}** transactions flagged by **both** models — highest-confidence fraud cases.")

# ── Helper: build feature row ────────────────────────────────
DOW_MAP   = {"Mon":0,"Tue":1,"Wed":2,"Thu":3,"Fri":4,"Sat":5,"Sun":6}
MONTH_MAP = {"Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
             "Jul":7,"Aug":8,"Sep":9,"Oct":10,"Nov":11,"Dec":12}
MEDIAN_FREQ = int(df["MerchantID"].value_counts().median())

def build_row(amount, tx_type, location, merch_id, dow_str, month_str, day=15):
    dow_val   = DOW_MAP[dow_str]
    month_val = MONTH_MAP[month_str]
    is_wknd   = 1 if dow_val >= 5 else 0
    quarter   = (month_val - 1) // 3 + 1
    amt_sc    = scaler.transform([[amount]])[0][0]
    amt_bkt   = min(4, int(amount>500)+int(amount>1000)+int(amount>2000)+int(amount>3500))
    type_enc  = 1 if tx_type == "refund" else 0
    loc_enc   = le_loc.transform([location])[0]
    mfreq     = int(df["MerchantID"].value_counts().get(merch_id, MEDIAN_FREQ))
    return [[amount, amt_sc, amt_bkt,
             dow_val, month_val, day, is_wknd, quarter,
             merch_id, mfreq, type_enc, loc_enc]]

# ═══════ TAB 5 — EXPLORER ════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Transaction Explorer</div>', unsafe_allow_html=True)
    view = st.radio("View", ["All","Fraud Only","Legit Only"], horizontal=True)
    view_df = dff if view == "All" else dff[dff["prediction"]==(1 if view=="Fraud Only" else 0)]

    show_cols = ["TransactionID","TransactionDate","Amount","MerchantID",
                 "TransactionType","Location","IsFraud","fraud_prob","prediction","iso_anomaly"]
    show_df = view_df[show_cols].sort_values("fraud_prob", ascending=False).head(300).copy()
    show_df["prediction"]  = show_df["prediction"].map({0:"✅ Legit",1:"🚨 Fraud"})
    show_df["iso_anomaly"] = show_df["iso_anomaly"].map({0:"Normal",1:"⚠️ Anomaly"})
    show_df["fraud_prob"]  = show_df["fraud_prob"].map("{:.3f}".format)
    st.dataframe(show_df, use_container_width=True, height=500)
    st.caption(f"Top 300 by fraud probability from {len(view_df):,} filtered rows")

# ═══════ TAB 6 — SINGLE PREDICTION ══════════════════════════
with tab6:
    st.markdown("### 🔮 Single Transaction Fraud Predictor")
    st.caption("Fill in the transaction details and get an instant fraud risk assessment from the trained model.")
    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**💳 Transaction Details**")
        inp_amount = st.number_input("Amount ($)", 1.0, 5000.0, 2500.0, step=50.0)
        inp_type   = st.selectbox("Transaction Type", ["purchase", "refund"])
        inp_merch  = st.number_input("Merchant ID", 1, 1000, 500)
    with c2:
        st.markdown("**📍 Location & Time**")
        inp_loc   = st.selectbox("City", sorted(metrics["locations"]))
        inp_dow   = st.selectbox("Day of Week", list(DOW_MAP.keys()))
        inp_month = st.selectbox("Month", list(MONTH_MAP.keys()))
    with c3:
        st.markdown("**⚙️ Risk Indicators**")
        dow_val   = DOW_MAP[inp_dow]
        month_val = MONTH_MAP[inp_month]
        st.info(f"""
        **Pre-computed signals:**
        - Is Weekend: {'Yes ⚠️' if dow_val >= 5 else 'No ✅'}
        - Is Refund: {'Yes ⚠️' if inp_type == 'refund' else 'No ✅'}
        - High Amount: {'Yes ⚠️' if inp_amount > 3000 else 'No ✅'}
        - High-Risk City: {'Yes ⚠️' if inp_loc in ['New York','Los Angeles','Houston'] else 'No ✅'}
        - Quarter: Q{(month_val - 1) // 3 + 1}
        """)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔍  Run Fraud Prediction", use_container_width=True, type="primary"):
        row  = build_row(inp_amount, inp_type, inp_loc, inp_merch, inp_dow, inp_month)
        prob = rf.predict_proba(row)[0][1]
        pred = rf.predict(row)[0]
        iso_r= iso.predict(row)[0]

        st.divider()
        # ── Verdict banner ────────────────────────────────────
        if prob >= 0.5:
            st.error(f"## 🚨 FRAUD DETECTED\nModel confidence: **{prob*100:.1f}%** — Block this transaction immediately.")
        elif prob >= 0.25:
            st.warning(f"## ⚠️ SUSPICIOUS TRANSACTION\nRisk score: **{prob*100:.1f}%** — Flag for manual review.")
        else:
            st.success(f"## ✅ LEGITIMATE TRANSACTION\nRisk score: **{prob*100:.1f}%** — Safe to approve.")

        # ── Metrics row ───────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fraud Probability",  f"{prob*100:.2f}%")
        m2.metric("Legit Probability",  f"{(1-prob)*100:.2f}%")
        m3.metric("Random Forest",      "🚨 Fraud" if pred == 1 else "✅ Legit")
        m4.metric("Isolation Forest",   "⚠️ Anomaly" if iso_r == -1 else "✅ Normal")

        # ── Probability gauge bar ─────────────────────────────
        st.markdown('<div class="section-title">Risk Score Gauge</div>', unsafe_allow_html=True)
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(prob * 100, 2),
            delta={"reference": 25, "valueformat": ".1f"},
            title={"text": "Fraud Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#ff4d4d" if prob > 0.5 else "#ffd700" if prob > 0.25 else "#00cc88"},
                "steps": [
                    {"range": [0,  25],  "color": "#1a2b1a"},
                    {"range": [25, 50],  "color": "#2b2a1a"},
                    {"range": [50, 100], "color": "#2b1a1a"},
                ],
                "threshold": {"line": {"color": "white", "width": 3}, "value": 50}
            }
        ))
        fig_g.update_layout(paper_bgcolor=BG, font_color=FONT, height=300, margin=dict(t=30,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

        # ── Feature contribution breakdown ────────────────────
        st.markdown('<div class="section-title">Why this prediction? — Feature Breakdown</div>',
                    unsafe_allow_html=True)
        feat_names = metrics["features"]
        feat_imp   = rf.feature_importances_
        feat_vals  = row[0]
        contrib_df = pd.DataFrame({
            "Feature":    feat_names,
            "Importance": feat_imp,
            "Value":      [f"{v:.3f}" for v in feat_vals]
        }).sort_values("Importance", ascending=True)
        fig_c = px.bar(contrib_df, x="Importance", y="Feature", orientation="h",
                       color="Importance", color_continuous_scale=["#2d3a5e","#4d9fff","#ff4d4d"],
                       hover_data={"Value": True},
                       labels={"Importance":"Feature Weight","Feature":""})
        fig_c.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color=FONT,
                            coloraxis_showscale=False, height=380, margin=dict(t=10,b=10))
        st.plotly_chart(fig_c, use_container_width=True)

# ═══════ TAB 7 — BATCH PREDICTION ════════════════════════════
with tab7:
    st.markdown("### 📂 Batch CSV Prediction")
    st.caption("Upload a CSV of transactions and get fraud predictions for all rows instantly.")
    st.divider()

    # ── Template download ─────────────────────────────────────
    st.markdown('<div class="section-title">Step 1 — Download Template</div>', unsafe_allow_html=True)
    template = pd.DataFrame({
        "Amount":          [1500.0, 4800.0, 320.0],
        "TransactionType": ["purchase", "refund", "purchase"],
        "Location":        ["New York", "Houston", "Chicago"],
        "MerchantID":      [101, 850, 300],
        "DayOfWeek":       ["Mon", "Sat", "Wed"],
        "Month":           ["Jan", "Dec", "Mar"],
    })
    csv_template = template.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV Template", csv_template,
        file_name="batch_template.csv", mime="text/csv"
    )
    st.dataframe(template, use_container_width=True)

    # ── Upload & predict ──────────────────────────────────────
    st.markdown('<div class="section-title">Step 2 — Upload Your CSV</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV (same columns as template)", type=["csv"])

    if uploaded:
        try:
            udf = pd.read_csv(uploaded)
            required = ["Amount","TransactionType","Location","MerchantID","DayOfWeek","Month"]
            missing  = [c for c in required if c not in udf.columns]

            if missing:
                st.error(f"❌ Missing columns: {missing}  |  Required: {required}")
            else:
                st.success(f"✅ Loaded **{len(udf):,}** transactions — running predictions...")

                rows, probs, preds, iso_rs = [], [], [], []
                for _, r in udf.iterrows():
                    try:
                        row = build_row(
                            float(r["Amount"]), str(r["TransactionType"]),
                            str(r["Location"]),  int(r["MerchantID"]),
                            str(r["DayOfWeek"]),  str(r["Month"])
                        )
                        prob = rf.predict_proba(row)[0][1]
                        pred = rf.predict(row)[0]
                        iso_r= iso.predict(row)[0]
                        probs.append(round(prob, 4))
                        preds.append(pred)
                        iso_rs.append(1 if iso_r == -1 else 0)
                    except Exception:
                        probs.append(None); preds.append(None); iso_rs.append(None)

                udf["fraud_probability"] = probs
                udf["rf_prediction"]     = preds
                udf["iso_anomaly"]       = iso_rs

                # ── Summary KPIs ──────────────────────────────
                valid = udf.dropna(subset=["rf_prediction"])
                total_b  = len(valid)
                fraud_b  = int(valid["rf_prediction"].sum())
                safe_b   = total_b - fraud_b
                both_b   = int(((valid["rf_prediction"]==1) & (valid["iso_anomaly"]==1)).sum())

                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Total Processed", f"{total_b:,}")
                b2.metric("🚨 Fraud Flagged", f"{fraud_b:,}")
                b3.metric("✅ Legitimate",    f"{safe_b:,}")
                b4.metric("🔗 Both Models",   f"{both_b:,}")

                # ── Results table ─────────────────────────────
                st.markdown('<div class="section-title">Prediction Results</div>', unsafe_allow_html=True)
                display_df = udf.copy()
                display_df["rf_prediction"] = display_df["rf_prediction"].map(
                    {0:"✅ Legit", 1:"🚨 Fraud", None:"Error"})
                display_df["iso_anomaly"]   = display_df["iso_anomaly"].map(
                    {0:"Normal", 1:"⚠️ Anomaly", None:"Error"})
                display_df["fraud_probability"] = display_df["fraud_probability"].apply(
                    lambda x: f"{x:.2%}" if x is not None else "—")
                st.dataframe(display_df.sort_values("fraud_probability", ascending=False),
                             use_container_width=True, height=400)

                # ── Fraud distribution chart ──────────────────
                if fraud_b > 0:
                    st.markdown('<div class="section-title">Fraud Distribution in Upload</div>',
                                unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        pie2 = pd.DataFrame({"Status":["Legitimate","Fraud"],"Count":[safe_b,fraud_b]})
                        fig_p = px.pie(pie2, values="Count", names="Status", hole=0.5,
                                       color="Status", color_discrete_map=COLORS)
                        st.plotly_chart(base_layout(fig_p, 280), use_container_width=True)
                    with c2:
                        fig_h = px.histogram(udf.dropna(), x="fraud_probability", nbins=20,
                                             color_discrete_sequence=["#4d9fff"],
                                             labels={"fraud_probability":"Fraud Score"})
                        st.plotly_chart(base_layout(fig_h, 280), use_container_width=True)

                # ── Download results ──────────────────────────
                st.markdown('<div class="section-title">Step 3 — Download Results</div>',
                            unsafe_allow_html=True)
                result_csv = udf.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Predictions CSV", result_csv,
                    file_name="batch_predictions.csv", mime="text/csv",
                    type="primary"
                )

        except Exception as e:
            st.error(f"❌ Error processing file: {e}")

# ── Footer ────────────────────────────────────────────────────
st.divider()
st.caption("🔍 Credit Card Fraud Detection | Random Forest + Isolation Forest | Streamlit + Plotly")
