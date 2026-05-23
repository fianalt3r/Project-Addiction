import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

# ── Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Addiction Analysis",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL   = os.environ.get("API_URL", "https://project-addiction.onrender.com")
ROOT      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(ROOT, "data", "students_social_media_addiction.csv")

ADDICTION_COLORS = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"}

# Inject Premium Design Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}
div[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 16px 20px;
    box-shadow: 0 8px 32px 0 rgba(0,0,0,0.2);
    border: 1px solid rgba(255, 255, 255, 0.05);
}
.custom-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.05);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px 0 rgba(0,0,0,0.2);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.custom-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 12px 40px 0 rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# Helper function to style Plotly charts consistently and ensure text contrast
def style_chart(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Outfit, sans-serif", size=12, color="#e2e8f0"),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig

# Shared data loader
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    if "Student_ID" in df.columns:
        df = df.drop(columns=["Student_ID"])
    df["Addiction_Level"] = pd.cut(
        df["Addicted_Score"], bins=[0, 4, 7, 10],
        labels=["Low", "Moderate", "High"],
    ).astype(str)
    return df

# Sidebar navigation
st.sidebar.title("📱 Social Media Addiction")
st.sidebar.caption("Data Study of 705 Students")
st.sidebar.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.2); margin: 10px 0;">', unsafe_allow_html=True)

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "📺 Screen Time Analysis",
    "📚 Productivity and Concentration",
    "😊 Mood and Screen Time",
    "🤖 Model Performance",
    "🔮 Predict My Risk",
    "🧠 SHAP Insights and Digital Ethics",
])

st.sidebar.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.2); margin: 10px 0;">', unsafe_allow_html=True)
st.sidebar.caption("Active features: 11 categories")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════
def page_overview():
    st.title("🏠 Overview")
    st.markdown("High level snapshot of the student study data and general addiction risk levels.")

    df = load_data()

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Students", f"{len(df):,}")
    k2.metric("Avg Daily Usage", f"{df['Avg_Daily_Usage_Hours'].mean():.1f} hours")
    k3.metric("Avg Mood Score", f"{df['Mental_Health_Score'].mean():.1f} out of 10")
    k4.metric("Avg Sleep Time", f"{df['Sleep_Hours_Per_Night'].mean():.1f} hours")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)

    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.markdown("### Addiction Risk Distribution")
        dist = df["Addiction_Level"].value_counts().reindex(["Low", "Moderate", "High"])
        fig  = px.bar(x=dist.index, y=dist.values, color=dist.index,
                      color_discrete_map=ADDICTION_COLORS, text=dist.values,
                      labels={"x": "Risk Level", "y": "Students"})
        fig.update_traces(textposition="outside")
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Insight**: This bar chart shows how many students fall into each addiction category. We see that a significant portion of students are at moderate or high risk, indicating that excessive screen time is a widespread concern.")

    with c2:
        st.markdown("### Percentage Split")
        fig = px.pie(values=dist.values, names=dist.index, color=dist.index,
                     color_discrete_map=ADDICTION_COLORS, hole=0.45)
        fig.update_traces(textinfo="percent+label")
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Insight**: This donut chart details the proportional breakdown. More than half of the surveyed students show moderate to high addiction signs.")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("### Gender Distribution")
        g = df["Gender"].value_counts()
        fig = px.pie(values=g.values, names=g.index, hole=0.4,
                     color_discrete_sequence=["#6C63FF", "#FF6584"])
        fig.update_traces(textinfo="percent+label")
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Insight**: Gender representation in our study group, showing a balanced participation of female and male students.")

    with c4:
        st.markdown("### Academic Level Breakdown")
        a = df["Academic_Level"].value_counts()
        fig = px.bar(x=a.index, y=a.values, color=a.index,
                     color_discrete_sequence=px.colors.qualitative.Set2,
                     text=a.values, labels={"x": "Level", "y": "Count"})
        fig.update_traces(textposition="outside")
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Insight**: A count of students across high school, undergraduate, and graduate levels. Undergraduates form the largest segment.")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    st.markdown("### Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
    with st.expander("📊 Full Summary Statistics"):
        st.dataframe(df.describe().round(2), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 2: SCREEN TIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
def page_screen_time():
    st.title("📺 Screen Time Analysis")
    st.markdown("Details on how much time students spend on social media and what influences their habits.")

    df = load_data()

    st.sidebar.markdown("### Filters")
    genders   = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))
    academs   = st.sidebar.multiselect("Academic Level", df["Academic_Level"].unique(), default=list(df["Academic_Level"].unique()))
    platforms = st.sidebar.multiselect("Platform", df["Most_Used_Platform"].unique(), default=list(df["Most_Used_Platform"].unique()))
    
    fdf = df[df["Gender"].isin(genders) & df["Academic_Level"].isin(academs) & df["Most_Used_Platform"].isin(platforms)]

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return
        
    st.caption(f"Showing details for {len(fdf)} students.")
    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Daily Usage Hours Distribution")
        fig = px.histogram(fdf, x="Avg_Daily_Usage_Hours", nbins=20,
                           color_discrete_sequence=["#6C63FF"],
                           labels={"Avg_Daily_Usage_Hours": "Hours per day"})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: This histogram displays the spread of daily screen time. A high density of students spend between 4 to 7 hours online every day.")

    with c2:
        st.markdown("### Daily Usage by Addiction Risk")
        fig = px.box(fdf, x="Addiction_Level", y="Avg_Daily_Usage_Hours",
                     color="Addiction_Level", color_discrete_map=ADDICTION_COLORS,
                     category_orders={"Addiction_Level": ["Low", "Moderate", "High"]},
                     labels={"Avg_Daily_Usage_Hours": "Hours per day"})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: This box plot highlights the clear correlation between total hours spent and addiction labels. High risk students consistently exhibit much higher usage ranges.")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### Average Daily Hours by Platform")
        p = fdf.groupby("Most_Used_Platform")["Avg_Daily_Usage_Hours"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(p, x="Avg_Daily_Usage_Hours", y="Most_Used_Platform",
                     orientation="h", color="Avg_Daily_Usage_Hours",
                     color_continuous_scale="Purples", text=p["Avg_Daily_Usage_Hours"].round(1))
        fig.update_traces(textposition="outside")
        style_chart(fig)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: This chart compares how different social platforms hold user attention. Video platforms like TikTok and image sharing apps like Instagram generally rank highest.")

    with c4:
        st.markdown("### Risk Mix per Platform")
        pr = fdf.groupby(["Most_Used_Platform", "Addiction_Level"]).size().reset_index(name="count")
        fig = px.bar(pr, x="Most_Used_Platform", y="count", color="Addiction_Level",
                     color_discrete_map=ADDICTION_COLORS, barmode="stack",
                     labels={"count": "Students"})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Displays how risk levels are distributed across primary platforms. Certain platforms show a higher density of high risk students.")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    c5, c6 = st.columns(2)

    with c5:
        st.markdown("### Daily Usage by Academic Level")
        a = fdf.groupby("Academic_Level")["Avg_Daily_Usage_Hours"].mean().sort_values(ascending=False).reset_index()
        fig = px.bar(a, x="Academic_Level", y="Avg_Daily_Usage_Hours",
                     color="Academic_Level", color_discrete_sequence=px.colors.qualitative.Set2,
                     text=a["Avg_Daily_Usage_Hours"].round(1))
        fig.update_traces(textposition="outside")
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Displays the average hours spent online across high school, undergraduate, and graduate students.")

    with c6:
        st.markdown("### Top 10 Countries by Average Screen Time")
        co = fdf.groupby("Country")["Avg_Daily_Usage_Hours"].mean().sort_values(ascending=False).head(10).reset_index()
        fig = px.bar(co, x="Avg_Daily_Usage_Hours", y="Country", orientation="h",
                     color="Avg_Daily_Usage_Hours", color_continuous_scale="Blues",
                     text=co["Avg_Daily_Usage_Hours"].round(1))
        fig.update_traces(textposition="outside")
        style_chart(fig)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Highlights the geographical locations showing the highest average daily screen time in our dataset.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 3: PRODUCTIVITY AND CONCENTRATION
# ══════════════════════════════════════════════════════════════════════════
def page_productivity():
    st.title("📚 Productivity and Concentration")
    st.markdown("Analyzing how digital habits impact studies, sleep quality, and lifestyle harmony.")

    df = load_data()

    st.sidebar.markdown("### Filters")
    genders = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))
    academs = st.sidebar.multiselect("Academic Level", df["Academic_Level"].unique(), default=list(df["Academic_Level"].unique()))
    fdf = df[df["Gender"].isin(genders) & df["Academic_Level"].isin(academs)]

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return

    pct   = (fdf["Affects_Academic_Performance"] == "Yes").mean() * 100
    s_hi  = fdf[fdf["Addiction_Level"] == "High"]["Sleep_Hours_Per_Night"].mean()
    s_lo  = fdf[fdf["Addiction_Level"] == "Low"]["Sleep_Hours_Per_Night"].mean()

    k1, k2, k3 = st.columns(3)
    k1.metric("Studies Affected", f"{pct:.1f}%")
    k2.metric("Avg Sleep (High Risk)", f"{s_hi:.1f} hours")
    k3.metric("Avg Sleep (Low Risk)", f"{s_lo:.1f} hours")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Academic Impact by Risk Group")
        ap = fdf.groupby(["Addiction_Level", "Affects_Academic_Performance"]).size().reset_index(name="count")
        tot = ap.groupby("Addiction_Level")["count"].transform("sum")
        ap["pct"] = (ap["count"] / tot * 100).round(1)
        ap_yes = ap[ap["Affects_Academic_Performance"] == "Yes"]
        fig = px.bar(ap_yes, x="Addiction_Level", y="pct", color="Addiction_Level",
                     color_discrete_map=ADDICTION_COLORS, text="pct",
                     category_orders={"Addiction_Level": ["Low", "Moderate", "High"]},
                     labels={"pct": "Percentage Impacted"})
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        style_chart(fig)
        fig.update_layout(yaxis_range=[0, 110])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: This chart shows the percentage of students in each risk group who report that social media hurts their studies. The impact rises sharply with higher risk.")

    with c2:
        st.markdown("### Daily Usage and Academic Impact")
        fig = px.box(fdf, x="Affects_Academic_Performance", y="Avg_Daily_Usage_Hours",
                     color="Affects_Academic_Performance",
                     color_discrete_map={"Yes": "#e74c3c", "No": "#2ecc71"},
                     labels={"Avg_Daily_Usage_Hours": "Daily Usage (hours)"})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Illustrates that students who acknowledge a drop in studies spend much longer hours online compared to those who do not.")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### Sleep Duration by Addiction Risk")
        fig = px.box(fdf, x="Addiction_Level", y="Sleep_Hours_Per_Night",
                     color="Addiction_Level", color_discrete_map=ADDICTION_COLORS,
                     category_orders={"Addiction_Level": ["Low", "Moderate", "High"]},
                     labels={"Sleep_Hours_Per_Night": "Sleep hours per night"})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: A visual comparison of nightly sleep hours across risk groups. High risk students get far less sleep, which can harm their concentration.")

    with c4:
        st.markdown("### Screen Time vs Sleep Duration")
        fig = px.scatter(fdf, x="Avg_Daily_Usage_Hours", y="Sleep_Hours_Per_Night",
                         color="Addiction_Level", color_discrete_map=ADDICTION_COLORS,
                         trendline="ols", opacity=0.6,
                         labels={"Avg_Daily_Usage_Hours": "Daily Usage (hours)", "Sleep_Hours_Per_Night": "Sleep (hours)"})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: The downward trendline demonstrates that as daily screen time increases, sleep duration steadily drops.")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    c5, c6 = st.columns(2)

    with c5:
        st.markdown("### Arguments and Conflicts by Risk Level")
        fig = px.box(fdf, x="Addiction_Level", y="Conflicts_Over_Social_Media",
                     color="Addiction_Level", color_discrete_map=ADDICTION_COLORS,
                     category_orders={"Addiction_Level": ["Low", "Moderate", "High"]},
                     labels={"Conflicts_Over_Social_Media": "Conflict Score (0 to 5)"})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: High risk students experience more arguments and conflicts with family or friends due to their screen usage.")

    with c6:
        st.markdown("### Average Sleep by Academic Level")
        a = fdf.groupby("Academic_Level")["Sleep_Hours_Per_Night"].mean().sort_values().reset_index()
        fig = px.bar(a, x="Academic_Level", y="Sleep_Hours_Per_Night",
                     color="Academic_Level", color_discrete_sequence=px.colors.qualitative.Set2,
                     text=a["Sleep_Hours_Per_Night"].round(1))
        fig.update_traces(textposition="outside")
        style_chart(fig)
        fig.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Average nightly sleep duration across academic groups in the dataset.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 4: MOOD AND SCREEN TIME
# ══════════════════════════════════════════════════════════════════════════
def page_mood():
    st.title("😊 Mood and Screen Time")
    st.markdown("Investigating the relationship between emotional wellbeing, screen time, and lifestyle factors.")

    df = load_data()

    st.sidebar.markdown("### Filters")
    genders  = st.sidebar.multiselect("Gender", df["Gender"].unique(), default=list(df["Gender"].unique()))
    rel_stat = st.sidebar.multiselect("Relationship Status", df["Relationship_Status"].unique(), default=list(df["Relationship_Status"].unique()))
    fdf = df[df["Gender"].isin(genders) & df["Relationship_Status"].isin(rel_stat)]

    if fdf.empty:
        st.warning("No data matches the selected filters.")
        return

    corr = fdf["Avg_Daily_Usage_Hours"].corr(fdf["Mental_Health_Score"])
    mh_hi = fdf[fdf["Addiction_Level"] == "High"]["Mental_Health_Score"].mean()
    mh_lo = fdf[fdf["Addiction_Level"] == "Low"]["Mental_Health_Score"].mean()

    k1, k2, k3 = st.columns(3)
    k1.metric("Avg Mood (High Risk)", f"{mh_hi:.1f} / 10")
    k2.metric("Avg Mood (Low Risk)", f"{mh_lo:.1f} / 10")
    k3.metric("Correlation Coefficient", f"{corr:.3f}")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("### Screen Time vs Mood Score")
        fig = px.scatter(fdf, x="Avg_Daily_Usage_Hours", y="Mental_Health_Score",
                         color="Addiction_Level", color_discrete_map=ADDICTION_COLORS,
                         trendline="ols", opacity=0.65,
                         labels={"Avg_Daily_Usage_Hours": "Daily Usage (hours)", "Mental_Health_Score": "Mood Score (1 to 10)"})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Displays individual student data points. The downward slope reveals that higher screen time correlates with lower self reported mood scores.")

    with c2:
        st.markdown("### Mood Score Distribution by Risk Level")
        fig = px.box(fdf, x="Addiction_Level", y="Mental_Health_Score",
                     color="Addiction_Level", color_discrete_map=ADDICTION_COLORS,
                     category_orders={"Addiction_Level": ["Low", "Moderate", "High"]},
                     points="all", labels={"Mental_Health_Score": "Mood Score"})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Compares mood spans across risk groups. Low risk students report higher and more stable mood scores.")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("### Average Mood Score by Platform")
        p = fdf.groupby("Most_Used_Platform")["Mental_Health_Score"].mean().sort_values().reset_index()
        fig = px.bar(p, x="Mental_Health_Score", y="Most_Used_Platform", orientation="h",
                     color="Mental_Health_Score", color_continuous_scale="RdYlGn",
                     text=p["Mental_Health_Score"].round(1))
        fig.update_traces(textposition="outside")
        style_chart(fig)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Average mood score grouped by the platform the student uses the most.")

    with c4:
        st.markdown("### Sleep Hours vs Mood Score")
        fig = px.scatter(fdf, x="Sleep_Hours_Per_Night", y="Mental_Health_Score",
                         color="Addiction_Level", color_discrete_map=ADDICTION_COLORS,
                         trendline="ols", opacity=0.65,
                         labels={"Sleep_Hours_Per_Night": "Sleep (hours)", "Mental_Health_Score": "Mood Score"})
        style_chart(fig)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Points upward, showing that more sleep is strongly connected to better mood scores.")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    c5, c6 = st.columns(2)

    with c5:
        st.markdown("### Mood by Relationship Status")
        r = fdf.groupby("Relationship_Status")["Mental_Health_Score"].mean().reset_index()
        fig = px.bar(r, x="Relationship_Status", y="Mental_Health_Score",
                     color="Relationship_Status", color_discrete_sequence=px.colors.qualitative.Pastel,
                     text=r["Mental_Health_Score"].round(1))
        fig.update_traces(textposition="outside")
        style_chart(fig)
        fig.update_layout(yaxis_range=[0, 10])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Compares average mood scores across different relationship status categories.")

    with c6:
        st.markdown("### Habit Correlation Heatmap")
        cols = ["Avg_Daily_Usage_Hours", "Sleep_Hours_Per_Night", "Mental_Health_Score",
                "Conflicts_Over_Social_Media", "Addicted_Score"]
        corr_mat = fdf[cols].corr().round(2)
        fig = px.imshow(corr_mat, text_auto=True, color_continuous_scale="RdBu_r",
                        zmin=-1, zmax=1, aspect="auto")
        style_chart(fig)
        fig.update_layout(height=360)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Compares variables side by side. Red blocks show positive links (like usage hours and conflicts), while blue blocks show negative links (like usage hours and sleep).")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 5: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════
def page_model_performance():
    st.title("🤖 Model Performance")
    st.markdown("Evaluating classifier scores with proper cross validation and hyperparameter tuning.")

    try:
        r = requests.get(f"{API_URL}/model-metrics", timeout=5)
        r.raise_for_status()
        m = r.json()
    except Exception as e:
        st.error(f"Cannot connect to the backend. Please verify that the FastAPI backend is running.\n\n{e}")
        return

    # Visualizing metrics side by side
    st.markdown("### Model Comparison Summary")
    
    tab1, tab2, tab3 = st.tabs(["🌳 Random Forest", "📈 Logistic Regression", "🚀 XGBoost"])
    classes = m["classes"]

    with tab1:
        st.markdown("#### Performance Metrics")
        st.markdown(f"**Cross Validation Accuracy**: {m['rf_cv_score']*100:.2f}%")
        st.markdown(f"**Holdout Test Accuracy**: {m['rf_accuracy']*100:.2f}%")
        st.markdown(f"**Weighted F1 Score**: {m['rf_f1']:.3f}")
        st.markdown("**Best Hyperparameters**:")
        for k, v in m["rf_best_params"].items():
            st.markdown(f"* {k}: `{v}`")
        
        st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
        st.markdown("#### Confusion Matrix")
        cm = np.array(m["rf_confusion_matrix"])
        fig = px.imshow(cm, x=classes, y=classes, text_auto=True,
                        color_continuous_scale="Blues",
                        labels={"x": "Predicted", "y": "Actual"}, aspect="auto")
        style_chart(fig)
        fig.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: The diagonal squares show correct predictions. Random Forest makes very few errors across all risk levels.")

    with tab2:
        st.markdown("#### Performance Metrics")
        st.markdown(f"**Cross Validation Accuracy**: {m['lr_cv_score']*100:.2f}%")
        st.markdown(f"**Holdout Test Accuracy**: {m['lr_accuracy']*100:.2f}%")
        st.markdown(f"**Weighted F1 Score**: {m['lr_f1']:.3f}")
        st.markdown("**Best Hyperparameters**:")
        for k, v in m["lr_best_params"].items():
            st.markdown(f"* {k}: `{v}`")
            
        st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
        st.markdown("#### Confusion Matrix")
        cm_lr = np.array(m["lr_confusion_matrix"])
        fig = px.imshow(cm_lr, x=classes, y=classes, text_auto=True,
                        color_continuous_scale="Oranges",
                        labels={"x": "Predicted", "y": "Actual"}, aspect="auto")
        style_chart(fig)
        fig.update_layout(height=360, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Shows prediction splits for Logistic Regression. It exhibits slightly more misclassifications between Low and Moderate risk.")

    with tab3:
        st.markdown("#### Performance Metrics")
        st.markdown(f"**Cross Validation Accuracy**: {m['xgb_cv_score']*100:.2f}%")
        st.markdown(f"**Holdout Test Accuracy**: {m['xgb_accuracy']*100:.2f}%")
        st.markdown(f"**Weighted F1 Score**: {m['xgb_f1']:.3f}")
        st.markdown("**Best Hyperparameters**:")
        for k, v in m["xgb_best_params"].items():
            st.markdown(f"* {k}: `{v}`")
            
        st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
        st.markdown("#### Confusion Matrix")
        cm_xgb = np.array(m["xgb_confusion_matrix"])
        fig = px.imshow(cm_xgb, x=classes, y=classes, text_auto=True,
                        color_continuous_scale="Greens",
                        labels={"x": "Predicted", "y": "Actual"}, aspect="auto")
        style_chart(fig)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("* **Explanation**: Displays prediction splits for XGBoost. It effectively handles complex non-linear patterns, performing on par with or better than Random Forest.")

    st.markdown("### Performance Justification")
    st.markdown(
        """
        The **Random Forest** and **XGBoost** classifiers achieve slightly higher cross validation and test accuracy compared to **Logistic Regression**. 
        This performance gap exists because tree-based models (like Random Forest and XGBoost) can automatically capture complex, non linear interactions 
        between features (like how the combination of high usage hours and very low sleep together amplifies risk). 
        
        **XGBoost**, an advanced gradient boosting algorithm, typically provides the highest predictive power by sequentially correcting errors from previous trees.
        However, **Logistic Regression** serves as a strong, highly explainable model that performs close to the tree-based models, making it an excellent baseline for comparison.
        """
    )

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    
    st.markdown("### Accuracy and F1 Score Comparison")
    fig = go.Figure()
    fig.add_bar(name="Cross Val Accuracy", x=["Random Forest", "Logistic Regression", "XGBoost"],
                y=[m["rf_cv_score"], m["lr_cv_score"], m["xgb_cv_score"]],
                marker_color="#43B89C", text=[round(m["rf_cv_score"], 3), round(m["lr_cv_score"], 3), round(m["xgb_cv_score"], 3)],
                textposition="outside")
    fig.add_bar(name="Test Accuracy", x=["Random Forest", "Logistic Regression", "XGBoost"],
                y=[m["rf_accuracy"], m["lr_accuracy"], m["xgb_accuracy"]],
                marker_color="#6C63FF", text=[round(m["rf_accuracy"], 3), round(m["lr_accuracy"], 3), round(m["xgb_accuracy"], 3)],
                textposition="outside")
    fig.add_bar(name="F1 Score", x=["Random Forest", "Logistic Regression", "XGBoost"],
                y=[m["rf_f1"], m["lr_f1"], m["xgb_f1"]],
                marker_color="#FF6584", text=[round(m["rf_f1"], 3), round(m["lr_f1"], 3), round(m["xgb_f1"], 3)],
                textposition="outside")
    style_chart(fig)
    fig.update_layout(barmode="group", yaxis_range=[0, 1.2])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("* **Explanation**: Displays validation and test statistics. Both models show high generalization scores, with Random Forest taking a slight lead.")

    with st.expander("📋 Full Random Forest Classification Report"):
        report = m["rf_classification_report"]
        rows   = {k: v for k, v in report.items() if isinstance(v, dict)}
        st.dataframe(pd.DataFrame(rows).T.round(3), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# PAGE 6: PREDICT MY RISK
# ══════════════════════════════════════════════════════════════════════════
def page_predict():
    st.title("🔮 Predict My Risk")
    st.markdown("Input your personal habits to calculate your estimated social media addiction risk.")

    df = load_data()
    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 10px 0;">', unsafe_allow_html=True)

    # Model Switch Feature
    selected_model = st.selectbox("Choose Machine Learning Model to use", ["Random Forest", "Logistic Regression", "XGBoost"])
    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 10px 0;">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📺 Screen Time Habits**")
        daily_usage = st.slider("Daily Usage Hours", 0.0, 24.0, 5.0, 0.5)
        platform    = st.selectbox("Most Used Platform", sorted(df["Most_Used_Platform"].unique()))

    with c2:
        st.markdown("**📚 Study and Lifestyle Impact**")
        affects = st.selectbox("Does usage affect your studies?", ["Yes", "No"])
        sleep   = st.slider("Sleep Hours Per Night", 3.0, 10.0, 7.0, 0.5)
        conflicts = st.slider("Conflicts Over Social Media (0 to 5)", 0, 5, 1)

    with c3:
        st.markdown("**😊 Emotional Wellbeing**")
        mental = st.slider("Mental Health Score (1 to 10)", 1, 10, 6)
        rel    = st.selectbox("Relationship Status", sorted(df["Relationship_Status"].unique()))

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 10px 0;">', unsafe_allow_html=True)
    st.markdown("**👤 Demographics**")
    d1, d2, d3, d4 = st.columns(4)
    age      = d1.number_input("Age", 15, 35, 20)
    gender   = d2.selectbox("Gender", sorted(df["Gender"].unique()))
    academic = d3.selectbox("Academic Level", sorted(df["Academic_Level"].unique()))
    country  = d4.selectbox("Country", sorted(df["Country"].unique()))

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 20px 0;">', unsafe_allow_html=True)

    if st.button("Calculate My Risk", type="primary", use_container_width=True):
        payload = {
            "Age": age, "Gender": gender, "Academic_Level": academic,
            "Country": country, "Avg_Daily_Usage_Hours": daily_usage,
            "Most_Used_Platform": platform, "Affects_Academic_Performance": affects,
            "Sleep_Hours_Per_Night": sleep, "Mental_Health_Score": float(mental),
            "Conflicts_Over_Social_Media": conflicts, "Relationship_Status": rel,
            "Model_Name": selected_model
        }
        with st.spinner("Calculating risk values..."):
            try:
                r = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                r.raise_for_status()
                result = r.json()
            except Exception as e:
                st.error(f"Cannot reach the prediction API.\n\n{e}")
                return

        pred  = result["prediction"]
        p_low = result["probability_low"]
        p_mod = result["probability_moderate"]
        p_hi  = result["probability_high"]

        st.markdown("## 📊 Prediction Results")
        icons  = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
        colors = {"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"}

        r1, r2 = st.columns([1, 1.8])
        with r1:
            st.markdown("### Risk Level")
            st.markdown(f"""
            <div style='background:{colors[pred]};color:white;padding:14px 28px;
            border-radius:30px;font-size:1.5rem;font-weight:700;display:inline-block;'>
            {icons[pred]} {pred} Risk</div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Prediction Confidence")
            probs = pd.DataFrame({"Level": ["Low", "Moderate", "High"],
                                   "Probability": [p_low, p_mod, p_hi]})
            fig = px.bar(probs, x="Level", y="Probability", color="Level",
                         color_discrete_map={"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"},
                         text=probs["Probability"].map(lambda v: f"{v*100:.1f}%"),
                         range_y=[0, 1])
            fig.update_traces(textposition="outside")
            style_chart(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"* **Explanation**: The model's confidence across low, moderate, and high classes. The model used for this calculation was {selected_model}.")

        with r2:
            st.markdown(f"### Feature Importance (SHAP values for {pred} class)")
            shap_df = pd.DataFrame({"Feature": result["feature_names"], "SHAP": result["shap_values"]})
            shap_df = shap_df.sort_values("SHAP", key=abs, ascending=True)
            shap_df["Direction"] = shap_df["SHAP"].apply(lambda v: "Increases Risk" if v > 0 else "Decreases Risk")
            fig = px.bar(shap_df, x="SHAP", y="Feature", orientation="h", color="Direction",
                         color_discrete_map={"Increases Risk": "#e74c3c", "Decreases Risk": "#2ecc71"},
                         text=shap_df["SHAP"].map(lambda v: f"{v:+.3f}"))
            fig.add_vline(x=0, line_width=1.5, line_color="black")
            fig.update_traces(textposition="outside")
            style_chart(fig)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("* **Explanation**: This chart displays which of your inputs contributed most to the prediction. Red bars pushed the risk rating up, while green bars pulled it down.")

        st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 20px 0;">', unsafe_allow_html=True)
        st.markdown("### Guidance")
        if pred == "High":
            st.error("High risk detected. Consider capping daily screen time to under 3 hours, setting phone free periods before sleep, and swapping scrolling with offline activities.")
        elif pred == "Moderate":
            st.warning("Moderate risk detected. Keep track of daily usage, turn off non important notifications, and avoid social media usage in bed.")
        else:
            st.success("Low risk detected. Keep up the good habits, maintain regular sleep times, and share tips on healthy device habits with friends.")


# ══════════════════════════════════════════════════════════════════════════
# PAGE 7: SHAP AND ETHICS
# ══════════════════════════════════════════════════════════════════════════
def page_shap_ethics():
    st.title("🧠 SHAP Insights and Digital Ethics")
    st.markdown("Understanding which features influence predictions overall, and exploring simulated habit changes.")

    df = load_data()

    # Explainer model choice
    selected_model = st.selectbox("Choose model to view Global Feature Importance", ["Random Forest", "Logistic Regression", "XGBoost"])

    try:
        r = requests.get(f"{API_URL}/feature-importance?model_name={selected_model}", timeout=15)
        r.raise_for_status()
        shap_data = r.json()
        shap_ok   = True
    except Exception as e:
        shap_ok = False
        st.warning(f"Could not reach backend for SHAP metrics.\n\n{e}")

    if shap_ok:
        st.markdown(f"### Global Feature Importance for {selected_model}")
        theme_map = {
            "Avg_Daily_Usage_Hours": "📺 Screen Time", "Most_Used_Platform": "📺 Screen Time",
            "Sleep_Hours_Per_Night": "📚 Studies and Lifestyle", "Affects_Academic_Performance": "📚 Studies and Lifestyle",
            "Conflicts_Over_Social_Media": "📚 Studies and Lifestyle",
            "Mental_Health_Score": "😊 Mood", "Relationship_Status": "😊 Mood",
            "Age": "👤 Demographics", "Gender": "👤 Demographics",
            "Academic_Level": "👤 Demographics", "Country": "👤 Demographics",
        }
        color_map = {"📺 Screen Time": "#6C63FF", "📚 Studies and Lifestyle": "#f39c12",
                     "😊 Mood": "#2ecc71", "👤 Demographics": "#95a5a6"}

        shap_df = pd.DataFrame({"Feature": shap_data["feature_names"],
                                 "Mean SHAP Value": shap_data["mean_shap_values"]})
        shap_df = shap_df.sort_values("Mean SHAP Value", ascending=True)
        shap_df["Theme"] = shap_df["Feature"].map(theme_map).fillna("Other")

        fig = px.bar(shap_df, x="Mean SHAP Value", y="Feature", orientation="h",
                     color="Theme", color_discrete_map=color_map,
                     text=shap_df["Mean SHAP Value"].map(lambda v: f"{v:.4f}"))
        fig.update_traces(textposition="outside")
        style_chart(fig)
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"* **Explanation**: Shows which features have the strongest impact on the {selected_model} model. Longer bars mean the feature has more overall weight in predictions.")

        top3 = shap_df.sort_values("Mean SHAP Value", ascending=False).head(3)
        st.info(f"Top 3 drivers: {top3.iloc[0]['Feature']}, {top3.iloc[1]['Feature']}, {top3.iloc[2]['Feature']}")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    st.markdown("### Simulated Habit Changes: Reducing Screen Time")
    st.markdown("Move the slider to see how the overall risk profile across all 705 students would shift if everyone capped their daily screen time.")
    
    what_if = st.slider("Imagine everyone caps daily usage to a maximum of:", 0.0, 24.0, 6.0, 0.5)
    sim = df.copy()
    sim["Simulated_Level"] = pd.cut(
        sim["Avg_Daily_Usage_Hours"].clip(upper=what_if),
        bins=[-1, 4, 7, 25], labels=["Low", "Moderate", "High"]
    )
    sim_dist = sim["Simulated_Level"].value_counts().reindex(["Low", "Moderate", "High"])
    fig = px.bar(x=sim_dist.index, y=sim_dist.values, color=sim_dist.index,
                 color_discrete_map={"Low": "#2ecc71", "Moderate": "#f39c12", "High": "#e74c3c"},
                 text=sim_dist.values, labels={"x": "Risk Level", "y": "Students"})
    fig.update_traces(textposition="outside")
    style_chart(fig)
    fig.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(f"* **Explanation**: By setting a maximum screen time cap of {what_if} hours, the overall risk across the student population decreases. This illustrates the positive impact of setting personal boundaries.")

    st.markdown('<hr style="border: 0; height: 1px; background: rgba(128,128,128,0.15); margin: 24px 0;">', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        st.markdown(
            """
            #### Digital Discipline and Awareness
            These predictions should serve as a starting point for personal reflection.
            * Predictions are estimates and not definitive verdicts.
            * The model assesses general correlation patterns and does not measure individual worth.
            * Take results as a helpful checkpoint to evaluate screen usage.
            """
        )
    with e2:
        st.markdown(
            """
            #### Recommended Guidelines
            * Set app timers to manage passive scrolling.
            * Use grayscale screen options to make apps less stimulating.
            * Keep devices out of the bedroom for better sleep.
            * Track weekly screen hours to build awareness.
            """
        )

    with st.expander("📋 Model Specs and Log Info"):
        st.markdown(
            """
            * **Study Group**: 705 students, 11 active features.
            * **Label Mapping**: Low (0 to 4), Moderate (5 to 7), High (8 to 10) addiction scores.
            * **Classifiers**: Random Forest and Logistic Regression.
            * **Explainability Engine**: SHAP values (Tree and Linear explainers).
            * **Logging**: Query data is logged anonymously to predictions log.
            """
        )


# ══════════════════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════════════════
if   page == "🏠 Overview":                          page_overview()
elif page == "📺 Screen Time Analysis":              page_screen_time()
elif page == "📚 Productivity and Concentration":     page_productivity()
elif page == "😊 Mood and Screen Time":              page_mood()
elif page == "🤖 Model Performance":                  page_model_performance()
elif page == "🔮 Predict My Risk":                    page_predict()
elif page == "🧠 SHAP Insights and Digital Ethics":   page_shap_ethics()
