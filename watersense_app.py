import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import folium
from streamlit_folium import st_folium
from reportlab.pdfgen import canvas
from io import BytesIO
from datetime import datetime
import os
import json


# ------------------------------------------------------
# BASIC PAGE CONFIG
# ------------------------------------------------------
st.set_page_config(
    page_title="WaterSense – Smart Groundwater Risk Predictor",
    page_icon="💧",
    layout="wide",
)

# ------------------------------------------------------
# DATA + MODEL LOADERS
# ------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("district_groundwater.csv")
    df.columns = df.columns.str.strip()

    # Build the same modelling dataframe as in training
    df_model = df[[
        "Name of District",
        "Recharge from rainfall During Monsoon Season",
        "Recharge from other sources During Monsoon Season",
        "Recharge from rainfall During Non Monsoon Season",
        "Recharge from other sources During Non Monsoon Season",
        "Net Ground Water Availability for future use",
        "Current Annual Ground Water Extraction For Irrigation",
        "Current Annual Ground Water Extraction For Domestic & Industrial Use",
        "Stage of Ground Water Extraction (%)"
    ]].copy()

    # Fill numeric NaNs
    numeric_cols = df_model.columns.drop("Name of District")
    df_model[numeric_cols] = df_model[numeric_cols].fillna(df_model[numeric_cols].median())

    # Target (not strictly needed in app, but handy)
    df_model["groundwater_score"] = 100 - df_model["Stage of Ground Water Extraction (%)"]

    # Encode district
    encoder = LabelEncoder()
    df_model["district_encoded"] = encoder.fit_transform(df_model["Name of District"])

    # Feature columns used by the model (must match training)
    feature_cols = df_model.drop(columns=["Name of District", "groundwater_score"]).columns.tolist()

    # Add a simple risk band for visuals
    def risk_band(score):
        if score < 30:
            return "Low"
        elif score < 60:
            return "Moderate"
        elif score < 90:
            return "High"
        else:
            return "Critical"

    df_model["Risk Band"] = df_model["Stage of Ground Water Extraction (%)"].apply(risk_band)

    return df, df_model, encoder, feature_cols


@st.cache_resource
def load_model():
    with open("watersense_model.pkl", "rb") as f:
        model = pickle.load(f)
    return model


df_raw, df_model, encoder, FEATURE_COLS = load_data()
model = load_model()

DISTRICTS = sorted(df_model["Name of District"].unique())


# ------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------
def predict_for_district(district: str) -> float:
    row = df_model[df_model["Name of District"] == district].iloc[0]
    x = row[FEATURE_COLS].values.reshape(1, -1)
    pred = float(model.predict(x)[0])
    return max(0.0, min(100.0, pred))


def label_from_score(score: float) -> str:
    if score >= 70:
        return "🟢 Safe / Comfortable"
    elif score >= 40:
        return "🟠 Moderate Stress"
    else:
        return "🔴 High Water Stress"


def recommendation_text(score: float) -> str:
    if score >= 70:
        return (
            "- Current groundwater situation looks **comfortable**.\n"
            "- New borewells are relatively safer, but continuous monitoring is still important.\n"
            "- Good opportunity to promote **water-saving crops** and recharge structures."
        )
    elif score >= 40:
        return (
            "- Groundwater is under **moderate pressure**.\n"
            "- Encourage **drip/sprinkler irrigation**, crop diversification, and farm ponds.\n"
            "- New borewells should be taken up with caution and proper technical survey."
        )
    else:
        return (
            "- Area is in **high water stress**.\n"
            "- Avoid new borewells without strong hydrogeological survey.\n"
            "- Focus on **recharge structures, rainwater harvesting, and alternative water sources**.\n"
            "- Strong candidate for government intervention and water budgeting."
        )


def make_pdf_report(district: str, score: float) -> bytes:
    """Create a simple one-page PDF report in memory."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, 800, "WaterSense – Groundwater Risk Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 770, f"District: {district}")
    c.drawString(50, 750, f"Predicted Groundwater Score: {round(score, 2)} / 100")
    c.drawString(50, 730, f"Status: {label_from_score(score)}")

    c.drawString(50, 705, "Recommendations:")
    text_obj = c.beginText(60, 685)
    text_obj.setFont("Helvetica", 11)
    for line in recommendation_text(score).split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(50, 50, f"Generated on: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}  •  Prototype only")

    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.getvalue()


# fake but stable coords for map demo (conceptual, not real GIS)
def lat_lon_for_district(name: str):
    h = abs(hash(name)) % 10000
    lat = 8 + (h % 2700) / 100   # 8–35 approx
    lon = 68 + (h // 2700) / 100  # 68–90 approx
    return lat, lon


# ------------------------------------------------------
# SIDEBAR NAV
# ------------------------------------------------------
st.sidebar.title("💧 WaterSense")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "🔍 Prediction", "📊 Visualizations","📄 PDF Report"],
)


# ------------------------------------------------------
# 🏠 HOME
# ------------------------------------------------------
# ------------------------------------------------------
# 🏠 HOME (Updated Modern UI Layout)
# ------------------------------------------------------
if page == "🏠 Home":

    st.markdown(
        """
        <h1 style="text-align:center;color:#1e88e5;font-weight:900;margin-bottom:10px;">
            WaterSense – AI Groundwater Risk Predictor
        </h1>
        <p style="text-align:center;font-size:18px;">
            A decision-support tool to help farmers, planners, and policy makers forecast groundwater safety,
            prevent failed borewells, and guide sustainable water usage.
        </p>
        """,
        unsafe_allow_html=True,
    )

    st.write("---")

    # ------------------ PROBLEM ------------------
    st.markdown(
        """
        <h3 style="color:#d32f2f; font-weight:700;">🚨 The Problem</h3>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        - Groundwater levels are rapidly declining due to **over-extraction** and poor monitoring.  
        - Farmers often drill borewells **without knowing if groundwater exists**, wasting money and resources.  
        - There is **no accessible predictive tool** to understand groundwater stress at district level.  
        """
    )

    st.write("---")

    # ------------------ SOLUTION ------------------
    st.markdown(
        """
        <h3 style="color:#2e7d32; font-weight:700;">💡 The Solution</h3>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        WaterSense uses real groundwater data + machine learning to:
        - Predict groundwater safety using a **0-100 score**  
        - Classify regions as **Safe / Moderate / High Stress**  
        - Provide **actionable recommendations** to guide borewell planning and policy decisions  
        """
    )

    st.write("---")

    # ------------------ FEATURES ------------------
    st.markdown(
        """
        <h3 style="color:#1e88e5; font-weight:700;">⭐ Key Features</h3>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("✔ AI-based groundwater score prediction")
        st.markdown("✔ District-wise stress rating")

    with col2:
        st.markdown("✔ Insights dashboard and visual analytics")
        st.markdown("✔ Risk-based recommendations")

    with col3:
        st.markdown("✔ One-click PDF report export")
        st.markdown("✔ Ready for rainfall API integration")

    st.write("---")

    st.markdown(
        "<p style='font-size:18px;text-align:center;margin-top:10px;'>🎯 Start by making your first prediction below.</p>",
        unsafe_allow_html=True
    )

    if st.button("🔍 Go to Prediction"):
        st.session_state["force_page"] = "🔍 Prediction"
        st.success("Open the Prediction tab from the left sidebar.")

# ------------------------------------------------------
# 🔍 PREDICTION
# ------------------------------------------------------
elif page == "🔍 Prediction":
    st.header("🔍 Groundwater Risk Prediction")

    district = st.selectbox("Select a district", DISTRICTS, index=DISTRICTS.index("Nashik") if "Nashik" in DISTRICTS else 0)

    if st.button("Predict Groundwater Score"):
        score = predict_for_district(district)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Predicted Groundwater Score", f"{round(score, 2)}/100")
        with col2:
            st.metric("Risk Level", label_from_score(score))
        with col3:
            ext = df_model.loc[df_model["Name of District"] == district, "Stage of Ground Water Extraction (%)"].iloc[0]
            st.metric("Current Extraction", f"{round(ext, 1)} %")

        st.write("---")
        st.subheader("📌 Recommendations")
        st.markdown(recommendation_text(score))


# ------------------------------------------------------
# 📊 VISUALIZATIONS
# ------------------------------------------------------
elif page == "📊 Visualizations":
    st.header("📊 Groundwater Insights Dashboard")

    st.markdown("#### Overall Extraction Stress Across All Districts")
    fig_hist = px.histogram(
        df_model,
        x="Recharge from rainfall During Monsoon Season",
        nbins=25,
        title="Distribution of Groundwater Extraction (%)",
        color_discrete_sequence=["#1e88e5"],
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    st.write("---")
    st.markdown("#### Rainfall Recharge vs Extraction Risk")

    fig_scatter = px.scatter(
    df_model,
    x="Recharge from rainfall During Monsoon Season",
    y="groundwater_score",
    hover_name="Name of District",
    color="Risk Band",
    labels={
        "Recharge from rainfall During Monsoon Season": "Monsoon Recharge",
        "groundwater_score": "Groundwater Score (0–100)",
    },
    color_discrete_map={
        "Low": "green",
        "Moderate": "yellow",
        "High": "orange",
        "Critical": "red",
    },
)

    
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.write("---")
    st.markdown("#### Detailed Profile for a District")

    dsel = st.selectbox("Choose district to inspect", DISTRICTS)
    row = df_model[df_model["Name of District"] == dsel].iloc[0]

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Monsoon Rainfall Recharge", f"{row['Recharge from rainfall During Monsoon Season']:.2f}")
        st.metric("Non-Monsoon Recharge", f"{row['Recharge from rainfall During Non Monsoon Season']:.2f}")
    with col_b:
        st.metric("Extraction – Irrigation", f"{row['Current Annual Ground Water Extraction For Irrigation']:.2f}")
        st.metric("Extraction – Domestic/Industrial", f"{row['Current Annual Ground Water Extraction For Domestic & Industrial Use']:.2f}")

    radar_df = pd.DataFrame({
        "Metric": [
            "Monsoon Recharge",
            "Non-Monsoon Recharge",
            "Irrigation Extraction",
            "Domestic/Industrial Extraction",
            "Net Availability",
        ],
        "Value": [
            row["Recharge from rainfall During Monsoon Season"],
            row["Recharge from rainfall During Non Monsoon Season"],
            row["Current Annual Ground Water Extraction For Irrigation"],
            row["Current Annual Ground Water Extraction For Domestic & Industrial Use"],
            row["Net Ground Water Availability for future use"],
        ],
    })

    fig_radar = px.line_polar(
        radar_df,
        r="Value",
        theta="Metric",
        line_close=True,
        title=f"Groundwater Profile – {dsel}",
    )
    st.plotly_chart(fig_radar, use_container_width=True)


# ------------------------------------------------------

   



# ------------------------------------------------------
# 📄 PDF REPORT
# ------------------------------------------------------
elif page == "📄 PDF Report":
    st.header("📄 Download Groundwater Risk Report")

    district = st.selectbox("Select district to generate report", DISTRICTS)

    if st.button("Generate PDF Report"):
        score = predict_for_district(district)
        pdf_bytes = make_pdf_report(district, score)

        st.download_button(
            label="⬇ Download PDF",
            data=pdf_bytes,
            file_name=f"WaterSense_{district.replace(' ', '_')}.pdf",
            mime="application/pdf",
        )

    st.info("You can share this report with mentors, judges, or policy stakeholders as a concept demo.")
