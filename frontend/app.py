# frontend/app.py

# ----------------------------------------------------------
# 🚀 AI Survey Data Cleaner & Reporter (Streamlit Frontend)
# Author: Aadhi Kesavan
# ----------------------------------------------------------

import streamlit as st
import pandas as pd
import os, sys
import altair as alt

# ✅ Set page configuration
st.set_page_config(
    page_title="AI Survey Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ✅ Add backend path so you can import cleaning & reporting logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

# ✅ Import backend functions
from cleaning import clean_data
from estimation import weighted_average
from report_gen import generate_report

# ✅ Sidebar - Branding & Navigation
st.sidebar.image("https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.png", width=200)
st.sidebar.title("📋 Survey Data Tool")
st.sidebar.markdown("### Navigate")
st.sidebar.markdown("- 📁 Upload CSV\n- 🧼 Clean Data\n- ⚖️ Estimate\n- 🧾 Generate Report")

# ✅ App Header
st.markdown("<h1 style='text-align:center; color:#4CAF50;'>🧠 AI-Powered Survey Data Processor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Smartly clean, analyze, and generate survey reports with ease</p>", unsafe_allow_html=True)
st.markdown("---")

# ✅ Upload Section
uploaded_file = st.file_uploader("📤 Upload your CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
    except:
        df = pd.read_excel(uploaded_file)

    st.subheader("🔍 Preview: Raw Data")
    st.dataframe(df.head())

    # Clean + Generate
    if st.button("🧼 Clean & Generate Report"):
        cleaned = clean_data(df)

        st.subheader("✅ Cleaned Data")
        st.dataframe(cleaned)

        try:
            # Estimate average income
            avg_income = weighted_average(cleaned, "Income", "Weight")

            # Summary section
            col1, col2 = st.columns(2)
            col1.metric("🧾 Total Records", len(cleaned))
            col2.metric("💰 Weighted Avg Income", f"₹{avg_income:,.2f}")

            # Chart (if 'Income' exists)
            if "Income" in cleaned.columns:
                st.subheader("📈 Income Distribution")
                chart = alt.Chart(cleaned).mark_bar().encode(
                    x='Income:Q',
                    y='count():Q'
                ).properties(width=700, height=300)
                st.altair_chart(chart)

            # Generate report
            summary = {"total": len(cleaned), "income": avg_income}
            report_path = generate_report(summary)

            with open(report_path, "rb") as file:
                st.download_button("📄 Download Report", file, "survey_report.pdf", mime="application/pdf")
                st.success("🎉 Report generated successfully!")

        except Exception as e:
            st.error(f"⚠️ Something went wrong: {e}")

# ✅ Footer
st.markdown("---")
st.markdown("<center style='color:gray;'>Made by Aadhi & Team | Powered by Streamlit</center>", unsafe_allow_html=True)
