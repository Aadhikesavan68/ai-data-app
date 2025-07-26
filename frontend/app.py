import streamlit as st
import pandas as pd
import os, sys

# Add backend path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from cleaning import clean_data
from estimation import weighted_average
from report_gen import generate_report

# ✅ Set Streamlit page config at the top
st.set_page_config(page_title="AI Survey Processor")

st.title("📊 Survey Data Uploader & Reporter")

uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("🔍 Raw Data")
    st.dataframe(df)

    if st.button("Clean & Generate Report"):
        cleaned = clean_data(df)
        st.subheader("✅ Cleaned Data")
        st.dataframe(cleaned)

        try:
            avg_income = weighted_average(cleaned, "Income", "Weight")
            st.success(f"Weighted Average Income: ₹{avg_income}")

            summary = {"total": len(cleaned), "income": avg_income}
            generate_report(summary)

            with open("report.pdf", "rb") as file:
                st.download_button("📄 Download Report", file, "survey_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"Error: {e}")
