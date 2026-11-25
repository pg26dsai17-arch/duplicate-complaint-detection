"""Entry point. Start the whole application with:

    streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="Complaint Desk", page_icon="📋", layout="wide")

from src.ui import get_connection, setup_page  # noqa: E402

setup_page()
get_connection()      # creates data/complaints.db and its tables on first start

pages = [
    st.Page("app_pages/1_Home.py", title="Home", icon=":material/home:", default=True),
    st.Page("app_pages/2_Submit_Complaint.py", title="Submit complaint", icon=":material/edit_note:"),
    st.Page("app_pages/3_Duplicate_Detection.py", title="Duplicate detection", icon=":material/content_copy:"),
    st.Page("app_pages/4_Model_Evaluation.py", title="Model evaluation", icon=":material/analytics:"),
    st.Page("app_pages/5_All_Complaints.py", title="All complaints", icon=":material/table_rows:"),
    st.Page("app_pages/6_Admin_Dashboard.py", title="Admin dashboard", icon=":material/admin_panel_settings:"),
]
with st.sidebar:
    st.markdown("### Complaint Desk")
    st.caption("Duplicate complaint detection with TF-IDF and Sentence-BERT")
st.navigation(pages).run()
