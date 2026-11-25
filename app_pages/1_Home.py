import plotly.express as px
import streamlit as st

from src import database
from src.data_loader import DatasetMissing, load_complaints, load_pairs
from src.ui import get_connection, get_detector, require_data, synthetic_note

st.title("Duplicate complaint detection")
st.write("Citizens often report the same problem in different words. This application compares the "
         "**meaning** of a new complaint with stored complaints, so one problem becomes one complaint.")

conn = get_connection()
if not require_data(conn):
    st.stop()

df = database.all_complaints(conn)


@st.cache_data
def dataset_summary():
    """Read the CSV files once per session (cached), not on every page refresh."""
    try:
        c, p = load_complaints(), load_pairs()
    except DatasetMissing:
        return None
    return {"pairs": len(p), "dup": int(p.duplicate_label.sum()), "non": int((p.duplicate_label == 0).sum()),
            "issues": c.issue_id.nunique(), "localities": c.locality.nunique()}


info = dataset_summary() or {"pairs": 0, "dup": 0, "non": 0, "issues": 0, "localities": 0}
dup, non = info["dup"], info["non"]
pending = len(database.matches_table(conn, "pending"))

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Complaints stored", f"{len(df):,}")
c2.metric("Submitted here", f"{(df.data_origin == 'user').sum():,}")
c3.metric("Duplicate pairs", f"{dup:,}", help="Labelled pairs in the synthetic dataset")
c4.metric("Non-duplicate pairs", f"{non:,}", help="Labelled pairs in the synthetic dataset")
c5.metric("Categories", df.category.nunique())

left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Complaints by category")
    counts = df.category.value_counts().sort_values()
    fig = px.bar(x=counts.values, y=counts.index, orientation="h", labels={"x": "Complaints", "y": ""},
                 color_discrete_sequence=["#0F6E6A"])
    fig.update_layout(height=520, margin=dict(l=0, r=10, t=10, b=0), plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Matching models")
    det = get_detector()
    for name in ("tfidf", "sbert"):
        label = "TF-IDF" if name == "tfidf" else "Sentence-BERT (all-MiniLM-L6-v2)"
        if name in det.models:
            t = det.thresholds[name]
            st.markdown(f"**{label}**: ready. Threshold {t['text']:.3f} ({t['source']}).")
        else:
            st.markdown(f"**{label}**: not available. {det.unavailable.get(name, '')}")
    if pending:
        st.warning(f"{pending} suggested duplicate(s) are waiting for review on the Admin dashboard.")

    st.subheader("Dataset")
    st.markdown(
        "- **Synthetic** civic complaints generated for this project (not real SMC data)\n"
        f"- {info['pairs']:,} labelled pairs; {info['issues']:,} underlying issues in {info['localities']} fictional localities\n"
        "- Split by locality: 70% train, 15% validation, 15% test, so no paraphrase family is shared between splits\n"
        "- Details: `data/DATASET_README.md`")

st.subheader("Recent complaints")
st.dataframe(df.head(10)[["complaint_reference", "complaint_text", "category", "locality", "status", "created_at",
                          "data_origin"]],
             hide_index=True, use_container_width=True,
             column_config={"complaint_reference": "Reference", "complaint_text": st.column_config.TextColumn("Complaint", width="large"),
                            "category": "Category", "locality": "Locality", "status": "Status",
                            "created_at": "Created", "data_origin": "Origin"})
synthetic_note()
