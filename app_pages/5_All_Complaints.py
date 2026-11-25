import streamlit as st

from src import config, database
from src.ui import get_connection, priority_badge, require_data, status_pill, synthetic_note

st.title("All complaints")
conn = get_connection()
if not require_data(conn):
    st.stop()
df = database.all_complaints(conn)

q = st.text_input("Search text or reference", placeholder="e.g. streetlight, CMP00042, CD-2026")
a, b, c, d, e = st.columns(5)
cats = a.multiselect("Category", sorted(df.category.unique()))
locs = b.multiselect("Locality", sorted(df.locality.unique()))
stats = c.multiselect("Status", config.STATUSES)
origin = d.selectbox("Source", ["All", "Submitted in this app", "Synthetic dataset"])
prios = e.multiselect("Priority", ["Critical", "High", "Medium", "Low"])

f = df
if q:
    f = f[f.complaint_text.str.contains(q, case=False, regex=False) | f.complaint_reference.str.contains(q, case=False, regex=False)]
if cats:
    f = f[f.category.isin(cats)]
if locs:
    f = f[f.locality.isin(locs)]
if stats:
    f = f[f.status.isin(stats)]
if prios:
    f = f[f.priority_level.isin(prios)]
if origin != "All":
    f = f[f.data_origin == ("user" if origin.startswith("Submitted") else "synthetic")]

st.caption(f"{len(f):,} of {len(df):,} complaints. Select a row to see its details.")
event = st.dataframe(
    f[["complaint_reference", "priority_level", "priority_score", "complaint_text", "category", "locality", "status",
       "created_at", "data_origin"]],
    hide_index=True, use_container_width=True, height=420, on_select="rerun", selection_mode="single-row",
    column_config={"complaint_reference": "Reference", "priority_level": "Priority",
                   "priority_score": st.column_config.NumberColumn("Score", format="%d"), "complaint_text": st.column_config.TextColumn("Complaint", width="large"),
                   "category": "Category", "locality": "Locality", "status": "Status", "created_at": "Created",
                   "data_origin": "Origin"})

rows = event.selection.rows if event and event.selection else []
if rows:
    c = f.iloc[rows[0]]
    st.subheader(f"{c.complaint_reference}")
    st.markdown(f"{status_pill(c.status)} &nbsp; {c.category}, {c.locality}, created {c.created_at[:10]}, "
                f"{'synthetic record' if c.data_origin == 'synthetic' else 'submitted in this app'}",
                unsafe_allow_html=True)
    st.write(c.complaint_text)
    st.markdown(f"Priority {priority_badge(c.priority_level, c.priority_score)}", unsafe_allow_html=True)
    st.caption(f"Why: {c.priority_reasons}")
    m = database.matches_table(conn)
    related = m[(m.complaint_id == c.id) | (m.matched_id == c.id)]
    if len(related):
        st.markdown("**Duplicate suggestions involving this complaint**")
        st.dataframe(related[["new_reference", "matched_reference", "similarity_score", "model_name", "review_status"]],
                     hide_index=True, use_container_width=True)
synthetic_note()
