import html

import plotly.express as px
import streamlit as st

from src import config, database
from src.ui import PRIORITY_COLORS, get_connection, meter_html, priority_badge, require_data, synthetic_note

st.title("Admin dashboard")

# Optional, deliberately simple protection: set the ADMIN_PASSWORD environment variable to enable it.
if config.ADMIN_PASSWORD and not st.session_state.get("admin_ok"):
    pw = st.text_input("Admin password", type="password")
    if st.button("Open dashboard"):
        st.session_state.admin_ok = pw == config.ADMIN_PASSWORD
        if not st.session_state.admin_ok:
            st.error("Wrong password.")
        st.rerun()
    st.stop()

conn = get_connection()
if not require_data(conn):
    st.stop()
df = database.all_complaints(conn)
m = database.matches_table(conn)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total complaints", f"{len(df):,}")
c2.metric("Open or in progress", f"{df.status.isin(['Open', 'In Progress']).sum():,}")
c3.metric("Resolved or closed", f"{df.status.isin(['Resolved', 'Closed']).sum():,}")
c4.metric("Awaiting review", f"{(m.review_status == 'pending').sum():,}")
c5.metric("Confirmed duplicates", f"{(m.review_status == 'confirmed').sum():,}")

# ------------------------------------------------------------------ priority queue
st.subheader("Priority queue: what to solve first")
st.caption("Unsolved complaints ranked by priority score (0 to 100). The score adds up the category, risk words "
           "in the text (safety, health, daily life, vulnerable people, wide impact, urgency), how long the problem "
           "has lasted, and how many other complaints were linked to it as duplicates. Select a row to see why.")
q = database.priority_queue(conn)
levels = ["Critical", "High", "Medium", "Low"]
cols = st.columns(4)
for col, lvl in zip(cols, levels):
    col.metric(f"{lvl} (unsolved)", f"{(q.final_level == lvl).sum():,}" if len(q) else "0")

f1, f2 = st.columns(2)
show_levels = f1.multiselect("Priority", levels, default=["Critical", "High"])
show_cats = f2.multiselect("Category", sorted(q.category.unique()) if len(q) else [])
view_q = q
if show_levels:
    view_q = view_q[view_q.final_level.isin(show_levels)]
if show_cats:
    view_q = view_q[view_q.category.isin(show_cats)]
view_q = view_q.head(300).reset_index(drop=True)
view_q.insert(0, "rank", range(1, len(view_q) + 1))

if view_q.empty:
    st.info("No unsolved complaints match these filters.")
else:
    pick = st.dataframe(
        view_q[["rank", "final_level", "final_score", "complaint_reference", "category", "locality", "status",
                "extra_reports", "complaint_text"]],
        hide_index=True, use_container_width=True, height=360, on_select="rerun", selection_mode="single-row",
        column_config={
            "rank": st.column_config.NumberColumn("#", width="small"), "final_level": "Priority",
            "final_score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%d"),
            "complaint_reference": "Reference", "category": "Category", "locality": "Locality", "status": "Status",
            "extra_reports": st.column_config.NumberColumn("Linked reports", help="Other complaints linked as duplicates"),
            "complaint_text": st.column_config.TextColumn("Complaint", width="large")})
    st.caption(f"Showing {len(view_q):,} of {len(q):,} unsolved complaints (at most 300).")
    rows_sel = pick.selection.rows if pick and pick.selection else []
    if rows_sel:
        r = view_q.iloc[rows_sel[0]]
        with st.container(border=True):
            st.markdown(f"**{html.escape(r.complaint_reference)}** &nbsp; {priority_badge(r.final_level, r.final_score)}"
                        f" &nbsp; {html.escape(r.category)}, {html.escape(r.locality)}, {html.escape(r.status)}",
                        unsafe_allow_html=True)
            st.write(r.complaint_text)
            st.markdown("**Why this priority:**\n" + "\n".join(f"- {x.strip()}" for x in r.priority_reasons.split(";") if x.strip()))
            b1, b2, _ = st.columns([1, 1, 3])
            if r.status == "Open" and b1.button("Mark In Progress", key="q_prog"):
                database.update_status(conn, int(r.id), "In Progress"); st.rerun()
            if b2.button("Mark Resolved", key="q_res", type="primary"):
                database.update_status(conn, int(r.id), "Resolved"); st.rerun()

st.divider()
a, b = st.columns([3, 2], gap="large")
with a:
    st.subheader("Open and closed complaints by category")
    t = df.assign(state=df.status.map(lambda s: "Open or in progress" if s in ("Open", "In Progress") else "Resolved or closed"))
    g = t.groupby(["category", "state"]).size().reset_index(name="n")
    fig = px.bar(g, x="n", y="category", color="state", orientation="h",
                 color_discrete_map={"Open or in progress": "#0F6E6A", "Resolved or closed": "#A9C9C5"},
                 labels={"n": "Complaints", "category": "", "state": ""})
    fig.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white",
                      yaxis={"categoryorder": "total ascending"}, legend=dict(orientation="h", y=1.06))
    st.plotly_chart(fig, use_container_width=True)
with b:
    st.subheader("Status distribution")
    s = df.status.value_counts().reindex(config.STATUSES).fillna(0)
    pie = px.pie(values=s.values, names=s.index, hole=0.55,
                 color=s.index, color_discrete_map={"Open": "#B4412F", "In Progress": "#E8B931",
                                                    "Resolved": "#2F7D4F", "Closed": "#8A9590"})
    pie.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(pie, use_container_width=True)

    st.subheader("Unsolved complaints by priority")
    pc = q.final_level.value_counts().reindex(levels).fillna(0) if len(q) else None
    if pc is not None:
        bar = px.bar(x=pc.index, y=pc.values, color=pc.index, color_discrete_map=PRIORITY_COLORS,
                     labels={"x": "", "y": "Complaints"})
        bar.update_layout(height=260, showlegend=False, margin=dict(l=0, r=0, t=10, b=0), plot_bgcolor="white")
        st.plotly_chart(bar, use_container_width=True)

    st.subheader("Update a complaint's status")
    ref = st.text_input("Complaint reference", placeholder="CMP00042 or CD-20261117-K7P3QX")
    if ref:
        c = database.get_by_reference(conn, ref)
        if not c:
            st.error("No complaint with this reference.")
        else:
            st.caption(c["complaint_text"])
            new = st.selectbox("New status", config.STATUSES, index=config.STATUSES.index(c["status"]))
            if st.button("Save status", type="primary", disabled=new == c["status"]):
                database.update_status(conn, c["id"], new)
                st.success(f"{c['complaint_reference']} is now {new}.")

st.subheader("Potential duplicates")
st.caption("Each pair is a newly submitted complaint and an existing complaint that scored at or above the "
           "threshold. Confirm only if both describe the same problem at the same place.")
view = st.radio("Show", ["pending", "confirmed", "rejected"], horizontal=True,
                format_func=lambda v: {"pending": "Awaiting review", "confirmed": "Confirmed", "rejected": "Rejected"}[v])
rows = database.matches_table(conn, view)
if rows.empty:
    st.info("Nothing here. Suggestions appear when someone submits a complaint similar to an existing one.")
for r in rows.itertuples():
    with st.container(border=True):
        left, right = st.columns(2)
        left.markdown(f"<span class='meta'>New complaint <b>{html.escape(r.new_reference)}</b>, {html.escape(r.new_locality)}</span>"
                      f"<div class='text'>{html.escape(r.new_text)}</div>", unsafe_allow_html=True)
        right.markdown(f"<span class='meta'>Existing complaint <b>{html.escape(r.matched_reference)}</b>, "
                       f"{html.escape(r.matched_locality)}, {html.escape(r.matched_status)}</span>"
                       f"<div class='text'>{html.escape(r.matched_text)}</div>", unsafe_allow_html=True)
        x, y, z = st.columns([3, 1, 1])
        model = "Sentence-BERT" if r.model_name == "sbert" else "TF-IDF"
        x.markdown(meter_html(r.similarity_score, None, f"{model} similarity"), unsafe_allow_html=True)
        if view != "confirmed" and y.button("Same problem", key=f"c{r.id}", type="primary"):
            database.set_review(conn, r.id, "confirmed"); st.rerun()
        if view != "rejected" and z.button("Different problem", key=f"r{r.id}"):
            database.set_review(conn, r.id, "rejected"); st.rerun()
synthetic_note()
