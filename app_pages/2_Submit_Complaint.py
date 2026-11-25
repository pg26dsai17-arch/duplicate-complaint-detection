import streamlit as st

from src import config, database
from src.preprocessing import MAX_CHARS, validate_complaint
from src.priority import score_complaint
from src.ui import get_connection, get_detector, require_data, show_matches, show_priority, synthetic_note

st.title("Submit a complaint")
st.write("Before your complaint is saved, it is compared with existing complaints from the same locality. "
         "If the same problem was already reported, you can follow that complaint instead.")

conn = get_connection()
if not require_data(conn):
    st.stop()
det = get_detector()
ss = st.session_state

with st.form("complaint"):
    text = st.text_area("Describe the problem", height=130, max_chars=MAX_CHARS,
                        placeholder="e.g. Garbage has not been collected from our street for three days")
    a, b = st.columns(2)
    category = a.selectbox("Category", config.CATEGORIES, index=None, placeholder="Choose a category")
    locality = b.selectbox("Locality", database.localities(conn), index=None, placeholder="Choose or type a locality",
                           accept_new_options=True)
    checked = st.form_submit_button("Check for similar complaints", type="primary")

if checked:
    ss.pop("submitted", None)
    try:
        clean = validate_complaint(text)
        if not category or not locality:
            raise ValueError("Please choose a category and a locality.")
        ss.pending = {"text": clean, "category": category, "locality": locality.strip(),
                      "result": det.check(clean, k=config.TOP_K, locality=locality.strip())}
    except ValueError as e:
        st.error(str(e))
        ss.pop("pending", None)

if "submitted" in ss:
    c = ss.submitted
    st.success(f"Complaint saved. Your reference is **{c['complaint_reference']}**. Keep it to track progress. "
               f"Priority: **{c['priority_level']}** ({c['priority_score']}/100).")
    if c.get("flagged"):
        st.info(f"{c['flagged']} similar complaint(s) were sent to the officers for duplicate review.")

elif "pending" in ss:
    p, r = ss.pending, ss.pending["result"]
    if r["potential_duplicate"]:
        st.warning("**This problem may already be reported.** Complaints at or above the threshold are highlighted. "
                   "If one describes your problem, you don't need to submit a new one.")
    else:
        st.success("No existing complaint looks like the same problem. You can submit it.")
    show_priority(score_complaint(p["text"], p["category"]))
    st.subheader("Top 5 similar complaints")
    show_matches(r, conn)

    label = "It's a different problem, submit anyway" if r["potential_duplicate"] else "Submit complaint"
    col1, col2 = st.columns([1, 3])
    if col1.button(label, type="primary"):
        c = database.add_complaint(conn, p["text"], p["category"], p["locality"])
        dups = [m for m in r["matches"] if m["is_duplicate"]]
        database.add_matches(conn, c["id"], dups, r["model"])
        det.add(c["id"], p["text"], p["locality"])
        ss.submitted = {**c, "flagged": len(dups)}
        ss.pop("pending")
        st.rerun()
    if col2.button("Cancel"):
        ss.pop("pending")
        st.rerun()

synthetic_note()
