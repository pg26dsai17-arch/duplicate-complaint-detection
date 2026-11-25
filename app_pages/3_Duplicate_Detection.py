import streamlit as st

from src import config, database
from src.preprocessing import validate_complaint
from src.sentence_bert_model import SbertUnavailable
from src.ui import MODEL_LABELS, get_connection, get_detector, require_data, show_matches, synthetic_note

st.title("Duplicate detection")
st.write("Try any complaint text, choose a model and adjust the threshold to see how the decision changes. "
         "Nothing is saved on this page.")

conn = get_connection()
if not require_data(conn):
    st.stop()
det = get_detector()

text = st.text_area("Complaint text", height=110,
                    value="Waste collection vehicle has not come to our lane since Monday")
a, b, c = st.columns([1, 1, 1])
options = ["tfidf", "sbert"]
model = a.radio("Model", options, format_func=lambda m: MODEL_LABELS[m],
                index=options.index(det.default_model), horizontal=True)
if model not in det.models:
    st.error(f"Sentence-BERT is not available: {det.unavailable.get(model)}")
    st.stop()
locality = b.selectbox("Only compare within locality (optional)", database.localities(conn), index=None,
                       placeholder="All localities")
default = det.threshold(model, bool(locality))
threshold = c.slider("Duplicate threshold", 0.0, 1.0, float(round(default, 3)), 0.001, format="%.3f",
                     help=f"Default {default:.3f} ({det.thresholds[model]['source']}).")
compare = st.toggle("Also show the other model side by side", value=False, disabled=len(det.models) < 2)

if st.button("Find similar complaints", type="primary"):
    try:
        clean = validate_complaint(text)
    except ValueError as e:
        st.error(str(e)); st.stop()
    models = [model] + ([m for m in det.models if m != model] if compare else [])
    cols = st.columns(len(models))
    for col, m in zip(cols, models):
        with col:
            thr = threshold if m == model else None
            r = det.check(clean, model=m, k=config.TOP_K, locality=locality, threshold=thr)
            st.subheader(MODEL_LABELS[m])
            if r["potential_duplicate"]:
                best = r["matches"][0]
                st.warning(f"**Potential duplicate.** The best match scores {best['score']:.3f}, at or above "
                           f"the threshold {r['threshold']:.3f}. A person should confirm it.")
            else:
                st.success(f"**No potential duplicate** at threshold {r['threshold']:.3f}.")
            show_matches(r, conn)
synthetic_note()
