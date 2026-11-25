"""Shared Streamlit helpers: cached resources, styling and the similarity meter."""
import html

import streamlit as st

from src import config, database
from src.detector import DuplicateDetector

CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1150px;}
  h1 {font-weight: 700; letter-spacing: -0.02em;}
  [data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #D8DFDB; border-radius: 10px; padding: 14px 16px;}
  [data-testid="stMetricLabel"] p {color: #5D6A65; font-size: 0.9rem;}
  .synthetic-note {color: #5D6A65; font-size: 0.85rem; border-top: 1px solid #D8DFDB; padding-top: .6rem; margin-top: 2rem;}
  .match {background:#FFFFFF; border:1px solid #D8DFDB; border-radius:10px; padding:14px 16px; margin-bottom:10px;}
  .match.dup {border-color:#E8B931; box-shadow: inset 4px 0 0 #E8B931;}
  .match .meta {color:#5D6A65; font-size:.85rem;}
  .match .text {margin:.35rem 0 .6rem 0; line-height:1.5;}
  .pill {display:inline-block; border-radius:999px; padding:1px 9px; font-size:.78rem; font-weight:600; margin-left:4px;}
  .Open {background:#F8E5E1; color:#B4412F;} .In.Progress, .InProgress {background:#FBF0CC; color:#5C4300;}
  .Resolved {background:#E3F1E8; color:#2F7D4F;} .Closed {background:#E6EAE8; color:#3A4643;}
  .meter {position:relative; height:10px; border-radius:3px; background:#E4E9E6; max-width:340px;}
  .meter .fill {position:absolute; inset:0 auto 0 0; border-radius:3px;}
  .meter .tick {position:absolute; top:-4px; bottom:-4px; width:3px; border-radius:2px; background:#17201E;}
  .score {font-weight:700; font-variant-numeric: tabular-nums;}
  .prio {display:inline-block; border-radius:6px; padding:2px 10px; font-weight:700; font-size:.85rem;}
  .prio.Critical {background:#B4412F; color:#FFFFFF;} .prio.High {background:#F9E1D8; color:#8A2F1E;}
  .prio.Medium {background:#FBF0CC; color:#5C4300;} .prio.Low {background:#E6EAE8; color:#3A4643;}
</style>
"""


def setup_page():
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def get_connection():
    conn = database.connect()
    database.create_tables(conn)        # the database is created automatically on first start
    return conn


@st.cache_resource(show_spinner="Loading models and stored complaints (first start can take a few minutes) ...")
def get_detector():
    return DuplicateDetector(database.corpus(get_connection()))


def synthetic_note():
    st.markdown('<p class="synthetic-note">Research prototype. The pre-loaded complaints are <b>synthetic</b> records '
                'generated for this project. They are not Surat Municipal Corporation data and were not submitted '
                'by real residents.</p>', unsafe_allow_html=True)


def require_data(conn) -> bool:
    """Show a clear message and a load button if the database is empty."""
    if database.count_complaints(conn) > 0:
        return True
    st.warning("The database is empty. Load the synthetic dataset to start "
               "(or run `python setup_project.py` in the terminal).")
    if st.button("Load the 10,000 synthetic complaints now", type="primary"):
        with st.spinner("Loading ..."):
            try:
                n = database.load_synthetic_dataset(conn)
            except FileNotFoundError as e:
                st.error(f"Dataset not found: {e}")
                return False
        get_detector.clear()
        st.success(f"Loaded {n:,} complaints.")
        st.rerun()
    return False


def meter_html(score: float, threshold: float | None, label: str | None = None) -> str:
    """Similarity meter: bar = score, black tick = threshold, yellow when at or above it.
    With threshold=None (stored matches) the bar is drawn without a tick."""
    s = max(0.0, min(1.0, score))
    above = threshold is None or score >= threshold
    color = "#E8B931" if above else "rgba(15,110,106,.55)"
    note = label or (f"{'at or above' if above else 'below'} threshold {threshold:.3f}" if threshold is not None else "")
    tick = "" if threshold is None else \
        f'<div class="tick" style="left:calc({max(0.0, min(1.0, threshold))*100:.1f}% - 1.5px)"></div>'
    return (f'<div><span class="score" style="color:{"#5C4300" if above else "#3A4643"}">{score:.3f}</span> '
            f'<span class="meta">{html.escape(note)}</span>'
            f'<div class="meter"><div class="fill" style="width:{s*100:.1f}%;background:{color}"></div>{tick}</div></div>')


def status_pill(status: str) -> str:
    return f'<span class="pill {status.replace(" ", "")}">{html.escape(status)}</span>'


def show_matches(result: dict, conn):
    """Render a detector result as match cards."""
    rows = database.complaints_by_ids(conn, [m["id"] for m in result["matches"]])
    if not result["matches"]:
        st.info("There are no stored complaints to compare with" +
                (f" in {result['locality']}." if result["locality"] else "."))
        return
    for m in result["matches"]:
        c = rows.get(m["id"])
        if not c:
            continue
        origin = ", synthetic record" if c["data_origin"] == "synthetic" else ""
        st.markdown(
            f'<div class="match {"dup" if m["is_duplicate"] else ""}">'
            f'<div class="meta">#{m["rank"]} &nbsp;<b>{html.escape(c["complaint_reference"])}</b>'
            f'{status_pill(c["status"])} &nbsp; {html.escape(c["category"])}, {html.escape(c["locality"])}, '
            f'{c["created_at"][:10]}{origin}</div>'
            f'<div class="text">{html.escape(c["complaint_text"])}</div>'
            f'{meter_html(m["score"], result["threshold"])}</div>', unsafe_allow_html=True)
    name = "Sentence-BERT" if result["model"] == "sbert" else "TF-IDF"
    st.caption(f"{name} compared the text with {result['corpus_size']:,} stored complaints"
               f"{' in ' + result['locality'] if result['locality'] else ''} in {result['seconds']*1000:.0f} ms. "
               "Scores are cosine similarities (0 = unrelated, 1 = identical meaning), not probabilities.")


def priority_badge(level: str, score: int | None = None) -> str:
    label = html.escape(level or "Unscored") + (f" ({int(score)})" if score is not None else "")
    return f'<span class="prio {html.escape(level or "")}">{label}</span>'


def show_priority(p) -> None:
    """Show a Priority result (src.priority.Priority) with its reasons."""
    st.markdown(f"**Estimated priority:** {priority_badge(p.level, p.score)}", unsafe_allow_html=True)
    st.caption("Why: " + p.reason_text + ". Scores range from 0 to 100; Critical starts at 75.")


PRIORITY_COLORS = {"Critical": "#B4412F", "High": "#E07A5F", "Medium": "#E8B931", "Low": "#A9B4AF"}

MODEL_LABELS = {"tfidf": "TF-IDF", "sbert": "Sentence-BERT"}
