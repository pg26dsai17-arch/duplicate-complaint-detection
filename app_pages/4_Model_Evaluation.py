import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.ui import synthetic_note

COLORS = {"tfidf": "#8A9590", "sbert": "#0F6E6A"}

st.title("Model evaluation")
st.write("Every number on this page is read from `results/metrics.json`, written by `python setup_project.py`. "
         "Thresholds were chosen on **validation** pairs; the metrics are measured once on the **test** split.")

if not config.METRICS_FILE.exists():
    st.warning("No results yet. Run `python setup_project.py` in the terminal, then reload this page.")
    st.stop()
R = json.loads(config.METRICS_FILE.read_text())
ok = {k: v for k, v in R["models"].items() if v.get("status") == "ok"}
for k, v in R["models"].items():
    if v.get("status") != "ok":
        st.warning(f"**{v['display_name']} has no results yet.** {v.get('reason', '')} "
                   "After fixing this, run `python setup_project.py` again.")
if not ok:
    st.stop()
st.caption(f"Generated {R['generated_at']} on {R['machine']}. Dataset: {R['dataset']}.")

st.subheader("Comparison on the test set")
rows = []
for k, m in ok.items():
    t = m["test"]
    rows.append({"Model": m["display_name"], "Threshold": m["threshold"], "Accuracy": t["accuracy"],
                 "Precision": t["precision"], "Recall": t["recall"], "F1-score": t["f1"], "ROC-AUC": t["roc_auc"],
                 "Avg. precision": t["average_precision"], "Recall@5": m["retrieval"]["recall@5"],
                 "Query time (ms)": m["speed_ms"]["mean"]})
st.dataframe(pd.DataFrame(rows).style.format(precision=3), hide_index=True, use_container_width=True)

fig = go.Figure()
names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
for k, m in ok.items():
    fig.add_bar(name=m["display_name"], x=["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
                y=[m["test"][n] for n in names], marker_color=COLORS[k],
                text=[f"{m['test'][n]:.3f}" for n in names], textposition="outside")
fig.update_layout(barmode="group", yaxis_range=[0, 1.08], height=360, plot_bgcolor="white",
                  margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.12))
st.plotly_chart(fig, use_container_width=True)

st.subheader("Confusion matrices (test)")
cols = st.columns(len(ok))
for col, (k, m) in zip(cols, ok.items()):
    c = m["test"]["confusion_matrix"]
    z = [[c["tn"], c["fp"]], [c["fn"], c["tp"]]]
    hm = go.Figure(go.Heatmap(z=z, x=["Predicted: not duplicate", "Predicted: duplicate"],
                              y=["Actual: not duplicate", "Actual: duplicate"], text=z, texttemplate="%{text}",
                              colorscale=[[0, "#F4F6F5"], [1, "#0F6E6A"]], showscale=False))
    hm.update_layout(title=f"{m['display_name']} (threshold {m['threshold']:.3f})", height=320,
                     margin=dict(l=0, r=0, t=40, b=0), yaxis_autorange="reversed")
    col.plotly_chart(hm, use_container_width=True)

st.subheader("Similarity score distributions (test)")
st.caption("Good models put duplicate pairs (coloured) to the right of non-duplicate pairs (grey). "
           "The dashed line is the threshold; overlap is where mistakes happen.")
cols = st.columns(len(ok))
for col, (k, m) in zip(cols, ok.items()):
    h = m["score_hist"]
    centers = [(a + b) / 2 for a, b in zip(h["bins"][:-1], h["bins"][1:])]
    f = go.Figure()
    f.add_bar(x=centers, y=h["not_duplicate"], name="Not duplicate", marker_color="#B9C4BF", opacity=0.8)
    f.add_bar(x=centers, y=h["duplicate"], name="Duplicate", marker_color=COLORS[k], opacity=0.8)
    f.add_vline(x=m["threshold"], line_dash="dash", line_color="#B4412F")
    f.update_layout(barmode="overlay", title=m["display_name"], height=300, plot_bgcolor="white",
                    xaxis_title="Cosine similarity", margin=dict(l=0, r=0, t=40, b=0), legend=dict(orientation="h", y=1.15))
    col.plotly_chart(f, use_container_width=True)

st.subheader("Precision-recall curve (test)")
pr = go.Figure()
for k, m in ok.items():
    pr.add_scatter(x=[p["recall"] for p in m["pr_curve"]], y=[p["precision"] for p in m["pr_curve"]], mode="lines",
                   name=f"{m['display_name']} (AP {m['test']['average_precision']:.3f})", line_color=COLORS[k])
pr.update_layout(xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0.4, 1.02], height=360,
                 plot_bgcolor="white", margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(pr, use_container_width=True)

a, b = st.columns(2, gap="large")
with a:
    st.subheader("Accuracy by pair type")
    types = sorted(next(iter(ok.values()))["accuracy_by_pair_type"])
    st.dataframe(pd.DataFrame({m["display_name"]: [m["accuracy_by_pair_type"][t] for t in types] for m in ok.values()},
                              index=[t.replace("_", " ") for t in types]).style.format("{:.3f}"), use_container_width=True)
    st.caption("Hard negatives are different problems that look alike: same locality, or same problem type elsewhere.")
with b:
    st.subheader("With the same-locality rule")
    st.dataframe(pd.DataFrame([{"Model": m["display_name"], "Threshold": m["locality_rule"]["threshold"],
                                "Precision": m["locality_rule"]["test"]["precision"],
                                "Recall": m["locality_rule"]["test"]["recall"], "F1-score": m["locality_rule"]["test"]["f1"]}
                               for m in ok.values()]).style.format(precision=3), hide_index=True, use_container_width=True)
    st.caption("A pair counts as a duplicate only if the score passes the threshold AND both complaints come "
               "from the same locality (a field the complaint form records).")
    st.subheader("Speed")
    st.dataframe(pd.DataFrame([{"Model": m["display_name"], "Mean (ms)": m["speed_ms"]["mean"],
                                "95th percentile (ms)": m["speed_ms"]["p95"], "Complaints searched": m["speed_ms"]["corpus"]}
                               for m in ok.values()]).style.format(precision=1), hide_index=True, use_container_width=True)
    st.caption("Time to encode one complaint and score it against the test complaints, on the machine above.")

st.info("These results come from a **synthetic** dataset. They show how the methods behave under controlled "
        "conditions and are likely better than what real complaints would give. They are not evidence of "
        "real-world municipal performance.")
synthetic_note()
