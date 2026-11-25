"""Compare TF-IDF and Sentence-BERT and save REAL measured results.

Run:  python setup_project.py          (does this as one of its steps)
  or: python -m src.evaluation         (only the evaluation)

Protocol (no test data is used for any decision):
1. TF-IDF vocabulary is fitted on TRAIN complaints only.
2. The duplicate threshold is chosen on VALIDATION pairs (the threshold with the best F1).
3. All reported numbers are computed once on TEST pairs and TEST complaints.
The dataset is split by locality, so no paraphrase family appears in two splits.
"""
import json
import platform
import time
from datetime import datetime

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

from src import config
from src.data_loader import load_complaints, load_pairs
from src.sentence_bert_model import SbertUnavailable, SentenceBertModel, get_embeddings
from src.similarity import paired_cosine, predict, select_threshold
from src.tfidf_model import TfidfModel


def metrics_at(labels, scores, threshold) -> dict:
    pred = predict(scores, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel()
    return {"accuracy": float(accuracy_score(labels, pred)),
            "precision": float(precision_score(labels, pred, zero_division=0)),
            "recall": float(recall_score(labels, pred, zero_division=0)),
            "f1": float(f1_score(labels, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(labels, scores)),
            "average_precision": float(average_precision_score(labels, scores)),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def evaluate_model(model, data: dict, cache_name: str | None = None) -> dict:
    """Full evaluation of one model. `data` holds the train/val/test DataFrames."""
    model.fit(data["train_c"].complaint_text.tolist())

    def vectors(split):
        c = data[f"{split}_c"]
        ids, texts = c.complaint_id.tolist(), c.complaint_text.tolist()
        vecs = (get_embeddings(model, f"{cache_name}_{split}", ids, texts) if cache_name
                else model.encode(texts))
        return dict(zip(ids, range(len(ids)))), vecs

    def pair_scores(split):
        pos, vecs = vectors(split)
        p = data[f"{split}_p"]
        return paired_cosine(vecs[[pos[i] for i in p.complaint_id_1]], vecs[[pos[i] for i in p.complaint_id_2]])

    val_p, test_p = data["val_p"], data["test_p"]
    yv, yt = val_p.duplicate_label.to_numpy(), test_p.duplicate_label.to_numpy()
    sv, st = pair_scores("val"), pair_scores("test")

    thr, val_f1 = select_threshold(sv, yv)
    test = metrics_at(yt, st, thr)

    # Same-locality rule: a duplicate must ALSO come from the same locality (a form field in real life).
    def same_loc(split):
        loc = data[f"{split}_c"].set_index("complaint_id").locality
        p = data[f"{split}_p"]
        return (p.complaint_id_1.map(loc) == p.complaint_id_2.map(loc)).to_numpy()
    hv, ht = np.where(same_loc("val"), sv, -1.0), np.where(same_loc("test"), st, -1.0)
    loc_thr, _ = select_threshold(hv, yv)

    pred = predict(st, thr)
    by_type = {t: float((pred[g.index] == g.duplicate_label).mean())
               for t, g in test_p.reset_index(drop=True).groupby("pair_type")}

    prec, rec, _ = precision_recall_curve(yt, st)
    step = max(1, len(prec) // 60)
    bins = np.linspace(min(-0.1, float(st.min())), 1.0, 41)

    # Top-5 retrieval: each test complaint with a known duplicate searches all other test complaints.
    tc = data["test_c"]
    _, tv = vectors("test")
    from src.similarity import normalize_rows
    n = normalize_rows(tv)
    sim = n @ n.T
    sim = np.asarray(sim.todense() if hasattr(sim, "todense") else sim)
    issue = tc.issue_id.to_numpy()
    sizes = tc.groupby("issue_id").size()
    queries = np.where(tc.issue_id.map(sizes).to_numpy() >= 2)[0]
    hit1 = hit5 = 0
    for q in queries:
        s = sim[q].copy(); s[q] = -np.inf
        order = np.argsort(-s)[:5]
        hit1 += issue[order[0]] == issue[q]
        hit5 += (issue[order] == issue[q]).any()

    # Speed: encode one complaint and score it against the test corpus.
    sample = tc.complaint_text.sample(100, random_state=0).tolist()
    times = []
    for text in sample:
        t0 = time.perf_counter()
        q = model.encode([text]); normalize_rows(tv) @ normalize_rows(q).T
        times.append((time.perf_counter() - t0) * 1000)

    return {
        "status": "ok", "display_name": model.display_name, "threshold": thr, "val_f1": val_f1,
        "test": test, "locality_rule": {"threshold": loc_thr, "test": metrics_at(yt, ht, loc_thr)},
        "accuracy_by_pair_type": by_type,
        "pr_curve": [{"recall": float(r), "precision": float(p)} for p, r in zip(prec[::step], rec[::step])],
        "score_hist": {"bins": bins.round(3).tolist(),
                       "duplicate": np.histogram(st[yt == 1], bins)[0].tolist(),
                       "not_duplicate": np.histogram(st[yt == 0], bins)[0].tolist()},
        "retrieval": {"recall@1": hit1 / len(queries), "recall@5": hit5 / len(queries), "queries": int(len(queries))},
        "speed_ms": {"mean": float(np.mean(times)), "p95": float(np.percentile(times, 95)), "corpus": int(len(tc))},
    }


def load_data() -> dict:
    d = {f"{s}_c": load_complaints(s) for s in ("train", "val", "test")}
    d.update({f"{s}_p": load_pairs(s) for s in ("val", "test")})
    return d


def run(models=("tfidf", "sbert"), sbert_encoder=None) -> dict:
    data = load_data()
    old = json.loads(config.METRICS_FILE.read_text()) if config.METRICS_FILE.exists() else {}
    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "synthetic (not real Surat Municipal Corporation data)",
        "split_sizes": {k: len(v) for k, v in data.items()},
        "machine": f"{platform.system()} {platform.machine()}, Python {platform.python_version()}",
        "models": old.get("models", {}),
    }
    for name in models:
        print(f"Evaluating {name} ...")
        try:
            if name == "tfidf":
                r = evaluate_model(TfidfModel(), data)
            else:
                r = evaluate_model(SentenceBertModel(encoder=sbert_encoder), data, cache_name="eval_sbert")
        except SbertUnavailable as e:
            print(f"  skipped: {e}")
            if results["models"].get(name, {}).get("status") != "ok":
                results["models"][name] = {"status": "not_run", "display_name": "Sentence-BERT", "reason": str(e)}
            continue
        results["models"][name] = r
        t = r["test"]
        print(f"  threshold={r['threshold']:.3f}  F1={t['f1']:.3f}  precision={t['precision']:.3f}  "
              f"recall={t['recall']:.3f}  recall@5={r['retrieval']['recall@5']:.3f}")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.METRICS_FILE.write_text(json.dumps(results, indent=2))
    make_figures(results)
    print(f"Saved {config.METRICS_FILE}")
    return results


def make_figures(results: dict) -> None:
    """PNG figures for the report (the app draws its own interactive charts)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    ok = {k: v for k, v in results["models"].items() if v.get("status") == "ok"}
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    colors = {"tfidf": "#8A9590", "sbert": "#0F6E6A"}
    for name, m in ok.items():
        cm = m["test"]["confusion_matrix"]
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
        sns.heatmap([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]], annot=True, fmt="d", cmap="Greens", cbar=False,
                    xticklabels=["Not duplicate", "Duplicate"], yticklabels=["Not duplicate", "Duplicate"], ax=ax)
        ax.set(xlabel="Predicted", ylabel="Actual", title=f"{m['display_name']} (test, threshold {m['threshold']:.3f})")
        fig.tight_layout(); fig.savefig(config.FIGURES_DIR / f"confusion_{name}.png", dpi=150); plt.close(fig)
    if ok:
        fig, ax = plt.subplots(figsize=(5.5, 4))
        for name, m in ok.items():
            ax.plot([p["recall"] for p in m["pr_curve"]], [p["precision"] for p in m["pr_curve"]],
                    color=colors[name], label=f"{m['display_name']} (AP {m['test']['average_precision']:.3f})")
        ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test)"); ax.legend()
        fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "pr_curves.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    run()
