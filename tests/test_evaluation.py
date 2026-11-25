import json

import numpy as np

from src import config, evaluation
from src.sentence_bert_model import SentenceBertModel
from src.tfidf_model import TfidfModel


def _check(r):
    t = r["test"]
    assert all(0 <= t[k] <= 1 for k in ("accuracy", "precision", "recall", "f1", "roc_auc"))
    cm = t["confusion_matrix"]
    assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == 1500
    assert 0 <= r["retrieval"]["recall@1"] <= r["retrieval"]["recall@5"] <= 1
    assert sum(r["score_hist"]["duplicate"]) == 750 and r["speed_ms"]["mean"] > 0


def test_tfidf_evaluation_reproduces_reported_numbers():
    r = evaluation.evaluate_model(TfidfModel(), evaluation.load_data())
    _check(r)
    assert np.isclose(r["threshold"], 0.0376, atol=1e-3) and np.isclose(r["test"]["f1"], 0.744, atol=1e-3)


def test_sbert_evaluation_code_path_runs(fake_encoder):
    _check(evaluation.evaluate_model(SentenceBertModel(encoder=fake_encoder), evaluation.load_data(),
                                     cache_name="test_eval"))


def test_run_writes_metrics_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "METRICS_FILE", tmp_path / "metrics.json")
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(config, "FIGURES_DIR", tmp_path / "fig")
    evaluation.run(models=("tfidf",))
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["models"]["tfidf"]["status"] == "ok" and (tmp_path / "fig" / "confusion_tfidf.png").exists()
