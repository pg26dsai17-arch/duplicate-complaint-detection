"""The duplicate detection engine used by the Streamlit pages.

It keeps, in memory, one vector per stored complaint for each model:
* TF-IDF: vocabulary fitted on all stored complaints when the engine is built.
* Sentence-BERT: embeddings loaded from data/embeddings/ (built and saved if missing or outdated).

check() turns a new complaint into a vector, scores it against every stored vector with
cosine similarity, and returns the Top-K. A match at or above the threshold is a POTENTIAL
duplicate: a person still decides.
"""
import json
import time

import numpy as np
from scipy import sparse

from src import config
from src.sentence_bert_model import SbertUnavailable, SentenceBertModel, get_embeddings, save_embeddings
from src.similarity import cosine_scores, top_k
from src.tfidf_model import TfidfModel

APP_EMBEDDINGS = "app_sbert"


def validated_thresholds() -> dict:
    """Thresholds chosen on validation data by the evaluation script, if it has been run."""
    out = {}
    if config.METRICS_FILE.exists():
        for name, m in json.loads(config.METRICS_FILE.read_text()).get("models", {}).items():
            if m.get("status") == "ok":
                out[name] = {"text": m["threshold"], "locality": m.get("locality_rule", {}).get("threshold")}
    return out


class DuplicateDetector:
    def __init__(self, corpus_df, enable_sbert: bool = config.ENABLE_SBERT, sbert_encoder=None):
        """corpus_df: DataFrame with columns id, complaint_text, locality (from database.corpus)."""
        self.ids = corpus_df["id"].astype(int).tolist()
        self.texts = corpus_df["complaint_text"].tolist()
        self.localities = np.array(corpus_df["locality"].tolist(), dtype=object)
        self.vectors, self.models, self.unavailable = {}, {}, {}

        self.tfidf = TfidfModel()
        if self.texts:
            self.tfidf.fit(self.texts)
            self.vectors["tfidf"] = self.tfidf.encode(self.texts)
        self.models["tfidf"] = self.tfidf

        if enable_sbert:
            sb = SentenceBertModel(encoder=sbert_encoder)
            try:
                if self.texts:
                    self.vectors["sbert"] = get_embeddings(sb, APP_EMBEDDINGS, self.ids, self.texts)
                else:
                    sb.load()
                self.models["sbert"] = sb
            except SbertUnavailable as e:
                self.unavailable["sbert"] = str(e)
        else:
            self.unavailable["sbert"] = "Sentence-BERT is switched off (ENABLE_SBERT=0)."

        found = validated_thresholds()
        self.thresholds = {}
        for name in ("tfidf", "sbert"):
            t = found.get(name)
            self.thresholds[name] = {
                "text": t["text"] if t else config.FALLBACK_THRESHOLDS[name],
                "locality": (t["locality"] if t and t["locality"] is not None else
                             (t["text"] if t else config.FALLBACK_THRESHOLDS[name])),
                "source": "chosen on validation data" if t else "default, not validated yet",
            }

    @property
    def available(self) -> list[str]:
        return list(self.models)

    @property
    def default_model(self) -> str:
        return "sbert" if "sbert" in self.models else "tfidf"

    def threshold(self, model: str, same_locality: bool) -> float:
        return self.thresholds[model]["locality" if same_locality else "text"]

    def check(self, text: str, model: str | None = None, k: int = config.TOP_K, locality: str | None = None,
              threshold: float | None = None, exclude_id: int | None = None) -> dict:
        model = model or self.default_model
        if model not in self.models:
            raise SbertUnavailable(self.unavailable.get(model, f"Unknown model {model}"))
        thr = self.threshold(model, bool(locality)) if threshold is None else float(threshold)
        t0 = time.perf_counter()
        matches = []
        if self.ids:
            scores = cosine_scores(self.models[model].encode([text]), self.vectors[model])
            mask = np.ones(len(self.ids), dtype=bool)
            if locality:
                mask &= self.localities == locality
            if exclude_id is not None and exclude_id in self.ids:
                mask[self.ids.index(exclude_id)] = False
            matches = [{"id": self.ids[i], "score": s, "rank": r + 1, "is_duplicate": s >= thr}
                       for r, (i, s) in enumerate(top_k(scores, k, mask))]
        return {"model": model, "threshold": thr, "matches": matches, "locality": locality,
                "seconds": time.perf_counter() - t0, "corpus_size": len(self.ids),
                "potential_duplicate": any(m["is_duplicate"] for m in matches)}

    def add(self, complaint_id: int, text: str, locality: str) -> None:
        """Add a newly saved complaint so it can be matched immediately."""
        self.ids.append(int(complaint_id)); self.texts.append(text)
        self.localities = np.append(self.localities, locality)
        if "tfidf" not in self.vectors:                     # first ever complaint
            self.tfidf.fit(self.texts)
            self.vectors["tfidf"] = self.tfidf.encode(self.texts)
        else:
            self.vectors["tfidf"] = sparse.vstack([self.vectors["tfidf"], self.tfidf.encode([text])]).tocsr()
        if "sbert" in self.models:
            v = self.models["sbert"].encode([text])
            self.vectors["sbert"] = v if "sbert" not in self.vectors else np.vstack([self.vectors["sbert"], v])
            save_embeddings(APP_EMBEDDINGS, self.ids, self.texts, self.vectors["sbert"],
                            self.models["sbert"].model_name)
