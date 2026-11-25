"""Model B: pretrained Sentence-BERT (all-MiniLM-L6-v2) + cosine similarity.

The model reads a sentence with a 6-layer transformer, averages the token vectors
(mean pooling) and returns one 384-number vector per complaint. It was trained so that
sentences with the same meaning get nearby vectors. We use it as-is (no fine-tuning).

The weights are downloaded once by sentence-transformers and cached on disk
(in %USERPROFILE%\\.cache\\huggingface on Windows), so later runs work offline.
"""
import hashlib
import json

import numpy as np

from src.config import EMBEDDING_DIR, SBERT_MODEL_NAME
from src.preprocessing import clean_text


class SbertUnavailable(RuntimeError):
    pass


class SentenceBertModel:
    name = "sbert"
    display_name = "Sentence-BERT"

    def __init__(self, model_name: str = SBERT_MODEL_NAME, encoder=None, batch_size: int = 64):
        """`encoder` lets tests pass any object with .encode(list_of_texts) -> array."""
        self.model_name = model_name
        self.batch_size = batch_size
        self._encoder = encoder

    def fit(self, texts):            # nothing to learn: the model is pretrained
        return self

    def load(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise SbertUnavailable("The sentence-transformers package is not installed. "
                                       "Run: pip install -r requirements.txt.") from e
            try:
                self._encoder = SentenceTransformer(self.model_name)
            except Exception as e:
                raise SbertUnavailable(f"Could not load the model '{self.model_name}'. The first run needs "
                                       f"internet to download it (about 90 MB). Details: {e}") from e
        return self._encoder

    def encode(self, texts) -> np.ndarray:
        texts = [clean_text(t) for t in texts]
        vecs = self.load().encode(texts, batch_size=self.batch_size, show_progress_bar=False,
                                  convert_to_numpy=True, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype=np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------- embedding files
def fingerprint(ids, texts) -> str:
    """Changes if any complaint id or text changes, so stale embeddings are detected."""
    h = hashlib.sha256()
    for i, t in zip(ids, texts):
        h.update(f"{i}\x1f{t}\x1e".encode())
    return h.hexdigest()


def save_embeddings(name, ids, texts, vecs, model_name, folder=None):
    folder = folder or EMBEDDING_DIR
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / f"{name}.npy", vecs.astype(np.float32))
    (folder / f"{name}.json").write_text(json.dumps(
        {"model": model_name, "count": len(ids), "fingerprint": fingerprint(ids, texts)}))


def load_embeddings(name, ids, texts, model_name, folder=None):
    """Return saved vectors if they match these exact complaints and model, else None."""
    folder = folder or EMBEDDING_DIR
    vec_file, meta_file = folder / f"{name}.npy", folder / f"{name}.json"
    if not (vec_file.exists() and meta_file.exists()):
        return None
    meta = json.loads(meta_file.read_text())
    if meta.get("model") != model_name or meta.get("fingerprint") != fingerprint(ids, texts):
        return None
    return np.load(vec_file)


def get_embeddings(model: SentenceBertModel, name, ids, texts, folder=None):
    vecs = load_embeddings(name, ids, texts, model.model_name, folder)
    if vecs is None:
        vecs = model.encode(list(texts))
        save_embeddings(name, ids, texts, vecs, model.model_name, folder)
    return vecs
