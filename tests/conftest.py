"""Tests use a temporary database and embedding folder, and never touch data/complaints.db
or results/metrics.json. Sentence-BERT is replaced by a small stand-in encoder where needed."""
import os
import sys
import tempfile
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TMP = Path(tempfile.mkdtemp(prefix="complaints-tests-"))
os.environ["COMPLAINTS_DB"] = str(TMP / "test.db")
os.environ["ENABLE_SBERT"] = "0"

import src.sentence_bert_model as sbm  # noqa: E402

sbm.EMBEDDING_DIR = TMP / "embeddings"


class FakeEncoder:
    """Hashed bag of words in 64 dimensions. Checks plumbing only; it is NOT a semantic model
    and is never used for any reported result."""

    def encode(self, texts, **kwargs):
        out = np.full((len(texts), 64), 1e-3, dtype=np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, zlib.crc32(w.strip(".,!").encode()) % 64] += 1
        return out


@pytest.fixture
def fake_encoder():
    return FakeEncoder()


@pytest.fixture
def db(tmp_path):
    from src import database
    conn = database.connect(tmp_path / "unit.db")
    database.create_tables(conn)
    database.load_synthetic_dataset(conn, limit=300)
    return conn


@pytest.fixture(scope="session", autouse=True)
def app_database():
    """The database the Streamlit pages use during AppTest runs."""
    from src import database
    conn = database.connect()
    database.create_tables(conn)
    database.load_synthetic_dataset(conn, limit=800)
    conn.close()
