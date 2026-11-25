import numpy as np
import pytest
from scipy import sparse

from src.data_loader import load_complaints, load_pairs
from src.preprocessing import clean_text, normalize_for_tfidf, validate_complaint
from src.sentence_bert_model import SentenceBertModel, get_embeddings, load_embeddings
from src.similarity import cosine_scores, paired_cosine, predict, select_threshold, top_k
from src.tfidf_model import TfidfModel

DOCS = ["Garbage has not been collected in our street for three days",
        "Street lights are not working near the temple",
        "There is no water supply in our society since yesterday",
        "Huge pothole on the main road causing accidents"]


# ------------------------------------------------------------- dataset
def test_dataset_loads_and_has_expected_columns():
    c, p = load_complaints(), load_pairs()
    assert len(c) == 10_000 and len(p) == 10_000 and p.duplicate_label.sum() == 5_000
    assert {"complaint_id", "complaint_text", "category", "locality", "status"} <= set(c.columns)
    assert set(p.complaint_id_1) <= set(c.complaint_id)


def test_splits_are_leakage_free():
    seen_issue, seen_loc = set(), set()
    for s in ("train", "val", "test"):
        c = load_complaints(s)
        assert seen_issue.isdisjoint(c.issue_id) and seen_loc.isdisjoint(c.locality)
        seen_issue |= set(c.issue_id); seen_loc |= set(c.locality)


def test_bad_split_name():
    with pytest.raises(ValueError):
        load_pairs("dev")


# ------------------------------------------------------------- preprocessing
def test_preprocessing_keeps_negations():
    assert clean_text("  a   b  https://x.y ") == "a b"
    out = normalize_for_tfidf("The lights are NOT working, they are OFF!").split()
    assert "not" in out and "off" in out and "the" not in out


@pytest.mark.parametrize("bad", ["", "abc", "123456", "!!!!!!", "x" * 2001])
def test_validation_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        validate_complaint(bad)


# ------------------------------------------------------------- similarity
def test_cosine_and_paired_cosine():
    m = np.array([[1.0, 0.0], [1.0, 1.0], [0.0, 2.0]])
    s = cosine_scores(np.array([[2.0, 0.0]]), m)
    assert np.allclose(s, [1.0, 1 / np.sqrt(2), 0.0])
    assert np.allclose(cosine_scores(sparse.csr_matrix([[2.0, 0.0]]), sparse.csr_matrix(m)), s)
    assert np.allclose(paired_cosine(m, m), 1.0)


def test_top_k_order_and_mask():
    s = np.array([0.2, 0.9, 0.5, 0.7])
    assert [i for i, _ in top_k(s, 2)] == [1, 3]
    assert [i for i, _ in top_k(s, 5, mask=np.array([True, False, True, False]))] == [2, 0]
    assert top_k(s, 3, mask=np.zeros(4, dtype=bool)) == []


def test_threshold_selection_and_prediction():
    scores, labels = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9]), np.array([0, 0, 0, 1, 1, 1])
    t, f1 = select_threshold(scores, labels)
    assert 0.3 < t < 0.7 and f1 == 1.0
    assert predict(scores, t).tolist() == labels.tolist()


# ------------------------------------------------------------- models
def test_tfidf_ranks_shared_words_first():
    m = TfidfModel().fit(DOCS)
    s = cosine_scores(m.encode(["garbage not collected in the street"]), m.encode(DOCS))
    assert s.argmax() == 0 and m.vocabulary_size > 0
    with pytest.raises(RuntimeError):
        TfidfModel().encode(["x"])
    with pytest.raises(ValueError):
        TfidfModel().fit([])


def test_sbert_wrapper_and_embedding_files(fake_encoder, tmp_path):
    m = SentenceBertModel(encoder=fake_encoder)
    v = m.encode(DOCS)
    assert v.shape == (4, 64) and np.allclose(np.linalg.norm(v, axis=1), 1)
    ids = [1, 2, 3, 4]
    saved = get_embeddings(m, "t", ids, DOCS, folder=tmp_path)
    assert np.allclose(load_embeddings("t", ids, DOCS, m.model_name, tmp_path), saved)
    assert load_embeddings("t", ids, DOCS[:3] + ["changed"], m.model_name, tmp_path) is None   # stale


def test_real_sbert_if_installed():
    """Runs only where sentence-transformers and the downloaded model are available."""
    pytest.importorskip("sentence_transformers")
    from src.sentence_bert_model import SbertUnavailable
    m = SentenceBertModel()
    try:
        v = m.encode(["Garbage has not been collected from my street for three days",
                      "The waste collection vehicle has not visited our locality since Monday",
                      "The streetlight near the park is not working"])
    except SbertUnavailable as e:
        pytest.skip(str(e))
    assert v.shape == (3, 384)
    assert v[0] @ v[1] > v[0] @ v[2]          # paraphrase closer than unrelated complaint
