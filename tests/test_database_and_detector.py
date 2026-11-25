from src import database
from src.detector import DuplicateDetector


def test_database_creation_and_no_double_loading(db):
    assert database.count_complaints(db) == 300
    assert database.load_synthetic_dataset(db, limit=300) == 0          # second load adds nothing
    row = db.execute("SELECT data_origin FROM complaints LIMIT 1").fetchone()
    assert row[0] == "synthetic"


def test_add_complaint_reference_status_and_matches(db):
    c = database.add_complaint(db, "Water pipe leaking near the school", "Water Leakage", "Sai Nagar")
    assert c["complaint_reference"].startswith("CD-") and c["status"] == "Open" and c["data_origin"] == "user"
    assert database.get_by_reference(db, c["complaint_reference"].lower())["id"] == c["id"]
    database.add_matches(db, c["id"], [{"id": 1, "score": 0.81}], "tfidf")
    m = database.matches_table(db, "pending")
    assert len(m) == 1 and m.iloc[0].matched_id == 1
    database.set_review(db, int(m.iloc[0].id), "confirmed")
    assert database.matches_table(db, "confirmed").shape[0] == 1
    database.update_status(db, c["id"], "Resolved")
    assert database.get_complaint(db, c["id"])["status"] == "Resolved"


def test_detector_tfidf_top5_threshold_and_locality(db):
    det = DuplicateDetector(database.corpus(db), enable_sbert=False)
    assert det.available == ["tfidf"] and "sbert" in det.unavailable
    first = database.get_complaint(db, 1)
    r = det.check(first["complaint_text"], k=5)
    assert len(r["matches"]) == 5 and r["matches"][0]["id"] == 1 and r["matches"][0]["score"] > 0.99
    assert [m["rank"] for m in r["matches"]] == [1, 2, 3, 4, 5]
    assert r["matches"][0]["is_duplicate"]
    r = det.check(first["complaint_text"], k=5, threshold=1.01)
    assert not r["potential_duplicate"]
    r = det.check(first["complaint_text"], k=5, locality=first["locality"], exclude_id=1)
    rows = database.complaints_by_ids(db, [m["id"] for m in r["matches"]])
    assert all(rows[m["id"]]["locality"] == first["locality"] for m in r["matches"]) and 1 not in rows


def test_detector_with_stand_in_sbert_and_add(db, fake_encoder):
    det = DuplicateDetector(database.corpus(db), enable_sbert=True, sbert_encoder=fake_encoder)
    assert det.default_model == "sbert"
    c = database.add_complaint(db, "Broken bench in the children's park", "Public Park Maintenance", "Test Area")
    det.add(c["id"], c["complaint_text"], "Test Area")
    for model in ("tfidf", "sbert"):
        r = det.check("broken bench in park", model=model, k=1, locality="Test Area")
        assert r["matches"][0]["id"] == c["id"]


def test_detector_handles_empty_database(tmp_path):
    conn = database.connect(tmp_path / "empty.db")
    database.create_tables(conn)
    det = DuplicateDetector(database.corpus(conn), enable_sbert=False)
    assert det.check("water not coming")["matches"] == []
