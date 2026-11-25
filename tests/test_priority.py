import sqlite3

import pytest

from src import database
from src.priority import duration_days, level_for, score_complaint, with_reports

SEWAGE = "Sewage is overflowing on our road"
SEWAGE_BAD = ("Sewage is overflowing on our road, there is a very bad smell and people have difficulty in "
              "walking. Children go to school through it. This has been going on for 2 weeks")


def groups(p):
    return {r.split(" +")[0] for r in p.reasons}


def test_users_example_is_critical_and_explained():
    p = score_complaint(SEWAGE_BAD, "Sewage Overflow")
    assert p.level == "Critical" and p.score >= 75
    assert {"Health risk", "Daily life disrupted", "Vulnerable people affected"} <= groups(p)
    assert "bad smell" in p.reason_text and "difficulty in walking" in p.reason_text


def test_more_risk_words_never_lower_the_score():
    assert score_complaint(SEWAGE_BAD, "Sewage Overflow").score > score_complaint(SEWAGE, "Sewage Overflow").score


def test_hinglish_words_are_recognised():
    p = score_complaint("gutter ka ganda paani 5 din se, badbu se bachche bimar ho rahe", "Sewage Overflow")
    assert {"Health risk", "Vulnerable people affected"} <= groups(p)
    assert "Lasting about 5 days +4" in p.reason_text


def test_safety_hazard_near_children_is_critical():
    p = score_complaint("Open manhole near the school, someone could fall into it", "Drainage Problems")
    assert p.level == "Critical" and any("vulnerable" in r.lower() or "children" in r for r in p.reasons)


def test_minor_problems_stay_low():
    assert score_complaint("Street lights stay on during the day", "Streetlight Problems").level == "Low"
    assert score_complaint("Swings in the park are broken", "Public Park Maintenance").level == "Low"


def test_repeating_a_word_does_not_inflate_the_score():
    one = score_complaint("bad smell near the drain", "Drainage Problems").score
    many = score_complaint("bad smell, smell, smelly, stinking smell near the drain", "Drainage Problems").score
    assert one == many


def test_whole_word_matching():
    # "fire" must not match inside "firewood"; "kids" must not match inside "skids"
    assert "Safety risk" not in groups(score_complaint("A heap of firewood on the footpath", "Footpath Damage"))
    assert "Vulnerable people affected" not in groups(score_complaint("The van skidsx here", "Road Damage"))


@pytest.mark.parametrize("text,days", [("for 2 weeks", 14), ("5 din se", 5), ("for the last fortnight", 14),
                                       ("since yesterday", 1), ("for three days", 3), ("ek hafte se", 7),
                                       ("no time mentioned", 0)])
def test_duration_parsing(text, days):
    assert duration_days(text) == days


def test_levels_bonus_and_caps():
    assert [level_for(x) for x in (80, 60, 40, 10)] == ["Critical", "High", "Medium", "Low"]
    assert with_reports(50, 1) == 55 and with_reports(50, 10) == 65 and with_reports(95, 3) == 100
    assert score_complaint(SEWAGE_BAD + " urgent emergency, dangerous, accident, again", "Sewage Overflow",
                           extra_reports=5).score == 100


def test_old_database_is_upgraded_and_scored(tmp_path):
    path = tmp_path / "old.db"
    raw = sqlite3.connect(path)
    raw.execute("CREATE TABLE complaints (id INTEGER PRIMARY KEY AUTOINCREMENT, complaint_reference TEXT UNIQUE, "
                "complaint_text TEXT, category TEXT, locality TEXT, status TEXT, created_at TEXT, data_origin TEXT)")
    raw.execute("INSERT INTO complaints VALUES (1, 'OLD1', ?, 'Sewage Overflow', 'X', 'Open', '2026-01-01', 'user')",
                (SEWAGE_BAD,))
    raw.commit(); raw.close()
    conn = database.connect(path)
    database.create_tables(conn)
    row = database.get_complaint(conn, 1)
    assert row["priority_level"] == "Critical" and "bad smell" in row["priority_reasons"]


def test_new_complaints_are_scored_and_queue_is_ordered(db):
    urgent = database.add_complaint(db, SEWAGE_BAD, "Sewage Overflow", "Sai Nagar")
    minor = database.add_complaint(db, "Swings in the park are broken", "Public Park Maintenance", "Sai Nagar")
    assert urgent["priority_level"] == "Critical" and minor["priority_level"] == "Low"
    q = database.priority_queue(db)
    refs = q.complaint_reference.tolist()
    assert refs.index(urgent["complaint_reference"]) < refs.index(minor["complaint_reference"])
    assert q.final_score.is_monotonic_decreasing


def test_linked_reports_raise_priority_but_rejected_do_not(db):
    target = database.add_complaint(db, "Swings in the park are broken", "Public Park Maintenance", "Sai Nagar")
    others = [database.add_complaint(db, f"Park swings broken, report {i}", "Public Park Maintenance", "Sai Nagar")
              for i in range(3)]
    for o in others:
        database.add_matches(db, o["id"], [{"id": target["id"], "score": 0.9}], "tfidf")
    score = lambda: database.priority_queue(db).set_index("id").loc[target["id"]]
    assert score().extra_reports == 3 and score().final_score == target["priority_score"] + 15
    m = database.matches_table(db)
    database.set_review(db, int(m[m.matched_id == target["id"]].iloc[0].id), "rejected")
    assert score().extra_reports == 2 and "Reported by 3 complaints" in score().priority_reasons


def test_solved_complaints_leave_the_queue(db):
    c = database.add_complaint(db, SEWAGE_BAD, "Sewage Overflow", "Sai Nagar")
    database.update_status(db, c["id"], "Resolved")
    assert c["id"] not in database.priority_queue(db).id.tolist()
