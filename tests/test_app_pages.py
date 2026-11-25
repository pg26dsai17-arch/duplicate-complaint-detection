"""Run every Streamlit page with Streamlit's own test runner (no browser needed)."""
import pytest
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def app(path):
    return AppTest.from_file(str(ROOT / path), default_timeout=60)

PAGES = ["app_pages/1_Home.py", "app_pages/2_Submit_Complaint.py", "app_pages/3_Duplicate_Detection.py",
         "app_pages/4_Model_Evaluation.py", "app_pages/5_All_Complaints.py", "app_pages/6_Admin_Dashboard.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_runs_without_errors(page):
    at = app(page).run()
    assert not at.exception, at.exception


def test_app_entry_point_runs():
    at = app("app.py").run()
    assert not at.exception and at.title[0].value == "Duplicate complaint detection"


def test_detection_page_finds_matches():
    at = app("app_pages/3_Duplicate_Detection.py").run()
    at.button[0].click().run()
    assert not at.exception
    assert "Top" not in str(at.error)                    # no error shown
    assert any("potential duplicate" in w.value.lower() for w in list(at.warning) + list(at.success))


def test_submit_flow_saves_complaint():
    at = app("app_pages/2_Submit_Complaint.py").run()
    at.text_area[0].input("Nobody has picked up the household waste in our lane for four days")
    at.selectbox[0].select("Garbage Collection")
    at.selectbox[1].select_index(0)
    at.button[0].click().run()                            # check
    assert not at.exception and not at.error
    assert any("Estimated priority" in m.value for m in at.markdown)
    submit = [b for b in at.button if "submit" in b.label.lower()][0]
    submit.click().run()                                  # save
    assert not at.exception
    assert any("Complaint saved" in s.value and "Priority:" in s.value for s in at.success)


def test_submit_rejects_too_short_text():
    at = app("app_pages/2_Submit_Complaint.py").run()
    at.text_area[0].input("bad")
    at.selectbox[0].select("Garbage Collection")
    at.selectbox[1].select_index(0)
    at.button[0].click().run()
    assert any("at least" in e.value for e in at.error)
