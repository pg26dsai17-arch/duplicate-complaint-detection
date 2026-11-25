"""All paths and settings in one place."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SPLIT_DIR = DATA_DIR / "splits"
EMBEDDING_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_FILE = RESULTS_DIR / "metrics.json"

# Environment variables let tests use a temporary database and switch SBERT off.
DB_PATH = Path(os.getenv("COMPLAINTS_DB", DATA_DIR / "complaints.db"))
SBERT_MODEL_NAME = os.getenv("SBERT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
ENABLE_SBERT = os.getenv("ENABLE_SBERT", "1") != "0"
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")      # optional: protects the Admin page

TOP_K = 5
# Used only until `python setup_project.py` has produced validated thresholds.
FALLBACK_THRESHOLDS = {"tfidf": 0.30, "sbert": 0.60}

CATEGORIES = [
    "Water Supply", "Garbage Collection", "Road Damage", "Potholes", "Drainage Problems",
    "Sewage Overflow", "Streetlight Problems", "Mosquito Control", "Public Toilets",
    "Stormwater Drainage", "Illegal Encroachment", "Footpath Damage", "Dead Animal Removal",
    "Public Health and Sanitation", "Water Leakage", "Stray Animal Problems",
    "Traffic and Road Obstruction", "Public Park Maintenance", "Construction Waste",
    "Other Municipal Issues",
]
STATUSES = ["Open", "In Progress", "Resolved", "Closed"]
