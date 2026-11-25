"""Read the synthetic dataset CSV files.

complaints.csv columns: complaint_id, complaint_text, category, subcategory, locality, ward, zone,
submission_date, status, data_origin, issue_id, subcategory_key, writing_style, split
complaint_pairs.csv columns: pair_id, complaint_id_1, complaint_id_2, complaint_1, complaint_2,
category_1, category_2, duplicate_label, data_origin, pair_type, split
"""
import pandas as pd

from src.config import RAW_DIR, SPLIT_DIR

SPLITS = ("train", "val", "test")


class DatasetMissing(FileNotFoundError):
    pass


def _read(path):
    if not path.exists():
        raise DatasetMissing(f"Dataset file not found: {path}. Put the CSV files in data/raw and data/splits.")
    return pd.read_csv(path)


def load_complaints(split: str | None = None) -> pd.DataFrame:
    if split is None:
        return _read(RAW_DIR / "complaints.csv")
    _check(split)
    return _read(SPLIT_DIR / f"{split}_complaints.csv")


def load_pairs(split: str | None = None) -> pd.DataFrame:
    if split is None:
        return _read(RAW_DIR / "complaint_pairs.csv")
    _check(split)
    return _read(SPLIT_DIR / f"{split}_pairs.csv")


def _check(split):
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
