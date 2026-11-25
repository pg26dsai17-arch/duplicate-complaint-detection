"""SQLite storage using Python's built-in sqlite3 module.

Tables are created automatically (create_tables). The synthetic dataset is loaded once by
load_synthetic_dataset(); because complaint_reference is UNIQUE and rows are inserted with
INSERT OR IGNORE, running it again never creates duplicates.
"""
import secrets
import sqlite3
import string
from datetime import datetime
from pathlib import Path

import pandas as pd

from src import config
from src.priority import level_for, score_complaint, with_reports

SCHEMA = """
CREATE TABLE IF NOT EXISTS complaints (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_reference TEXT NOT NULL UNIQUE,
    complaint_text      TEXT NOT NULL,
    category            TEXT NOT NULL,
    locality            TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'Open'
                        CHECK (status IN ('Open', 'In Progress', 'Resolved', 'Closed')),
    created_at          TEXT NOT NULL,
    data_origin         TEXT NOT NULL DEFAULT 'user',     -- 'synthetic' or 'user'
    priority_score      INTEGER,                          -- 0-100, from src/priority.py
    priority_level      TEXT,                             -- Critical, High, Medium, Low
    priority_reasons    TEXT                              -- why, in words
);
CREATE INDEX IF NOT EXISTS idx_complaints_category ON complaints(category);
CREATE INDEX IF NOT EXISTS idx_complaints_locality ON complaints(locality);
CREATE INDEX IF NOT EXISTS idx_complaints_status   ON complaints(status);

CREATE TABLE IF NOT EXISTS duplicate_matches (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id         INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
    matched_complaint_id INTEGER NOT NULL REFERENCES complaints(id) ON DELETE CASCADE,
    similarity_score     REAL NOT NULL,
    model_name           TEXT NOT NULL,
    review_status        TEXT NOT NULL DEFAULT 'pending'
                         CHECK (review_status IN ('pending', 'confirmed', 'rejected')),
    created_at           TEXT NOT NULL,
    UNIQUE (complaint_id, matched_complaint_id, model_name)
);
CREATE INDEX IF NOT EXISTS idx_matches_review ON duplicate_matches(review_status);
"""


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    path = Path(db_path or config.DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_tables(conn) -> None:
    conn.executescript(SCHEMA)
    upgrade_schema(conn)
    conn.commit()


PRIORITY_COLUMNS = {"priority_score": "INTEGER", "priority_level": "TEXT", "priority_reasons": "TEXT"}


def upgrade_schema(conn) -> None:
    """Add the priority columns to a database created before they existed, then score any
    complaints that have no priority yet. Safe to run every time (does nothing if up to date)."""
    have = {r[1] for r in conn.execute("PRAGMA table_info(complaints)")}
    for col, typ in PRIORITY_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE complaints ADD COLUMN {col} {typ}")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_complaints_priority ON complaints(priority_score)")
    backfill_priorities(conn)


def backfill_priorities(conn) -> int:
    rows = conn.execute("SELECT id, complaint_text, category FROM complaints WHERE priority_score IS NULL").fetchall()
    updates = []
    for r in rows:
        p = score_complaint(r["complaint_text"], r["category"])
        updates.append((p.score, p.level, p.reason_text, r["id"]))
    conn.executemany("UPDATE complaints SET priority_score = ?, priority_level = ?, priority_reasons = ? WHERE id = ?",
                     updates)
    conn.commit()
    return len(updates)


def count_complaints(conn) -> int:
    return conn.execute("SELECT COUNT(*) FROM complaints").fetchone()[0]


def load_synthetic_dataset(conn, csv_path: Path | None = None, limit: int | None = None) -> int:
    """Insert data/raw/complaints.csv. Returns the number of NEW rows added."""
    df = pd.read_csv(csv_path or config.RAW_DIR / "complaints.csv")
    if limit:
        df = df.head(limit)
    before = count_complaints(conn)
    conn.executemany(
        "INSERT OR IGNORE INTO complaints (complaint_reference, complaint_text, category, locality, "
        "status, created_at, data_origin) VALUES (?, ?, ?, ?, ?, ?, 'synthetic')",
        df[["complaint_id", "complaint_text", "category", "locality", "status", "submission_date"]]
        .itertuples(index=False, name=None))
    conn.commit()
    backfill_priorities(conn)
    return count_complaints(conn) - before


def new_reference(conn) -> str:
    """Readable unique reference, e.g. CD-20261117-K7P3QX."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    while True:
        ref = f"CD-{datetime.now():%Y%m%d}-" + "".join(secrets.choice(alphabet) for _ in range(6))
        if not conn.execute("SELECT 1 FROM complaints WHERE complaint_reference = ?", (ref,)).fetchone():
            return ref


def add_complaint(conn, text: str, category: str, locality: str) -> dict:
    ref = new_reference(conn)
    now = datetime.now().isoformat(timespec="seconds")
    p = score_complaint(text, category)
    cur = conn.execute(
        "INSERT INTO complaints (complaint_reference, complaint_text, category, locality, status, created_at, "
        "data_origin, priority_score, priority_level, priority_reasons) VALUES (?, ?, ?, ?, 'Open', ?, 'user', ?, ?, ?)",
        (ref, text, category, locality, now, p.score, p.level, p.reason_text))
    conn.commit()
    return get_complaint(conn, cur.lastrowid)


def add_matches(conn, complaint_id: int, matches: list[dict], model_name: str) -> int:
    now = datetime.now().isoformat(timespec="seconds")
    conn.executemany(
        "INSERT OR IGNORE INTO duplicate_matches (complaint_id, matched_complaint_id, similarity_score, "
        "model_name, review_status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
        [(complaint_id, m["id"], round(m["score"], 4), model_name, now) for m in matches])
    conn.commit()
    return len(matches)


def get_complaint(conn, complaint_id: int) -> dict | None:
    row = conn.execute("SELECT * FROM complaints WHERE id = ?", (complaint_id,)).fetchone()
    return dict(row) if row else None


def get_by_reference(conn, reference: str) -> dict | None:
    row = conn.execute("SELECT * FROM complaints WHERE complaint_reference = ?", (reference.strip().upper(),)).fetchone()
    return dict(row) if row else None


def all_complaints(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM complaints ORDER BY created_at DESC, id DESC", conn)


def corpus(conn) -> pd.DataFrame:
    """id, text and locality of every complaint, in id order (used to build the search index)."""
    return pd.read_sql_query("SELECT id, complaint_text, locality FROM complaints ORDER BY id", conn)


def update_status(conn, complaint_id: int, status: str) -> None:
    if status not in config.STATUSES:
        raise ValueError(f"Unknown status: {status}")
    conn.execute("UPDATE complaints SET status = ? WHERE id = ?", (status, complaint_id))
    conn.commit()


def matches_table(conn, review_status: str | None = None) -> pd.DataFrame:
    sql = """
        SELECT m.id, m.similarity_score, m.model_name, m.review_status, m.created_at,
               c.id  AS complaint_id,  c.complaint_reference AS new_reference,  c.complaint_text AS new_text,
               c.locality AS new_locality,
               o.id  AS matched_id,    o.complaint_reference AS matched_reference, o.complaint_text AS matched_text,
               o.locality AS matched_locality, o.status AS matched_status
        FROM duplicate_matches m
        JOIN complaints c ON c.id = m.complaint_id
        JOIN complaints o ON o.id = m.matched_complaint_id
    """
    params = ()
    if review_status:
        sql += " WHERE m.review_status = ?"
        params = (review_status,)
    sql += " ORDER BY m.complaint_id DESC, m.similarity_score DESC"
    return pd.read_sql_query(sql, conn, params=params)


def set_review(conn, match_id: int, decision: str) -> None:
    if decision not in ("pending", "confirmed", "rejected"):
        raise ValueError(f"Unknown decision: {decision}")
    conn.execute("UPDATE duplicate_matches SET review_status = ? WHERE id = ?", (decision, match_id))
    conn.commit()


def complaints_by_ids(conn, ids: list[int]) -> dict[int, dict]:
    if not ids:
        return {}
    rows = conn.execute(f"SELECT * FROM complaints WHERE id IN ({','.join('?' * len(ids))})", ids).fetchall()
    return {r["id"]: dict(r) for r in rows}


def localities(conn) -> list[str]:
    return [r[0] for r in conn.execute("SELECT DISTINCT locality FROM complaints ORDER BY locality")]


def priority_queue(conn, statuses=("Open", "In Progress")) -> pd.DataFrame:
    """Unsolved complaints, most urgent first.

    final_score = text-based priority + a bonus for every other complaint linked to this one as a
    duplicate (pending or confirmed; rejected links do not count). Ties: the older complaint first.
    """
    df = pd.read_sql_query(f"""
        SELECT c.*, COUNT(m.id) AS extra_reports
        FROM complaints c
        LEFT JOIN duplicate_matches m
               ON m.matched_complaint_id = c.id AND m.review_status != 'rejected'
        WHERE c.status IN ({",".join("?" * len(statuses))})
        GROUP BY c.id""", conn, params=list(statuses))
    if df.empty:
        return df.assign(final_score=[], final_level=[])
    df["final_score"] = [with_reports(s or 0, r) for s, r in zip(df.priority_score, df.extra_reports)]
    df["final_level"] = df.final_score.map(level_for)
    df["priority_reasons"] = [reasons + (f"; Reported by {r + 1} complaints +{min(15, 5 * r)}" if r else "")
                              for reasons, r in zip(df.priority_reasons.fillna(""), df.extra_reports)]
    return df.sort_values(["final_score", "created_at"], ascending=[False, True]).reset_index(drop=True)
