"""One-time setup. Run from the project folder:

    python setup_project.py

Steps (each is safe to run again):
  1. Check the dataset files.
  2. Create data/complaints.db and its tables.
  3. Load the 10,000 synthetic complaints (skipped if already loaded).
  4. Download Sentence-BERT once and save embeddings to data/embeddings/.
  5. Evaluate TF-IDF and Sentence-BERT and save results/metrics.json.

Options:  --skip-sbert   use TF-IDF only (no download needed)
          --skip-eval    skip step 5
"""
import argparse
import sys
import time

from src import config, database
from src.data_loader import DatasetMissing, load_complaints, load_pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-sbert", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args(argv)

    print("1/5 Checking dataset files ...")
    try:
        c, p = load_complaints(), load_pairs()
        for s in ("train", "val", "test"):
            load_complaints(s), load_pairs(s)
    except DatasetMissing as e:
        sys.exit(f"   ERROR: {e}")
    print(f"   {len(c):,} complaints, {len(p):,} pairs "
          f"({int(p.duplicate_label.sum()):,} duplicate). Data origin: synthetic.")

    print(f"2/5 Creating database {config.DB_PATH} ...")
    conn = database.connect()
    database.create_tables(conn)

    print("3/5 Loading synthetic complaints into the database ...")
    added = database.load_synthetic_dataset(conn)
    print(f"   added {added:,} new rows; database now holds {database.count_complaints(conn):,} complaints.")

    if args.skip_sbert:
        print("4/5 Skipped Sentence-BERT (--skip-sbert).")
    else:
        print("4/5 Building Sentence-BERT embeddings (first run downloads ~90 MB, then a few minutes on CPU) ...")
        from src.detector import DuplicateDetector
        t0 = time.perf_counter()
        det = DuplicateDetector(database.corpus(conn), enable_sbert=True)
        if "sbert" in det.models:
            print(f"   saved {len(det.ids):,} embeddings in {time.perf_counter() - t0:.0f}s to {config.EMBEDDING_DIR}")
        else:
            print(f"   WARNING: {det.unavailable['sbert']}\n   The app will still work with TF-IDF.")

    if args.skip_eval:
        print("5/5 Skipped evaluation (--skip-eval).")
    else:
        print("5/5 Evaluating models (threshold on validation, metrics on test) ...")
        from src.evaluation import run
        run(models=("tfidf",) if args.skip_sbert else ("tfidf", "sbert"))

    print("\nDone. Start the app with:  streamlit run app.py")


if __name__ == "__main__":
    main()
