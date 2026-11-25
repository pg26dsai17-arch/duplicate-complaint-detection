# Duplicate Complaint Detection Using Semantic Similarity and NLP

A Python-only Streamlit application that compares a new municipal complaint with stored complaints, shows the
Top-5 most similar ones with similarity scores, and warns about potential duplicates before the complaint is
saved. Two models are compared: **TF-IDF** (baseline) and **Sentence-BERT** (`all-MiniLM-L6-v2`).

> **The complaint data is synthetic.** The 10,000 pre-loaded complaints were generated for this project. They
> are not Surat Municipal Corporation data and were not written by real residents. Results on this data are
> likely better than real-world performance.

## Project structure

```
Project_NLP/
├── app.py                     start here: streamlit run app.py
├── setup_project.py           one-time setup (database, embeddings, evaluation)
├── app_pages/                 the six app pages (not "pages/": that name triggers Streamlit's old automatic mode)
│   ├── 1_Home.py
│   ├── 2_Submit_Complaint.py
│   ├── 3_Duplicate_Detection.py
│   ├── 4_Model_Evaluation.py
│   ├── 5_All_Complaints.py
│   └── 6_Admin_Dashboard.py
├── src/
│   ├── config.py              paths and settings
│   ├── data_loader.py         reads the CSV files
│   ├── preprocessing.py       text cleaning and validation
│   ├── tfidf_model.py         Model A
│   ├── sentence_bert_model.py Model B + saving embeddings
│   ├── similarity.py          cosine similarity, Top-K, threshold selection
│   ├── detector.py            the duplicate detection engine used by the pages
│   ├── evaluation.py          model comparison -> results/metrics.json
│   ├── priority.py            complaint priority score (Critical / High / Medium / Low)
│   ├── database.py            SQLite (complaints, duplicate_matches)
│   └── ui.py                  shared Streamlit helpers and styling
├── data/
│   ├── raw/                   complaints.csv, complaint_pairs.csv
│   ├── splits/                train/val/test complaints and pairs
│   ├── processed/
│   ├── embeddings/            saved Sentence-BERT vectors (created by setup)
│   ├── complaints.db          SQLite database (already loaded with the 10,000 synthetic complaints)
│   ├── DATASET_README.md
│   └── validation_report.txt
├── results/                   metrics.json and figures (measured, never typed in)
├── notebooks/                 01–05, step-by-step versions of the method
├── docs/PROJECT_REPORT.md     academic report with formulas, results and limitations
├── tests/                     automated tests
├── .streamlit/config.toml     colours and theme
├── requirements.txt
└── .gitignore
```

## Before you extract

I could not see your existing `Project_NLP` folder, so this archive was built separately.
**Extract it into a new, empty folder** (for example `Project_NLP_streamlit`) rather than over your old
project. In particular, if you extracted the earlier React/FastAPI version, its `src/config.py` differs from
this one. Copy across anything of yours you want to keep, and compare `requirements.txt` with your own.

## Setup on Windows (PowerShell, in VS Code)

You need **Python 3.10, 3.11 or 3.12** (3.11 recommended). Check with `python --version`.

Open the project folder in VS Code, then open a terminal with **Terminal → New Terminal** (it opens PowerShell).

### 1. Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

If PowerShell says running scripts is disabled, run this once in the same terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

You should now see `(venv)` at the start of the prompt. In VS Code, also choose this interpreter:
**Ctrl+Shift+P → Python: Select Interpreter → .\venv\Scripts\python.exe**.

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

The first line after the upgrade installs the CPU-only version of PyTorch, which is a much smaller download
than the default. Installation takes a few minutes.

### 3–5. Initialise the database, load the dataset, generate embeddings, evaluate

```powershell
python setup_project.py
```

This one script:
1. checks the dataset files,
2. creates `data\complaints.db` and its tables,
3. loads the 10,000 synthetic complaints (skipped if they are already there, so it is safe to run again),
4. downloads Sentence-BERT once (about 90 MB, needs internet) and saves embeddings to `data\embeddings\`
   (a few minutes on a laptop CPU),
5. evaluates both models and saves `results\metrics.json` and `results\figures\`.

No internet right now? Run `python setup_project.py --skip-sbert`. Everything works with TF-IDF, and you can
run the full command later.

### 6. Start the application

```powershell
streamlit run app.py
```

Your browser opens at http://localhost:8501. Stop the app with **Ctrl+C** in the terminal.

Next time, you only need:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

## Run the tests

```powershell
python -m pytest
```

Expected result: all tests pass. One test, `test_real_sbert_if_installed`, checks the real Sentence-BERT
model: it is skipped if the model cannot be loaded and runs automatically after setup step 4. The tests use a
temporary database and never change `data\complaints.db` or `results\metrics.json`.

## Using the app

| Page | What it does |
|---|---|
| Home | Totals, complaints by category, model status, dataset information, recent complaints |
| Submit complaint | Shows the estimated priority and checks the Top-5 similar complaints from the same locality first. Highlighted cards are at or above the threshold. You can submit anyway; the complaint gets a reference such as `CD-20261117-K7P3QX`. |
| Duplicate detection | Try any text with TF-IDF or Sentence-BERT, move the threshold slider, compare both models side by side. Nothing is saved. |
| Model evaluation | Measured accuracy, precision, recall, F1, confusion matrices, score distributions, precision-recall curve, speed |
| All complaints | Search and filter by category, locality, status, source and priority; select a row for details |
| Admin dashboard | **Priority queue** (unsolved complaints, most urgent first, with the reasons), statistics and charts, status updates, confirm or reject suggested duplicates |

**The similarity meter:** the bar shows the cosine similarity (0 = unrelated, 1 = same meaning). The black tick
is the threshold. The bar turns yellow when the score reaches it. A score is a similarity, not a probability,
and every suggested duplicate is only a suggestion for a person to confirm.

## Complaint priority: what to solve first

Every complaint gets a priority score from 0 to 100 and a level: **Critical** (75+), **High** (55+),
**Medium** (35+) or **Low**. The score is the sum of explainable parts (`src/priority.py`):

| Part | Points | Example |
|---|---|---|
| Category base | 8 to 30 | Sewage Overflow 30, Potholes 20, Public Park Maintenance 8 |
| Safety risk words | +25 | accident, injury, could fall, open manhole, dangerous, khatarnak |
| Health risk words | +18 | bad smell, stench, sewage, dengue, contaminated, badbu, bimari |
| Daily life disrupted | +12 | difficulty in walking, blocked, no water, ambulance, paani nahi |
| Vulnerable people | +10 | children, school, elderly, hospital, bachche, buzurg |
| Wide impact or repeated | +8 | whole area, residents, again, already complained |
| Marked urgent | +6 | urgent, immediately, emergency, turant |
| Safety risk **and** vulnerable people | +15 | open manhole outside a school |
| Duration | +4 / +7 / +10 | 3+ days / 1+ week / 2+ weeks ("for 2 weeks", "5 din se") |
| Linked duplicate reports | +5 each, max +15 | other complaints linked to this one (rejected links do not count) |

Each word group counts once, so repeating a word does not raise the score. The admin sees the reasons,
for example *"Health risk +18 (bad smell, sewage); Daily life disrupted +12 (difficulty in walking)"*.

Example: *"Sewage is overflowing on our road, there is a very bad smell and people have difficulty in walking.
Children go to school through it. This has been going on for 2 weeks"* scores **80, Critical**.

**Existing databases are upgraded automatically.** If your `data\complaints.db` was created before this feature,
the app adds the priority columns and scores all complaints the first time it starts (about 2 seconds).

**Why rules instead of a trained model?** The dataset has no "true priority" labels, so a model could not be
trained or evaluated honestly. Rules are transparent: every point can be traced to a word. Limitations: keyword
rules do not understand negation ("no accident yet"); longer, more detailed complaints mention more risk words
and so tend to score higher; and the weights are expert judgement, not learned. Recording officers' own
priority decisions would allow a trained model later.

## Settings (optional)

Set these in PowerShell before `streamlit run app.py`:

| Setting | Example | Effect |
|---|---|---|
| `ADMIN_PASSWORD` | `$env:ADMIN_PASSWORD="choose-a-password"` | The Admin page asks for this password |
| `ENABLE_SBERT` | `$env:ENABLE_SBERT="0"` | Use TF-IDF only (fast start, no model needed) |
| `COMPLAINTS_DB` | `$env:COMPLAINTS_DB="D:\demo.db"` | Use a different database file |

To start again with a fresh database: stop the app, delete `data\complaints.db`, and run
`python setup_project.py`.

## Results so far (test set, synthetic data)

| Model | Threshold (from validation) | Precision | Recall | F1 | Recall@5 |
|---|---|---|---|---|---|
| TF-IDF | 0.038 | 0.653 | 0.864 | 0.744 | 0.244 |
| TF-IDF + same locality | 0.027 | 0.721 | 0.939 | 0.815 | |
| Sentence-BERT | appears after `python setup_project.py` | | | | |

TF-IDF needs a very low threshold to catch paraphrases, so it raises many false alarms. That is the gap
Sentence-BERT is meant to close. See `docs/PROJECT_REPORT.md` for the method, formulas and discussion.

## Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit` is not recognised | Activate the virtual environment first: `.\venv\Scripts\Activate.ps1` |
| "Sentence-BERT is not available" on a page | Run `pip install -r requirements.txt`, then `python setup_project.py` with internet access |
| "The database is empty" | Click the load button on the page, or run `python setup_project.py` |
| Model evaluation page says "No results yet" | Run `python setup_project.py` |
| First start is slow | The app is loading the model and stored complaints once; later page changes are fast |
