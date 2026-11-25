# Synthetic Civic Complaint Dataset for Duplicate Complaint Detection

> **Important:** Every record in this dataset is **artificially generated**. It is inspired by common
> Indian municipal issues (including those typical of a city like Surat), but it is **not** real
> Surat Municipal Corporation (SMC) data and **no record was submitted by a real resident**.
> All locality names are fictional. Every row carries `data_origin = synthetic`.

## 1. Purpose

This dataset supports the M.Tech project *Duplicate Complaint Detection Using Semantic Similarity and
NLP*. It is used to build and compare a TF-IDF baseline and a Sentence-BERT model, select a cosine
similarity threshold, evaluate duplicate detection (Accuracy, Precision, Recall, F1), and power a
Top-K retrieval prototype. It was created because no real municipal complaint dataset was available
to the project team.

## 2. Files

| File | Rows | Description |
|---|---|---|
| `data/raw/complaints.csv` | 10,000 | Individual complaint records |
| `data/raw/complaint_pairs.csv` | 10,000 | Labeled pairs: 5,000 duplicates, 5,000 non-duplicates |
| `data/splits/train_complaints.csv` / `train_pairs.csv` | 6,924 / 7,000 | Training split |
| `data/splits/val_complaints.csv` / `val_pairs.csv` | 1,499 / 1,500 | Validation split |
| `data/splits/test_complaints.csv` / `test_pairs.csv` | 1,577 / 1,500 | Test split |
| `data/validation_report.txt` | – | Output of the validation script |
| `src/data/generate_dataset.py` | – | Generator (fixed seed 2026) |
| `src/data/content_bank.py` | – | Phrase bank used by the generator |
| `src/data/validate_dataset.py` | – | Independent validation |

## 3. Column descriptions

### complaints.csv

| Column | Description |
|---|---|
| `complaint_id` | Unique ID (`CMP00001` to `CMP10000`) |
| `complaint_text` | Full complaint description |
| `category` | One of 20 main categories |
| `subcategory` | Specific problem type (49 subcategories) |
| `locality` | Fictional locality (100 localities, e.g. "Navrang Nagar") |
| `ward` | Synthetic ward ID (`W01` to `W35`) |
| `zone` | Synthetic zone (7 zones, e.g. "North Zone") |
| `submission_date` | Date between July 2025 and October 2026 |
| `status` | Open, In Progress, Resolved or Closed (randomly assigned) |
| `data_origin` | Always `synthetic` |
| `issue_id` | *Extra.* ID of the underlying real-world problem the complaint describes |
| `subcategory_key` | *Extra.* Internal short code of the subcategory |
| `writing_style` | *Extra.* Style used to write the text (formal, Hinglish, short, ...) |
| `split` | *Extra.* train / val / test |

### complaint_pairs.csv

| Column | Description |
|---|---|
| `pair_id` | Unique ID (`PR00001` to `PR10000`) |
| `complaint_id_1`, `complaint_id_2` | IDs referring to `complaints.csv` |
| `complaint_1`, `complaint_2` | Texts of the two complaints |
| `category_1`, `category_2` | Categories of the two complaints |
| `duplicate_label` | 1 = duplicate, 0 = not duplicate |
| `data_origin` | Always `synthetic` |
| `pair_type` | *Extra.* How the pair was built (see Section 5) |
| `split` | *Extra.* train / val / test |

**Do not use `issue_id`, `subcategory_key` or `pair_type` as model input features.** They encode the
answer and exist only for validation and error analysis. Models should use only the complaint text
(and, if you choose, locality/date metadata).

## 4. Label definitions

- **1 (duplicate):** Both complaints describe the **same underlying issue**: the same problem
  (subcategory) at the same place (locality and landmark) in the same time window.
- **0 (not duplicate):** The complaints describe **different underlying issues**.

## 5. Generation methodology

1. **Issues.** An issue is one real-world problem, defined by a subcategory, a locality, a landmark
   (e.g. "near the community hall") and a start date. 3,606 issues were created. Each
   (subcategory, locality) combination is used at most once, so two different issues are never
   identical problems at the same place.
2. **Paraphrase families.** Each issue receives 1 to 6 complaints. Within a family, each complaint uses
   a different sentence variant and a different writing style:

   | Style | Example of what it looks like |
   |---|---|
   | basic / brief | Plain English statement, with or without impact and request |
   | formal / long | Letter style ("Respected Sir/Madam, I would like to bring to your notice that ...") |
   | angry | Frustrated tone ("We have already complained twice but nobody came.") |
   | subject | "Complaint regarding ... in ..." followed by details |
   | hinglish / hinglish_short | Romanised Hindi-English ("... me 3 din se paani nahi aa raha") |
   | mixed | English description with a Hinglish request |
   | short | Very short text ("open manhole Gokul Society") |

   About 15% of informal texts receive realistic spelling errors ("garbge", "plz", "watr"). Complaint
   dates differ by up to 4 days inside a family, so reported durations also differ naturally
   ("since 3 days" vs "4 din se").
3. **Related categories.** About 12% of complaints from some subcategories are filed under a closely
   related category (e.g. a pothole filed under *Road Damage*, sewage on the road under *Drainage
   Problems*). As a result, 272 duplicate pairs have different `category_1` and `category_2`, which is
   realistic because citizens do not always pick the same category.
4. **Positive pairs (5,000).** Two complaints from the same issue. At most 6 pairs per issue are used,
   and pairs with word overlap (Jaccard) above 0.75 are skipped to avoid near-copies.
5. **Negative pairs (5,000).** Two complaints from different issues, built in five types:

   | `pair_type` | Count | What it tests |
   |---|---|---|
   | `hard_same_locality_same_category` | 1,249 | Same area, same category, different problem (e.g. no water vs dirty water) |
   | `hard_same_subcategory_diff_locality` | 1,000 | Same problem in a different area (very similar text, different issue) |
   | `hard_same_locality_diff_category` | 1,000 | Same area, unrelated problem |
   | `medium_same_category_diff_locality` | 500 | Same category, different problem and area |
   | `easy_random` | 1,251 | Different category and area |

6. **Excluded ambiguous pairs.** Some problem types can describe the same event (e.g. "blocked drain"
   and "sewage on road", or "no water" and "low pressure"). Negative pairs between such
   subcategories **in the same locality** were never generated, because their label cannot be
   assigned with confidence. The list is in `CONFUSABLE` inside `content_bank.py`.
7. **Reproducibility.** A fixed random seed (2026) is used. Running the generator twice produces
   byte-identical CSV files (verified with MD5 checksums).

## 6. Train / validation / test split

The split is **grouped by locality**. Each of the 100 fictional localities is assigned entirely to
one split (70 train, 15 validation, 15 test), and pairs are formed only within a split.

This means:
- No paraphrase family (issue) appears in more than one split, so the model never sees a rephrased
  version of a test complaint during training.
- The test set contains localities the model has never seen.
- Each split still contains many issues from the same locality, which is needed for hard negatives.

| Split | Complaints | Localities | Duplicate pairs | Non-duplicate pairs |
|---|---|---|---|---|
| Train | 6,924 | 70 | 3,500 | 3,500 |
| Validation | 1,499 | 15 | 750 | 750 |
| Test | 1,577 | 15 | 750 | 750 |

Complaint counts are not exactly 70/15/15 because whole localities are assigned. Pair counts are
exactly 70/15/15. **Use the validation set to choose the similarity threshold and report final results
only on the test set.**

## 7. Validation results

All checks in `validate_dataset.py` passed (full output in `validation_report.txt`):

| Check | Result |
|---|---|
| Complaint records | 10,000 |
| Complaint pairs | 10,000 (5,000 label 1, 5,000 label 0) |
| Unique complaint IDs | 10,000 |
| Unique categories / subcategories | 20 / 49 |
| Missing values | 0 |
| Duplicate complaint IDs / pair IDs | 0 / 0 |
| Invalid ID references | 0 |
| Self-pairs | 0 |
| Reversed or repeated pairs | 0 |
| Identical complaint texts | 0 |
| Label 1 pairs from same issue / label 0 from different issues | 100% / 100% |
| Issues or localities crossing splits | 0 |

**Text length:** 3 to 76 words (mean 30.4, median 29); 994 short (8 words or fewer), 2,412 medium
(9 to 25), 6,594 long (more than 25).

**Category distribution:** largest is Water Supply (8.4%), smallest is Construction Waste (3.4%).
Categories with more subcategories have more complaints.

**Difficulty check (word overlap, Jaccard):**

| Pair type | Mean Jaccard |
|---|---|
| Duplicates | 0.151 |
| Hard: same locality, same category | 0.121 |
| Hard: same subcategory, different locality | 0.117 |
| Hard: same locality, different category | 0.106 |
| Medium | 0.083 |
| Easy random | 0.069 |

71.7% of duplicate pairs share less than 20% of their words, so duplicates cannot be detected by
simple keyword overlap. This is intended: it creates a fair test of whether semantic models
(Sentence-BERT) improve on lexical models (TF-IDF).

## 8. Limitations

1. **Not real data.** The texts were produced from a hand-written phrase bank. Real complaints are
   messier: longer stories, mixed issues in one complaint, Gujarati script, voice-to-text errors,
   irrelevant details and photos. Results on this dataset will likely be **higher** than on real data.
2. **Shared phrase bank across splits.** Splits are separated by issue and locality, but all splits are
   written from the same set of sentence variants. A model that is **fine-tuned** on the training set
   may partly learn these patterns, so test results would be optimistic. Pre-trained models used
   without fine-tuning (e.g. off-the-shelf Sentence-BERT) are not affected in the same way.
3. **One issue per complaint.** Every complaint describes exactly one problem.
4. **Location matters for labels.** The same problem in two different localities is labeled 0. A
   text-only model cannot always tell these apart, especially when the text names the locality only
   once. Combining text similarity with locality/date matching is expected to help.
5. **Simplified metadata.** Status, ward and zone are randomly assigned and carry no real meaning.
   Wards and zones do not correspond to real SMC boundaries.
6. **Limited language coverage.** English and romanised Hinglish only; no Gujarati or Devanagari.

## 9. Why synthetic data is not equivalent to real municipal data

Synthetic data lets us control labels exactly and test specific difficulties (paraphrases, hard
negatives), but it reflects the assumptions of whoever wrote the generator. It cannot reproduce the
true distribution of real complaint topics, writing habits, or the ambiguous cases that municipal
officers actually face. Any conclusion from this dataset should be presented as a **controlled
experiment**, and validated on real complaint data (e.g. obtained from a municipal body with
permission, with personal details removed) before any deployment claim is made.

## 10. How to regenerate and validate

From the `Project_NLP/` root:

```bash
python src/data/generate_dataset.py     # creates data/raw and data/splits
python src/data/validate_dataset.py     # prints checks and writes data/validation_report.txt
```

Requirements: Python 3.9+ and pandas.
