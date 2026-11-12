# Project Review — Technical Audit of the Academic Report

This document records an independent review of the B.Sc. major project report
*"Bridging the Gap Between Education and Industry: A Machine Learning-Based Study of
Career Change"* (Cilveri Prashanth, Bhavan's Vivekananda College, Osmania University,
2025–2026), carried out before publishing this repository.

The goal is accuracy, not cosmetics: where the report contradicts itself, the
contradiction is documented here, the evidence for each version is given, and the
version supported by the strongest evidence (actual code and actual program output)
is the one used in this repository.

---

## 1. What was examined

- The complete 80-page project report, including all chapters, figures and tables.
- Appendix A — the Python implementation printed in the report.
- Appendix B — the dataset description and the dataset screenshots (Figs. 8.1–8.3).
- The model-output screenshots printed as Figs. 4.1–4.6 (Chapter 4).
- The results figures printed in Chapter 6 (Figs. 6.1.1, 6.2.1, 6.3, 6.4, 6.4.1).
- The dashboard screenshots (Figs. 6.5.1–6.5.4).

## 2. What could NOT be examined

- **The original dataset file (`dataset5k.csv`) was not available.** It is not part of
  the report and was not found in the author's files. The models therefore could not
  be re-run, and **no result in this repository is claimed as reproduced**. All results
  are labelled either "documented run output" (read from the report's own output
  screenshots) or "as reported in Chapter 6" (narrative values that could not be
  verified against any output).

---

## 3. Findings

### 3.1 Target variable — the chapters describe the wrong problem

- **Appendix A code** trains on `y = data['Likely to Change Occupation']`.
- **Appendix B** also defines the target as *Likely to Change Occupation* (binary 0/1),
  and the dataset screenshot (Fig. 8.3.1) shows it as the final column.
- **Chapters 1, 3, 4, 5 and 6**, however, repeatedly describe the task as predicting
  *employed vs. unemployed* / "employability".

**Verdict:** the implemented task is predicting **likelihood of changing occupation**.
The employed/unemployed framing in the chapter prose does not match the code, the
dataset, or Appendix B, and is not used in this repository.

### 3.2 Dataset size — three incompatible numbers

| Source | Claim |
|---|---|
| Appendix B text | "Approximately 5000 observations" |
| File name loaded by the code | `dataset5k.csv` (suggests ~5k) |
| Dataset viewer screenshot (Fig. 8.1) | **38,444 entries** |
| Fig. 8.2 | "[5 rows x 23 columns]" — 23 columns total |
| Model outputs (Figs. 4.1–4.6) | test-set support = **5,763** rows |

With the code's 80/20 split, a 5,763-row test set implies ≈ 28,815 rows after
`dropna()` — consistent with a 38,444-row file losing roughly a quarter of its rows
when incomplete records are dropped, but **this is arithmetic inference, not
verification**. The true row count cannot be confirmed without the file.

**Verdict:** dataset size is **unresolved**. This repository states only what the
evidence shows and does not publish a definitive size.

### 3.3 Model accuracy — Chapter 6's table matches no recorded run

Chapter 6, Table 6.1 reports one set of accuracies; the actual program outputs
printed in the same report (Figs. 4.1–4.6, Chapter 4) show a different set. The
Chapter 6 accuracy *chart* (Fig. 6.1.1) agrees with the program outputs, not with
Table 6.1.

| Model | Table 6.1 (Ch. 6) | Documented run output (Figs. 4.1–4.6) |
|---|---|---|
| Logistic Regression | 78% | **90.28%** |
| Decision Tree | 82% | **100%** |
| Random Forest | 89% | **100%** |
| SVM | 87% | **97.37%** |
| KNN | 80% | **85.34%** |
| Naive Bayes | 75% | **61.88%** |

**Verdict:** Table 6.1 is treated as unverified (it may come from an earlier run or
be illustrative). This repository presents the documented run outputs as the
project's results, clearly labelled as *reported, not reproduced*.

### 3.4 Confusion matrix — the table and the figure describe different runs

- Table 6.2 (Random Forest): TP 85, FN 10, FP 8, TN 47 — total **150** samples.
- Fig. 6.2.1 (the plotted matrix) and Fig. 4.3 (the printed output):
  `[[2395, 0], [0, 3368]]` — total **5,763** samples, zero errors.

**Verdict:** Table 6.2 does not describe the documented run. The figure values are
used in this repository.

### 3.5 Feature importance — the narrative names features that do not exist

- Section 6.4's text states that *technical skills* and *internship experience* were
  the most significant predictors.
- The actual feature-importance output (Fig. 6.4.1) ranks:
  **Job Satisfaction 0.546, Career Change Interest 0.237, Salary 0.146**, with all
  remaining 19 features below 0.01. The dataset contains **no** "technical skills"
  or "internship experience" columns at all.

**Verdict:** the Section 6.4 narrative is incorrect for this dataset. The documented
ranking is used in this repository.

### 3.6 Two different dataset schemas are described

- **Appendix B** lists variables such as `Skill_Score`, `Internships_Completed`,
  `Projects_Completed`, `Communication_Skills` and `Technical_Skills`. None of these
  appear in the dataset screenshots or in the feature-importance output.
- The **actual schema** (from Figs. 6.4/6.4.1 and 8.3.1) has 22 features:
  Age, Gender, Field of Study, Current Occupation, Education Level,
  Years of Experience, Industry Growth Rate, Job Satisfaction, Work-Life Balance,
  Job Opportunities, Salary, Job Security, Career Change Interest, Skills Gap,
  Family Influence, Mentorship Available, Certifications, Freelancing Experience,
  Geographic Mobility, Professional Networks, Technology Adoption,
  Career Change Events — plus the target, 23 columns in total, matching Fig. 8.2.

**Verdict:** Appendix B describes an idealised schema, not the file that was used.
This repository documents the actual schema.

### 3.7 Appendix B preprocessing summary contradicts the run

Appendix B states that irrelevant columns "such as ID, Name, and **Salary**" were
removed. Salary was in fact retained — it is the third most important feature
(0.146) in the documented run.

### 3.8 Feature-name bug in Appendix A

The feature-importance block reads:

```python
feature_names = data.drop('Employment_Status', axis=1).columns
```

There is **no `Employment_Status` column** in this dataset. Run as printed, this line
raises a `KeyError` (and if it did not, the names would misalign with the features,
which were built by dropping `Likely to Change Occupation`). The correctly labelled
importance table in Fig. 6.4.1 shows the executed code must have differed from the
printed appendix. The code in this repository uses the true feature names.

### 3.9 Data leakage in the original pipeline (fixed here)

Appendix A fits `StandardScaler` on the **entire** feature matrix **before** the
train–test split:

```python
X = scaler.fit_transform(X)          # fitted on ALL rows
X_train, X_test, ... = train_test_split(X, y, ...)
```

Test-set statistics therefore leak into training. (`LabelEncoder` is likewise fitted
on the full data — a milder form of the same issue.) In this repository the split
happens first and the scaler is fitted on the training data only. This change is
marked in the code and, on its own, would be expected to change the results
slightly; it is one more reason the documented accuracies are not claimed as
reproducible by the code here.

### 3.10 Perfect scores — a property of the dataset, not of modelling skill

Decision Tree and Random Forest both score **100% on 5,763 unseen test rows**, and
three features (Job Satisfaction, Career Change Interest, Salary) account for ~93%
of the importance. Perfect separation of this kind, together with patterns visible
in the dataset screenshots (e.g. 25 years of experience at age 30; a "High School"
education paired with the occupation "Doctor"), indicates the data is **synthetic /
rule-generated**, with the target derived largely from a few input columns. The
documented accuracies therefore describe how well the models recover the data
generation rule — they should **not** be read as real-world predictive performance
on genuine survey data.

Related minor inconsistency: Section 6.3 lists Decision Tree AUC ≈ 0.88, which is
incompatible with its documented 100% accuracy (a perfect classifier has AUC 1.0).

### 3.11 Smaller code issues in Appendix A (noted, fixed or preserved deliberately)

- `data.dropna()` is executed twice.
- `y = y.astype(int)` appears **after** the split and has no effect on training.
- A single `LabelEncoder` instance is reused across all categorical columns
  (harmless here because each column is transformed in place, but fragile).

### 3.12 The dashboard is a separate prototype

The report documents an education–job matching website ("EduAlign", Figs.
6.5.1–6.5.4) that takes a job title and a degree and returns a match score, a skills
breakdown and recommended career paths. **No dashboard source code appears in the
report, and nothing in the report shows it calling the trained ML models.** It is
documented in this repository as a separate prototype (see `dashboard/`), and is
**not** described as ML-powered.

---

## 4. What this repository does differently from the report

1. Uses the target the code actually used: **Likely to Change Occupation**.
2. Documents the **actual 22-feature schema** evidenced by the report's figures.
3. Presents **both** accuracy sets, labelled by source, instead of silently picking
   one (see `reports/model_comparison.md`).
4. Fixes the scaling-before-split leakage and the `Employment_Status` feature-name
   bug, with the changes marked in the code.
5. States plainly that the dataset is unavailable and results are **not reproduced**.

## 5. What remains uncertain

- The true size and provenance of `dataset5k.csv`.
- Which run Table 6.1 and Table 6.2 in Chapter 6 came from.
- Whether the dashboard was ever connected to the trained models.
- The exact code that produced Fig. 6.4.1 (it differed from printed Appendix A).

*Review prepared from the academic report as submitted. If the original dataset or
notebook becomes available, the pipeline in `src/` can be run end-to-end and this
review updated with genuinely reproduced results.*
