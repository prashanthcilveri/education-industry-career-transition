# Model Comparison — Documented Results

> **Status of these numbers:** the original dataset (`dataset5k.csv`) is not
> available, so the models could not be re-run. **Nothing on this page is a
> reproduced result.** Two different result sets appear inside the academic
> report itself; both are recorded here with their sources. See
> [`PROJECT_REVIEW.md`](../PROJECT_REVIEW.md) for the full audit.

## Set A — Documented run output (primary)

Read from the actual program-output screenshots printed in the report
(Figs. 4.1–4.6), and consistent with the Chapter 6 charts (Figs. 6.1.1, 6.2.1,
6.3, 6.4.1). Test set: 5,763 rows (class support 2,395 / 3,368).

| Model | Accuracy | Confusion matrix |
|---|---|---|
| Logistic Regression | 90.28% | [[2141, 254], [306, 3062]] |
| Decision Tree | 100% | [[2395, 0], [0, 3368]] |
| Random Forest | 100% | [[2395, 0], [0, 3368]] |
| SVM | 97.37% | [[2303, 92], [58, 3310]] |
| KNN | 85.34% | [[2066, 329], [516, 2852]] |
| Naive Bayes | 61.88% | [[2395, 0], [2197, 1171]] |

Random Forest ROC AUC (Fig. 6.3): **1.00**.

### Random Forest feature importance (Fig. 6.4.1, documented output)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Job Satisfaction | 0.546312 |
| 2 | Career Change Interest | 0.236892 |
| 3 | Salary | 0.146497 |
| 4 | Job Opportunities | 0.008138 |
| 5 | Age | 0.006985 |
| 6 | Years of Experience | 0.006954 |
| 7 | Skills Gap | 0.004954 |
| 8 | Technology Adoption | 0.004843 |
| 9 | Work-Life Balance | 0.004793 |
| 10 | Professional Networks | 0.004717 |
| 11 | Field of Study | 0.004644 |
| 12 | Current Occupation | 0.004628 |
| 13 | Job Security | 0.004627 |
| 14 | Education Level | 0.002760 |
| 15 | Career Change Events | 0.002107 |
| 16 | Family Influence | 0.002101 |
| 17 | Industry Growth Rate | 0.002092 |
| 18 | Gender | 0.001349 |
| 19 | Mentorship Available | 0.001225 |
| 20 | Geographic Mobility | 0.001210 |
| 21 | Certifications | 0.001182 |
| 22 | Freelancing Experience | 0.000989 |

## Set B — As reported in Chapter 6, Table 6.1 (unverified)

These values appear only in the Chapter 6 narrative table. They match no program
output printed in the report, and the confusion matrix printed alongside them
(Table 6.2: TP 85, FN 10, FP 8, TN 47; n = 150) also matches no documented run.

| Model | Accuracy (Table 6.1) |
|---|---|
| Logistic Regression | 78% |
| Decision Tree | 82% |
| Random Forest | 89% |
| SVM | 87% |
| KNN | 80% |
| Naive Bayes | 75% |

## Reading these results honestly

Decision Tree and Random Forest scoring 100% on 5,763 unseen rows, with ~93% of
feature importance concentrated in Job Satisfaction, Career Change Interest and
Salary, indicates the target is (near-)deterministically derived from the inputs —
the dataset shows clear signs of being synthetically generated. The Set A numbers
show the models recover that rule; they are **not** evidence of real-world
predictive performance on genuine survey data. Any reproduction attempt should
start by examining the relationship between those three features and the target.
