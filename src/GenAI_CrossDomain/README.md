# Aspect-Based Sentiment Analysis with Gemini 2.5 Flash

This notebook runs zero-shot and few-shot sentiment classification experiments using Google's Gemini 2.5 Flash model across three educational review domains: **Course**, **Teacher**, and **University**.

---

## What This Does

The core idea is straightforward — given a student review and a specific aspect (e.g., *teaching*, *grading*, *library*), the model predicts whether the sentiment toward that aspect is **positive**, **negative**, or **neutral**.

Two prompting strategies are compared:

- **Zero-shot**: The model gets only the review and aspect, with a minimal instruction to classify.
- **Few-shot**: The model is given six labelled examples before being asked to classify, acting as an "expert ABSA classifier."

---

## Setup

Install the required packages before running anything:

```bash
pip install google-generativeai scikit-learn pandas tqdm openpyxl
```

> **Note:** The `google-generativeai` package is officially deprecated. If you hit import warnings, switch to `google-genai` (the newer SDK) — the commented-out code at the top of the notebook shows how to do that.

You'll also need to mount your Google Drive and provide your Gemini API key when prompted.

---

## Data

All datasets live in your Google Drive under `ABSA_GEMINI/`. Each domain has two files:

| File | Purpose |
|------|---------|
| `course_balanced-new.csv` | Balanced training data for Course domain |
| `teacher_balanced-new.csv` | Balanced training data for Teacher domain |
| `university_balanced-new.csv` | Balanced training data for University domain |
| `course_test.csv` | Fixed test set for Course domain |
| `teacher_test.csv` | Fixed test set for Teacher domain |
| `university_test.csv` | Fixed test set for University domain |

Each file contains at minimum: `review`, `aspect`, `sentiment`, and `row_id` columns. The sentiment labels are normalised to lowercase (`positive`, `negative`, `neutral`) before any experiment runs.

---

## Cross-Domain Setup

The experiments are deliberately **cross-domain** — the model is trained/prompted on data from two domains and evaluated on the third. The active configuration in the notebook at the time of export:

| Split | Domains Used |
|-------|-------------|
| Train | Teacher + University (7,498 rows) |
| Validation | 10% of train (938 rows) |
| Test | Course (844 rows) |

The train/validation split uses stratified sampling (`random_state=42`) to keep class proportions consistent.

> The commented-out cells above the active config show the other two cross-domain permutations (Course+University → Teacher, and Teacher+Course → University). You can swap them in by uncommenting the relevant block.

---

## Prompts

**Zero-shot prompt:**
```
Classify the sentiment of the aspect in the review.
Review: {review}
Aspect: {aspect}
Answer ONLY one word: positive, negative, or neutral.
```

**Few-shot prompt:**

Six examples are provided covering a range of aspects and all three sentiment labels. The examples include both domain-specific (teacher, grading, exam) and general university contexts (library, materials). The full few-shot block is injected into every request at inference time.

---

## Running the Experiments

Each experiment is triggered by calling the relevant function with a test dataframe and an experiment name:

```python
# Zero-shot
run_gemini_zero_shot(test_df, "gemini_zero_shot_teacher")

# Few-shot
run_gemini_few_shot(teacher_test, "gemini_few_shot_teacher")
```

The experiment name also controls where results are saved — a folder with that name is created inside `ABSA_GEMINI/` on your Drive.

**Retry logic:** Each API call gets up to 5 attempts. If all fail (e.g. due to 503 errors), the prediction defaults to `neutral`. The 503s you see in the output logs are transient Gemini API overloads — the retry mechanism handles most of them gracefully.

**Post-processing:** The raw model response is lowercased and stripped, then matched against the three valid labels. If none match, it defaults to `neutral`.

---

## Output Files

For each experiment, two CSVs are saved to Drive:

- `predictions.csv` — row-level predictions with `row_id`, `true_sentiment`, `predicted_sentiment`
- `metrics.csv` — summary scores: Accuracy, Macro F1, Weighted F1

---

## Results Summary

| Experiment | Accuracy | Macro F1 | Weighted F1 |
|------------|----------|----------|-------------|
| Zero-shot → Teacher test | 86.4% | 0.69 | 0.84 |
| Few-shot → Teacher test | 85.9% | 0.71 | 0.84 |
| Few-shot → Course test | 78.6% | 0.66 | 0.76 |
| Few-shot → University test | 94.2% | 0.74 | 0.94 |

A consistent pattern across all experiments: **positive and negative classes perform well**, while **neutral is the hardest to classify** (F1 scores typically between 0.25–0.35). This is expected given the class imbalance and the inherently ambiguous nature of neutral reviews.

The University domain sees the highest accuracy overall — likely due to the strong positive class dominance (676 out of 844 test samples).

---

## Known Issues

- **503 errors are frequent** during long runs. The 5-attempt retry + 5-second sleep handles most of these, but particularly heavy API traffic periods may cause some rows to default to `neutral`.
- The `google-generativeai` package will eventually stop receiving updates. Plan to migrate to `google-genai` before that becomes a blocker.
- `review` and `aspect` columns are cast to `str` explicitly — this is intentional, since occasional NaN or numeric values in the raw CSVs would otherwise break the prompt construction.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `google-generativeai` | Gemini API client |
| `pandas` | Data loading and manipulation |
| `scikit-learn` | Metrics (accuracy, F1, classification report) |
| `tqdm` | Progress bars during inference |
| `openpyxl` | Excel support (if needed for data export) |

Python 3.12, run on Google Colab.