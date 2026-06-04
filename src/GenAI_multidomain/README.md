# Generative AI for Aspect-Based Sentiment Analysis using Gemini 2.5 Flash

## Project Overview

This project investigates the use of Google's Gemini 2.5 Flash large language model for Aspect-Based Sentiment Analysis (ABSA) within educational review datasets.

The study evaluates:

1. Zero-Shot Prompting
2. Few-Shot Prompting
3. Multidomain Evaluation
4. Cross-Domain Comparison
5. Error Analysis
6. Neutral Sentiment Analysis

The work forms part of an MSc Dissertation comparing Generative AI approaches against traditional Transformer-based ABSA models including:

- ModernBERT
- BERTweet
- LLaMA 3B
- LLaMA + LoRA
- Gemini 2.5 Flash

---

# Dataset

Three balanced educational review datasets were used.

| Domain | Samples |
|----------|---------:|
| Course | 4218 |
| Teacher | 4218 |
| University | 4218 |
| Total | 12654 |

Each instance contains:

- Review
- Aspect
- Sentiment Label
- Domain

Possible sentiment labels:

- positive
- negative
- neutral

---

# Dataset Statistics

| Domain | Unique Aspects |
|----------|---------:|
| Course | 1485 |
| Teacher | 1697 |
| University | 503 |

Sentiment Distribution:

| Sentiment | Count |
|----------|---------:|
| Positive | 7764 |
| Negative | 3550 |
| Neutral | 1340 |

---

# Test Sets

Each domain contains an independent test set.

| Domain | Test Samples |
|----------|---------:|
| Course | 844 |
| Teacher | 844 |
| University | 844 |

Total evaluation samples:

2532

---

# Model

Google Gemini 2.5 Flash

```python
MODEL_NAME = "gemini-2.5-flash"
```

Inference performed through the Google GenAI SDK.

---

# Project Workflow

## Phase 1 – Data Preparation

Load:

- course_balanced.csv
- teacher_balanced.csv
- university_balanced.csv

Load shared test sets:

- course_test.csv
- teacher_test.csv
- university_test.csv

Clean labels:

```python
positive
negative
neutral
```

---

## Phase 2 – Multidomain Dataset Construction

Combine all domains:

```python
multidomain_df = pd.concat([
    course_df,
    teacher_df,
    university_df
])
```

Resulting dataset:

12654 samples

---

## Phase 3 – Exploratory Data Analysis

Performed:

### Domain Distribution

- Course
- Teacher
- University

### Sentiment Distribution

- Positive
- Negative
- Neutral

### Review Length Analysis

Statistics:

- Mean
- Median
- Standard Deviation
- Maximum Length

### Aspect Frequency Analysis

Top 20 aspects visualized for:

- Course
- Teacher
- University

---

# Zero-Shot Prompting

Prompt:

```text
You are an expert Aspect-Based Sentiment Analysis system.

Determine the sentiment expressed towards the given aspect.

Possible labels:

positive
negative
neutral

Review:
{review}

Aspect:
{aspect}

Output ONLY one label.
```

---

# Few-Shot Prompting

Few-shot examples sampled from multidomain training data.

Examples included:

- 3 Positive
- 3 Negative
- 3 Neutral

Total:

9 demonstrations

Prompt structure:

```text
Review: ...
Aspect: ...
Answer: positive

Review: ...
Aspect: ...
Answer: negative

Review: ...
Aspect: ...
Answer: neutral
```

followed by the target instance.

---

# Inference Pipeline

For each test instance:

1. Create prompt
2. Submit request to Gemini
3. Extract label
4. Save prediction

Retry logic:

```python
max_attempts = 8
```

Handles:

- 503 errors
- API throttling
- temporary service failures

Fallback:

```python
neutral
```

---

# Evaluation Metrics

Calculated using Scikit-Learn:

```python
accuracy_score
f1_score
classification_report
confusion_matrix
```

Metrics reported:

- Accuracy
- Macro F1
- Weighted F1
- Precision
- Recall

---

# Results

## Zero-Shot Results

| Domain | Accuracy | Macro F1 | Weighted F1 |
|----------|---------:|---------:|---------:|
| Course | 0.7832 | 0.6460 | 0.7462 |
| Teacher | 0.8531 | 0.6777 | 0.8303 |
| University | 0.9514 | 0.7383 | 0.9423 |

Average:

| Metric | Score |
|----------|---------:|
| Accuracy | 0.8626 |
| Macro F1 | 0.6873 |
| Weighted F1 | 0.8396 |

---

## Few-Shot Results

| Domain | Accuracy | Macro F1 | Weighted F1 |
|----------|---------:|---------:|---------:|
| Course | 0.7950 | 0.6895 | 0.7718 |
| Teacher | 0.8531 | 0.6996 | 0.8376 |
| University | 0.9455 | 0.7217 | 0.9382 |

Average:

| Metric | Score |
|----------|---------:|
| Accuracy | 0.8645 |
| Macro F1 | 0.7036 |
| Weighted F1 | 0.8492 |

---

# Zero-Shot vs Few-Shot Comparison

| Domain | Best Configuration |
|----------|----------|
| Course | Few-Shot |
| Teacher | Few-Shot |
| University | Zero-Shot |

Few-shot prompting improved sentiment boundary detection in Course and Teacher domains, particularly for neutral sentiment.

---

# Error Analysis

Generated:

- Misclassified instances
- Neutral sentiment errors
- Confusion matrices
- Classification reports

Files produced:

```text
*_errors.csv
*_neutral_errors.csv
*_classification_report.csv
*_metrics.csv
```

---

# Multidomain vs Cross-Domain Comparison

Winner analysis performed using shared row_id values.

Agreement Rates:

| Domain | Agreement |
|----------|---------:|
| Course | 91.94% |
| Teacher | 95.38% |
| University | 98.34% |

Winner Analysis:

| Model | Wins |
|----------|---------:|
| Multidomain Gemini | 48 |
| Cross-Domain Gemini | 49 |

Result:

Both approaches produced nearly identical performance.

---

# Output Files

Generated outputs include:

```text
predictions.csv
predictions_checkpoint.csv
*_metrics.csv
*_errors.csv
*_neutral_errors.csv
*_classification_report.csv
gemini_zero_shot_vs_few_shot_summary.csv
gemini_overall_multidomain_results.csv
gemini_overall_summary.csv
```

---

# Required Packages

Install:

```bash
pip install google-genai
pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install tqdm
pip install openpyxl
pip install scipy
```

---

# Import Dependencies

```python
import os
import time
import json
import random
import warnings

import numpy as np
import pandas as pd

from tqdm import tqdm

from google import genai

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

import matplotlib.pyplot as plt
import seaborn as sns
```

---

# Reproducing the Experiments

1. Mount Google Drive
2. Upload datasets
3. Configure Google API key
4. Load datasets
5. Generate multidomain dataset
6. Run EDA section
7. Run Zero-Shot experiments
8. Run Few-Shot experiments
9. Generate evaluation metrics
10. Generate confusion matrices
11. Run winner analysis
12. Export dissertation tables

---

# Dissertation Contribution

This study demonstrates that Gemini 2.5 Flash can achieve competitive Aspect-Based Sentiment Analysis performance without task-specific fine-tuning. Few-shot prompting improved performance in Course and Teacher domains, while zero-shot prompting remained highly effective in the University domain. Multidomain and cross-domain configurations produced nearly identical performance, highlighting the strong generalization capability of modern generative language models.