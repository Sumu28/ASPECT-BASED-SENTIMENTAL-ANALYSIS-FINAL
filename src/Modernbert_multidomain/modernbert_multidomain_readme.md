# ModernBERT Multidomain ABSA README

## Project Overview

This project implements a **Multidomain Aspect-Based Sentiment Analysis (ABSA)** pipeline using **ModernBERT** for educational review datasets.

The project compares:
- Multidomain learning
- Cross-domain learning
- Weighted vs non-weighted loss
- Different learning rates
- Different epoch configurations

The evaluation framework includes:
- Shared fixed test sets
- Row-level prediction tracking using `row_id`
- Domain-wise evaluation
- Error analysis
- Neutral sentiment analysis
- Multidomain vs cross-domain disagreement analysis

---

# Dataset Structure

The project uses three educational review datasets:

| Dataset | Description |
|---|---|
| `course_balanced.csv` | Course review dataset |
| `teacher_balanced.csv` | Teacher review dataset |
| `university_balanced.csv` | University review dataset |

Each dataset contains:
- `review`
- `aspect`
- `sentiment`
- `domain`

---

# Folder Structure

Recommended structure:

```text
ABSA/
├── datasets/
│   ├── course_balanced.csv
│   ├── teacher_balanced.csv
│   └── university_balanced.csv
│
├── test_sets/
│   ├── course_test.csv
│   ├── teacher_test.csv
│   └── university_test.csv
│
├── models/
├── predictions/
├── results/
├── figures/
└── notebooks/
```

---

# Required Packages

Install the following packages before running the notebook.

```python
pip install transformers
pip install datasets
pip install accelerate
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install torch
```

Recommended environment:
- Python 3.10+
- CUDA-enabled GPU
- Google Colab GPU runtime preferred

---

# Model Used

The project uses:

```python
answerdotai/ModernBERT-base
```

Imported using:

```python
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
```

---

# Notebook Workflow

## 1. Import Libraries

The notebook imports:
- pandas
- numpy
- torch
- sklearn
- transformers
- datasets
- matplotlib

Random seeds are fixed for reproducibility.

---

## 2. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## 3. Load Datasets

Datasets are loaded from Google Drive.

Example:

```python
course_df = pd.read_csv(
    "/content/drive/MyDrive/ABSA/course_balanced.csv"
)
```

---

## 4. Create Row IDs

Each dataset receives stable row identifiers.

Example:

```python
course_df["row_id"] = [
    f"course_{i}" for i in range(len(course_df))
]
```

This enables:
- reproducible evaluation
- row-level comparison
- disagreement analysis

---

## 5. Input Formatting

Input text is formatted as:

```text
[DOMAIN] ... [ASPECT] ... [TEXT] ...
```

Example:

```python
def format_input(row):

    return (
        "[DOMAIN] " + row["domain"] +
        " [ASPECT] " + row["aspect"] +
        " [TEXT] " + row["review"]
    )
```

---

## 6. Label Mapping

Sentiment labels are mapped:

| Sentiment | Label |
|---|---|
| negative | 0 |
| neutral | 1 |
| positive | 2 |

---

## 7. Shared Test Set Creation

Each domain is split independently using stratified splitting.

Example:

```python
course_train, course_test = train_test_split(
    course_df,
    test_size=0.2,
    stratify=course_df["label"],
    random_state=42
)
```

The resulting test sets are saved:

```python
teacher_test.to_csv(...)
course_test.to_csv(...)
university_test.to_csv(...)
```

These shared test sets are reused across:
- multidomain experiments
- cross-domain experiments

This ensures:
- aligned evaluation
- consistent row_ids
- fair comparison

---

## 8. Multidomain Training Setup

Training data:

```python
train_df = pd.concat([
    course_train,
    teacher_train,
    university_train
])
```

Validation split:

```python
train_df, val_df = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=42
)
```

---

## 9. HuggingFace Dataset Conversion

Example:

```python
train_dataset = Dataset.from_pandas(
    train_df[["input_text", "label"]]
)
```

---

## 10. Tokenization

Example:

```python
def tokenize_function(examples):

    return tokenizer(
        examples["input_text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )
```

---

## 11. Weighted Loss

Class imbalance is handled using weighted cross-entropy loss.

Weights are computed using:

```python
compute_class_weight()
```

Custom trainer:

```python
class WeightedTrainer(Trainer):
```

---

## 12. Training Configuration

Best configuration:

| Parameter | Value |
|---|---|
| Learning Rate | 3e-5 |
| Epochs | 5 |
| Batch Size | 16 |
| Weighted Loss | Enabled |
| Weight Decay | 0.01 |

---

## 13. Experiments Conducted

### Learning Rate Experiments
- 1e-5
- 2e-5
- 3e-5

### Epoch Experiments
- 4 epochs
- 5 epochs

### Loss Experiments
- Weighted loss
- Non-weighted loss

---

# Evaluation Metrics

The notebook evaluates:

- Accuracy
- Macro F1
- Weighted F1
- Precision
- Recall
- Confusion Matrix

---

# Error Analysis

The notebook performs:

- misclassification analysis
- neutral sentiment analysis
- disagreement analysis
- qualitative example extraction

---

# Multidomain vs Cross-domain Comparison

Cross-domain predictions are merged using:

```python
row_id
```

This enables:
- row-level comparison
- disagreement tracking
- model behavior analysis

---

# Key Findings

## Best Overall Configuration

| Metric | Score |
|---|---|
| Accuracy | 0.8835 |
| Macro F1 | 0.8056 |
| Neutral F1 | 0.6000 |

Best setup:
- ModernBERT
- LR = 3e-5
- 5 epochs
- Weighted loss

---

## Main Research Findings

### Multidomain Outperformed Cross-domain

Disagreement analysis:

| Result | Count |
|---|---|
| Multidomain Correct / Cross-domain Wrong | 181 |
| Cross-domain Correct / Multidomain Wrong | 89 |

---

### Neutral Sentiment Was Most Difficult

Frequent confusion:
- neutral vs negative
- neutral vs positive
- mixed polarity reviews
- conditional sentiment statements

---

# How to Run the Notebook

## Step 1
Enable GPU runtime:

```text
Runtime → Change runtime type → GPU
```

---

## Step 2
Mount Google Drive.

---

## Step 3
Upload datasets into:

```text
/content/drive/MyDrive/ABSA/
```

---

## Step 4
Run notebook cells sequentially.

Recommended order:
1. Imports
2. Dataset loading
3. Row ID creation
4. Input formatting
5. Label mapping
6. Shared test set creation
7. Dataset preparation
8. Tokenization
9. Training
10. Evaluation
11. Error analysis
12. Comparison analysis
13. Plot generation

---

# Outputs Generated

## Models
Saved inside:

```text
ABSA/models/
```

---

## Predictions
Saved inside:

```text
ABSA/predictions/
```

---

## Figures
Saved inside:

```text
ABSA/results/
```

Includes:
- Macro F1 plots
- Neutral F1 plots
- Multidomain vs cross-domain plots
- Disagreement analysis plots

---

# Reproducibility Notes

The notebook ensures reproducibility using:

```python
SEED = 42
```

and fixed:
- train/test splits
- row_ids
- shared test sets

---

# Author Notes

This project was developed for MSc dissertation research in:
- Aspect-Based Sentiment Analysis (ABSA)
- Educational review mining
- Domain generalization using transformer architectures

The final evaluation framework supports:
- reproducible experiments
- scientifically aligned comparisons
- row-level prediction analysis
- qualitative sentiment evaluation

