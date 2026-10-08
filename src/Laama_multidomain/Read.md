# Multidomain and Cross-domain LLaMA ABSA README

## Project Overview

This project implements a **Multidomain and Cross-domain Aspect-Based Sentiment Analysis (ABSA)** pipeline using **LLaMA 3.2 3B Instruct** for educational review datasets.

The project investigates:
- Multidomain learning
- Cross-domain learning
- Hypertuning strategies
- Learning-rate sensitivity
- Epoch sensitivity
- Neutral sentiment behavior
- Disagreement analysis
- Qualitative error analysis

The evaluation framework includes:
- Shared fixed test sets
- Row-level prediction tracking using `row_id`
- Domain-wise evaluation
- Cross-domain vs multidomain comparison
- Confusion matrix analysis
- Disagreement analysis
- Visualization generation

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

Additional generated fields:
- `label`
- `row_id`

---

# Folder Structure

Recommended structure:

```text
Dissertation/

├── shared_test_sets/
│
├── plots/
│
├── metrics/
│
├── multidomain_llama_hypertuning/
│   │
│   ├── lr_1e5/
│   │
│   ├── lr_5e5/
│   │
│   └── epoch_5/
│
├── final_model/
│
└── prediction_csvs/
```

---

# Required Packages

Install the following packages before running the notebook.

```python
pip install torch
pip install transformers
pip install datasets
pip install accelerate
pip install bitsandbytes
pip install scikit-learn
pip install pandas
pip install numpy
pip install matplotlib
pip install tqdm
pip install huggingface_hub
```

Recommended environment:
- Python 3.10+
- CUDA-enabled GPU
- RTX A6000 recommended
- Renku GPU session preferred

---

# Model Used

The project uses:

```python
meta-llama/Llama-3.2-3B-Instruct
```

Imported using:

```python
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
```

---

# HuggingFace Authentication

Before loading the model:

```python
from huggingface_hub import login

login("YOUR_HF_TOKEN")
```

You must have access to:

```text
meta-llama/Llama-3.2-3B-Instruct
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
- matplotlib
- datasets
- tqdm

Random seeds are fixed for reproducibility.

---

## 2. Load Datasets

Datasets are loaded from:

```python
course_df = pd.read_csv(
    "/home/jovyan/work/course_balanced.csv"
)
```

---

## 3. Create Label Mapping

Sentiment labels:

| Sentiment | Label |
|---|---|
| negative | 0 |
| neutral | 1 |
| positive | 2 |

---

## 4. Create Shared Test Sets

Each dataset is independently split using stratified splitting.

Example:

```python
course_train, course_test = train_test_split(
    course_df,
    test_size=0.2,
    stratify=course_df["label"],
    random_state=42
)
```

The resulting test sets are reused across:
- multidomain experiments
- cross-domain experiments

This ensures:
- aligned evaluation
- fair comparison
- consistent row-level analysis

---

## 5. Create Multidomain Dataset

Training data is combined:

```python
multidomain_train_df = pd.concat([
    course_train,
    teacher_train,
    university_train
])
```

---

## 6. Prompt Engineering

Training prompts are formatted as:

```text
Review:
...

Aspect:
...

Options:
positive
negative
neutral

Sentiment:
...
```

Example:

```python
def create_prompt(review, aspect, label=None):
```

---

## 7. Tokenization

Example:

```python
tokenizer(
    example["text"],
    truncation=True,
    padding="max_length",
    max_length=256
)
```

---

## 8. Model Loading

Example:

```python
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.bfloat16,
    device_map="auto"
)
```

Training uses:
- BF16
- gradient checkpointing
- 8-bit optimizer
- memory-efficient full finetuning

---

# Experiments Conducted

## Experiment 1 — Original Multidomain LLaMA

### Configuration

| Parameter | Value |
|---|---|
| Learning Rate | 1e-5 |
| Epochs | 3 |
| Max Length | 128 |

### Results

| Domain | Weighted F1 |
|---|---|
| Course | 0.5830 |
| Teacher | 0.6541 |
| University | 0.8567 |

Average F1:

```text
0.6979
```

### Key Findings

- severe positive ↔ neutral confusion
- weak neutral recall
- underfitting behavior
- unstable sentiment calibration

---

# Experiment 2 — Hypertuned Multidomain LLaMA (LR = 1e-5)

## Improvements Added

- sequence length increased to 256
- improved prompt engineering
- BF16 training enabled
- optimization strategy improved

### Results

| Domain | Weighted F1 |
|---|---|
| Course | 0.7824 |
| Teacher | 0.8080 |
| University | 0.9405 |

Average F1:

```text
0.8436
```

### Key Findings

- major reduction in neutral confusion
- stronger contextual understanding
- substantial multidomain improvement

---

# Experiment 3 — Final Hypertuned Multidomain LLaMA (LR = 5e-5)

## Final Best Configuration

| Parameter | Value |
|---|---|
| Learning Rate | 5e-5 |
| Epochs | 3 |
| Max Length | 256 |
| Batch Size | 1 |
| Gradient Accumulation | 16 |
| BF16 | Enabled |
| Gradient Checkpointing | Enabled |
| Optimizer | paged_adamw_8bit |

---

## Final Results

| Domain | Accuracy | Precision | Recall | Weighted F1 |
|---|---|---|---|---|
| Course | 0.8519 | 0.8537 | 0.8519 | 0.8476 |
| Teacher | 0.8329 | 0.8574 | 0.8329 | 0.8474 |
| University | 0.9775 | 0.9842 | 0.9775 | 0.9791 |

Average Weighted F1:

```text
0.8914
```

### Key Findings

- best overall multidomain configuration
- strong neutral sentiment improvement
- highly stable multidomain behavior
- major reduction in disagreement percentages

---

# Experiment 4 — Epoch Sensitivity Experiment (Epoch = 5)

## Goal

Investigate whether additional epochs improve generalization.

### Configuration

| Parameter | Value |
|---|---|
| Learning Rate | 5e-5 |
| Epochs | 5 |

### Findings

- training loss continued decreasing
- evaluation improvements became marginal
- convergence occurred around epoch 3
- slight overfitting observed in Course domain

---

# Cross-domain LLaMA Experiments

Cross-domain experiments were conducted by:
- training on two domains
- testing on the third domain

Examples:

```text
teacher + university → course
course + university → teacher
course + teacher → university
```

---

# Cross-domain Results

| Domain | Accuracy | Precision | Recall | Weighted F1 |
|---|---|---|---|---|
| Course | 0.7334 | 0.7421 | 0.7334 | 0.7329 |
| Teacher | 0.8459 | 0.8480 | 0.8459 | 0.8465 |
| University | 0.9336 | 0.9368 | 0.9336 | 0.9338 |

Average Weighted F1:

```text
0.8377
```

---

# Final Performance Comparison

| Pipeline | Average Weighted F1 |
|---|---|
| Original Multidomain LLaMA | 0.6979 |
| Hypertuned Multidomain (1e-5) | 0.8436 |
| Final Hypertuned Multidomain (5e-5) | 0.8914 |
| Cross-domain LLaMA | 0.8377 |

---

# Best Overall Pipeline

```text
Final Hypertuned Multidomain LLaMA
```

Main reasons:
- highest average F1
- best Course-domain performance
- best University-domain performance
- strong neutral sentiment recognition
- significantly reduced disagreement percentages

---

# Evaluation Metrics

The notebook evaluates:

- Accuracy
- Precision
- Recall
- Weighted F1
- Confusion Matrix
- Disagreement Percentage

---

# Disagreement Analysis

Cross-domain and multidomain predictions are merged using:

```python
row_id
```

This enables:
- row-level comparison
- disagreement tracking
- qualitative error analysis

---

# Qualitative Error Analysis

The notebook extracts:
- multidomain correct / cross-domain wrong examples
- cross-domain correct / multidomain wrong examples
- neutral sentiment failure cases
- positive ↔ neutral confusion examples

---

# How to Run the Notebook

## Step 1
Start a GPU session.

Recommended:
- RTX A6000
- BF16 enabled

---

## Step 2
Install required packages.

---

## Step 3
Authenticate HuggingFace access.

---

## Step 4
Upload datasets into:

```text
/home/jovyan/work/
```

Required files:

```text
course_balanced.csv
teacher_balanced.csv
university_balanced.csv
```

---

## Step 5
Run notebook cells sequentially.

Recommended order:

1. Imports
2. Dataset loading
3. Shared test set creation
4. Prompt generation
5. Tokenization
6. Model loading
7. Training
8. Prediction generation
9. Evaluation
10. Visualization
11. Disagreement analysis
12. Error analysis
13. Final summary generation

---

# Outputs Generated

## Models

Saved inside:

```text
multidomain_llama_hypertuning/*/final_model/
```

---

## Predictions

Saved inside:

```text
course_predictions_hypertuned.csv
teacher_predictions_hypertuned.csv
university_predictions_hypertuned.csv
```

---

## Metrics

Generated:
- weighted F1 tables
- comparison tables
- disagreement summaries
- evaluation reports

---

## Visualizations

Generated:
- F1-score plots
- confusion matrices
- multidomain vs cross-domain comparison plots

---

# Reproducibility Notes

The notebook ensures reproducibility using:

```python
SEED = 42
```

and fixed:
- shared test sets
- row_ids
- deterministic splitting

---

# Final Dissertation Findings

The experiments demonstrated that:

- LLaMA performance is highly sensitive to:
  - learning rate
  - sequence length
  - prompt engineering
  - optimization strategy

- Neutral sentiment classification is the most difficult ABSA challenge.

- Hypertuning substantially improved:
  - neutral recall
  - contextual understanding
  - multidomain stability
  - cross-domain competitiveness

- The original multidomain underperformance was primarily caused by:
  - suboptimal optimization strategy,
  - rather than the multidomain learning paradigm itself.

- Additional epochs produced diminishing generalization returns despite lower training loss.

---

# Dissertation Artifacts

All outputs are stored inside:

```text
/home/jovyan/work/Dissertation
```

---

# Backup

Final project archive:

```text
Dissertation_Backup.zip
```