# Multidomain Aspect-Based Sentiment Analysis using BERTweet

This repository contains the complete implementation of a research-grade multidomain Aspect-Based Sentiment Analysis (ABSA) pipeline using the BERTweet transformer model.

The project investigates whether multidomain learning improves sentiment classification performance compared to cross-domain transfer learning across educational-review datasets.

The implementation includes:

- Multidomain training
- Hyperparameter tuning
- Weighted-loss optimization
- Domain-wise evaluation
- Cross-domain comparison
- Disagreement analysis
- Neutral sentiment analysis
- Visualization pipeline
- Dissertation-ready table exports

Based on notebook implementation and experiments from: :contentReference[oaicite:0]{index=0}

---

# Domains Used

The experiments were conducted on three educational-review domains:

1. Course Reviews
2. Teacher Reviews
3. University Reviews

Each domain contains:

- Review text
- Aspect term
- Sentiment label
- Domain identifier
- Shared row_id for comparison analysis

---

# Model Used

| Component | Value |
|---|---|
| Base Model | vinai/bertweet-base |
| Architecture | Transformer |
| Task | Aspect-Based Sentiment Analysis |
| Classes | Negative, Neutral, Positive |

---

# Final Best Configuration

| Parameter | Value |
|---|---|
| Learning Rate | 2e-5 |
| Batch Size | 8 |
| Epochs | 7 |
| Weighted Loss | Enabled |
| Max Length | 128 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |

---

# Final Results

## Domain-wise Performance

| Domain | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| Course | 0.945498 | 0.937243 | 0.945623 |
| Teacher | 0.960900 | 0.939534 | 0.960667 |
| University | 0.996445 | 0.983757 | 0.996372 |

---

## Overall Performance

| Metric | Score |
|---|---|
| Mean Accuracy | 0.967615 |
| Mean Macro F1 | 0.953511 |
| Mean Weighted F1 | 0.967554 |

---

# Hyperparameter Tuning Summary

| Configuration | Weighted F1 |
|---|---|
| Batch8_Epoch7_FINAL | 0.889832 |
| Batch8_Epoch3 | 0.885685 |
| Batch8 | 0.883750 |
| Batch32 | 0.881994 |
| Baseline_BS16_E5_LR2e5 | 0.881769 |
| LR3e5 | 0.878705 |
| LR1e5 | 0.869966 |

---

# Multidomain vs Cross-domain Results

| Domain | Multidomain Better | Cross-domain Better |
|---|---|---|
| Course | 156 | 15 |
| Teacher | 102 | 4 |
| University | 64 | 0 |

Overall multidomain learning significantly outperformed cross-domain learning across all educational-review datasets.

---

# Key Findings

## 1. Multidomain Learning Improved Generalization

Multidomain BERTweet consistently achieved superior contextual sentiment understanding compared to cross-domain learning.

---

## 2. Weighted Loss Improved Neutral Sentiment Classification

The weighted-loss strategy substantially improved minority-class learning, especially for neutral sentiment prediction.

---

## 3. Smaller Batch Sizes Performed Better

Batch size 8 produced the strongest generalization performance across experiments.

---

## 4. Validation Loss Was Not the Best Early-Stopping Indicator

Although validation loss increased during later epochs, classification metrics continued improving.

This indicates that validation loss alone was not the best stopping criterion for multidomain ABSA.

---

## 5. Domain Difficulty Varied

| Domain | Difficulty |
|---|---|
| Course | Hardest |
| Teacher | Moderate |
| University | Easiest |

---

# Folder Structure

```bash
ABSA_BERT_MULTIDOMAIN/
│
├── models/
├── predictions/
├── results/
├── training_history/
│
├── course_balanced.csv
├── teacher_balanced.csv
├── university_balanced.csv
│
├── course_test.csv
├── teacher_test.csv
├── university_test.csv
│
└── bertweet_multidomain.ipynb
```

---

# Required Packages

Install all dependencies before running the notebook.

```bash
pip install -q transformers datasets accelerate evaluate scikit-learn emoji==0.6.0
```

Additional libraries used:

```python
torch
pandas
numpy
matplotlib
datasets
transformers
sklearn
```

---

# Hardware Requirements

Recommended:

| Component | Requirement |
|---|---|
| GPU | NVIDIA Tesla T4 or better |
| VRAM | 16GB recommended |
| RAM | 16GB+ |
| Platform | Google Colab GPU |

---

# How to Run the Project

## Step 1 — Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

---

## Step 2 — Set Base Directory

```python
BASE_PATH = "/content/drive/MyDrive/ABSA_BERT_MULTIDOMAIN"
```

---

## Step 3 — Place Dataset Files

Place the following CSV files inside the project directory:

```bash
course_balanced.csv
teacher_balanced.csv
university_balanced.csv

course_test.csv
teacher_test.csv
university_test.csv
```

---

## Step 4 — Run Notebook Cells Sequentially

Run all notebook cells in order:

1. Package installation
2. Dataset loading
3. Preprocessing
4. Tokenization
5. Weighted trainer creation
6. Hyperparameter experiments
7. Final training
8. Evaluation
9. Cross-domain comparison
10. Visualization generation
11. Dissertation export

---

# Output Files Generated

The pipeline automatically exports:

## Predictions

```bash
predictions/
```

Contains:

- Final predictions CSVs
- Row-level outputs
- Sentiment labels

---

## Results

```bash
results/
```

Contains:

- Classification reports
- Confusion matrices
- Error analysis
- Neutral error analysis
- Cross-domain comparison summaries
- Final metrics
- Dissertation-ready tables
- PNG visualizations

---

## Training History

```bash
training_history/
```

Contains:

- Epoch-wise logs
- Validation metrics
- Loss tracking

---

# Visualization Outputs

The notebook automatically generates:

- Hyperparameter tuning plots
- Domain performance plots
- Confusion matrices
- Cross-domain superiority comparison charts
- Disagreement analysis charts

---

# Dataset Input Format

Each dataset CSV should contain:

| Column | Description |
|---|---|
| review | Review text |
| aspect | Aspect term |
| sentiment | Sentiment label |
| domain | Domain identifier |

---

# Sentiment Labels

| Label | Encoded Value |
|---|---|
| Negative | 0 |
| Neutral | 1 |
| Positive | 2 |

---

# Training Strategy

The multidomain dataset is created by concatenating all three balanced datasets.

Input format:

```text
[DOMAIN] domain_name [ASPECT] aspect_term [TEXT] review_text
```

This helps the model learn:

- domain-aware sentiment patterns
- aspect-sensitive contextual relationships
- cross-domain semantic representations

---

# Evaluation Metrics

The project evaluates:

- Accuracy
- Macro F1
- Weighted F1
- Precision
- Recall
- Confusion Matrix
- Error Analysis
- Neutral Error Analysis
- Disagreement Analysis

---

# Final Conclusion

Multidomain BERTweet demonstrated highly robust and generalizable aspect-based sentiment classification performance across educational-review domains.

The multidomain learning strategy significantly outperformed cross-domain learning in both quantitative performance and qualitative contextual understanding.

The final multidomain model achieved:

- Strong neutral sentiment understanding
- High cross-domain generalization
- Superior contextual semantic learning
- Research-grade ABSA performance

---
