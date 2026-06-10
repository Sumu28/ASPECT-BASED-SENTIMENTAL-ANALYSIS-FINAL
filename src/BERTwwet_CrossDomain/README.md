# BERTweet Cross-Domain Aspect-Based Sentiment Analysis

This repository contains three cross-domain transfer learning experiments using `vinai/bertweet-base` for Aspect-Based Sentiment Analysis (ABSA) across three educational review domains: **Course**, **Teacher**, and **University**.

Each notebook trains on two domains and tests on the held-out third domain, evaluating how well sentiment knowledge transfers across domains.

---

## Experiments

| Notebook | Train Domains | Test Domain |
|---|---|---|
| Teacher + University → Course | Teacher, University | Course |
| Course + Teacher → University | Course, Teacher | University |
| Course + University → Teacher | Course, University | Teacher |

---

## Dataset

All data is loaded from Google Drive under `/content/drive/MyDrive/ABSA_BERT/` and consists of:

- `course_balanced-new.csv` — balanced training data for Course domain
- `teacher_balanced-new.csv` — balanced training data for Teacher domain
- `university_balanced-new.csv` — balanced training data for University domain
- `course_test.csv`, `teacher_test.csv`, `university_test.csv` — fixed held-out test sets (844 samples each)

Each dataset contains: `review`, `aspect`, `sentiment`, `domain`, `row_id`

Sentiment labels: `negative` (0), `neutral` (1), `positive` (2)

Train/validation split is 90/10 with stratification by label. The test set is never used during training.

---

## Input Format

Each sample is formatted as a structured string before tokenization:

```
[DOMAIN] {domain} [ASPECT] {aspect} [TEXT] {review}
```

This format explicitly encodes domain and aspect context into the input so the model can attend to them during classification.

---

## Model

- Base model: `vinai/bertweet-base` (RoBERTa-based, pretrained on 850M English tweets)
- Task head: sequence classification with 3 output labels
- Tokenizer: BERTweet tokenizer with `normalization=True`, max length 128

---

## Training Setup

Each notebook runs two experiments using a shared `run_experiment()` function:

- Experiment 1: default learning rate (1e-5), 5 epochs
- Experiment 2: higher learning rate (3e-5), 5 epochs

Common hyperparameters across all runs:

| Parameter | Value |
|---|---|
| Batch size (train/eval) | 16 |
| Epochs | 5 |
| Weight decay | 0.01 |
| Optimizer | AdamW (via HuggingFace Trainer) |
| Mixed precision | fp16 |
| Class weighting | Balanced (via sklearn) |

Class weights are computed from the training set using `compute_class_weight("balanced")` and applied through a custom `WeightedTrainer` that overrides `compute_loss` with a weighted cross-entropy loss. This helps handle class imbalance, especially for the neutral class.

---

## Outputs

All outputs are saved to Google Drive. For each experiment run, the following files are written:

| File | Description |
|---|---|
| `{name}_classification_report.csv` | Per-class precision, recall, F1 |
| `{name}_confusion_matrix.csv` | 3x3 confusion matrix |
| `{name}_training_history.csv` | Loss and metrics per epoch |
| `{name}_predictions.csv` | Full predictions on test set |
| `{name}_errors.csv` | Misclassified samples |
| `{name}_neutral_errors.csv` | Errors where true label is neutral |
| `{name}_metrics.csv` | Final accuracy, macro F1, weighted F1 |
| `models/{name}/` | Saved model weights and tokenizer |

---

## Results Summary

### Teacher + University → Course (Test: Course domain)

| Experiment | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| 5 Epochs (LR=1e-5) | 0.7844 | 0.7134 | 0.7802 |
| LR=3e-5 | 0.7784 | 0.6958 | 0.7669 |

### Course + Teacher → University (Test: University domain)

| Experiment | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| 5 Epochs (LR=1e-5) | 0.9325 | 0.7395 | 0.9357 |
| LR=3e-5 | 0.9159 | 0.7103 | 0.9192 |

### Course + University → Teacher (Test: Teacher domain)

| Experiment | Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|
| 5 Epochs (LR=1e-5) | 0.8353 | 0.7069 | 0.8288 |
| LR=3e-5 | — | — | — |

The University domain consistently achieves the highest transfer accuracy, likely due to the more formal and uniform language in university reviews. The neutral class is the hardest to transfer across all three experiments.

---

## Evaluation Metrics

- Accuracy: overall proportion of correct predictions
- Macro F1: unweighted average F1 across all three classes — gives equal weight to minority classes
- Weighted F1: F1 weighted by class support — reflects real-world class distribution

Macro F1 is the primary metric reported in the dissertation, as it penalises poor performance on the neutral class equally.

---

## Requirements

```
torch
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
```

Running on Google Colab with a GPU runtime (T4 or better) is recommended. Each experiment run takes approximately 5–6 minutes.

---

## Notes

- The UNEXPECTED keys in the model load report (e.g. `lm_head`, `roberta.pooler`) are expected and can be safely ignored — they come from loading a masked LM checkpoint into a sequence classification head.
- The MISSING classifier weights are newly initialised for the classification task, which is standard fine-tuning behaviour.
- The emoji package is not required but improves tokenization if installed: `pip install emoji==0.6.0`
