# Cross-Domain Aspect-Based Sentiment Analysis with LLaMA 3

This project investigates **cross-domain generalisation** for Aspect-Based Sentiment Analysis (ABSA) using Meta's **LLaMA 3 (8B Instruct)** model. The goal is to train on reviews from one or two academic domains and test performance on a held-out domain, measuring how well sentiment knowledge transfers.

## Task

Given a review and an aspect, classify the sentiment as **positive**, **negative**, or **neutral**.

The three domains used are:
- **Course** – student reviews of university courses
- **Teacher** – student reviews of teachers/lecturers
- **University** – student reviews of universities

---

## Approach

- **Base model:** `meta-llama/Meta-Llama-3-8B-Instruct`
- **Training:** Full fine-tuning (no LoRA/PEFT) with instruction-formatted prompts in the `[INST]` template
- **Class balancing:** Balanced training datasets + class-weighted loss (inverse frequency weights)
- **Optimiser:** `paged_adamw_8bit` with FP16 disabled, gradient checkpointing enabled
- **Epochs:** 5
- **Batch size:** 1 with gradient accumulation steps of 32 (effective batch = 32)
- **Inference:** Greedy decoding (`do_sample=False`), `max_new_tokens=5`
- **Platform:** [Renku / DCU HPC](https://renku.computing.dcu.ie)

### Prompt Format

```
<s>[INST] You are an expert in aspect-based sentiment analysis.
Classify the sentiment into: positive, negative, or neutral.
Review: {review}
Aspect: {aspect}
Sentiment:
[/INST] {sentiment}
```

---

## Experiments & Results

Each experiment trains on two domains and tests cross-domain on the third.

### Train: Course + University → Test: Teacher

| Metric | Score |
|---|---|
| Accuracy | **85.91%** |
| Macro F1 | 0.760 |
| Weighted F1 | 0.857 |

### Train: Course + Teacher → Test: University

| Metric | Score |
|---|---|
| Accuracy | **93.36%** |
| Macro F1 | 0.747 |
| Weighted F1 | 0.934 |

### Train: Teacher + University → Test: Course

| Metric | Score |
|---|---|
| Accuracy | **74.17%** |
| Macro F1 | 0.671 |
| Weighted F1 | 0.733 |

---

## Key Observations

- Cross-domain transfer is strongest when predicting on **University** reviews (93% accuracy), suggesting university sentiment is most similar in style to course and teacher reviews.
- Transfer to **Course** reviews is hardest (~74%), indicating course-specific language is more domain-specific.
- **Neutral** sentiment is consistently the hardest class to predict across all experiments, due to lower support and more ambiguous language.
- Class weighting helps partially counteract the neutral underrepresentation.

---

## Data Files

| File | Description |
|---|---|
| `course_balanced-new.csv` | Balanced training data – Course domain |
| `teacher_balanced-new.csv` | Balanced training data – Teacher domain |
| `university_balanced-new.csv` | Balanced training data – University domain |
| `course_test.csv` | Shared fixed test set – Course |
| `teacher_test.csv` | Shared fixed test set – Teacher |
| `university_test.csv` | Shared fixed test set – University |

Each CSV contains `review`, `aspect`, and `sentiment` columns (plus metadata columns in test sets).

---

## Output Files (per experiment)

Saved under `{experiment_name}_model/`, `*_predictions.csv`, and `*_metrics.csv`:

- **`*_predictions.csv`** – per-sample true and predicted sentiments
- **`*_metrics.csv`** – accuracy, macro F1, weighted F1
- **`*_model/`** – saved model weights and tokenizer

---

## Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft bitsandbytes accelerate scikit-learn -U
```

A Hugging Face account with access to `meta-llama/Meta-Llama-3-8B-Instruct` is required. Set your token at runtime when prompted.

---

## Running

Open the relevant notebook on Renku or locally and run all cells. Each notebook corresponds to one train/test split:

| Notebook | Train Domains | Test Domain |
|---|---|---|
| `Course + Uni → Teach (LLaMA)` | Course, University | Teacher |
| `Course + Teach → Uni (LLaMA)` | Course, Teacher | University |
| `Teach + Uni → Course (LLaMA)` | Teacher, University | Course |

---

## Environment

- GPU: NVIDIA (confirmed via `nvidia-smi`)
- CUDA: enabled
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (set to reduce fragmentation)