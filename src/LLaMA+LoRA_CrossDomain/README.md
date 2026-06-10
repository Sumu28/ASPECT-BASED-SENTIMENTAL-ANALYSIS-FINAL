# Cross-Domain Aspect-Based Sentiment Analysis with LLaMA 3 + LoRA

This project applies **parameter-efficient fine-tuning (PEFT)** using **LoRA (Low-Rank Adaptation)** on top of **Meta's LLaMA 3 (8B Instruct)** model for cross-domain Aspect-Based Sentiment Analysis (ABSA). The model is trained on two academic review domains and tested on a third unseen domain to evaluate cross-domain generalisation.

## Task

Given a review and a specific aspect, classify the sentiment as **positive**, **negative**, or **neutral**.

The three domains used are:
- **Course** – student reviews of university courses
- **Teacher** – student reviews of teachers/lecturers
- **University** – student reviews of universities

---

## Approach

- **Base model:** meta-llama/Meta-Llama-3-8B-Instruct
- **Fine-tuning method:** LoRA (via peft) on top of 4-bit quantised model
- **Quantisation:** 4-bit NF4 with double quantisation (BitsAndBytesConfig)
- **LoRA config:**
  - r = 8
  - lora_alpha = 16
  - target_modules = ["q_proj", "v_proj"]
  - lora_dropout = 0.05
  - bias = "none"
  - task_type = CAUSAL_LM
- **Class balancing:** Balanced training datasets + class-weighted loss
- **Optimiser:** paged_adamw_8bit
- **Batch size:** 1 with gradient accumulation steps of 8 (effective batch = 8)
- **Inference:** Greedy decoding, max_new_tokens=3
- **GPU:** NVIDIA RTX A6000
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
| Accuracy | **77.37%** |
| Macro F1 | 0.654 |
| Weighted F1 | 0.764 |

### Train: Course + Teacher → Test: University

| Metric | Score |
|---|---|
| Accuracy | **89.57%** |
| Macro F1 | 0.677 |
| Weighted F1 | 0.881 |

### Train: Teacher + University → Test: Course

| Metric | Score |
|---|---|
| Accuracy | **74.76%** |
| Macro F1 | 0.654 |
| Weighted F1 | 0.725 |

---

## Key Observations

- **University** reviews are the easiest to predict cross-domain (~90% accuracy), suggesting university sentiment language overlaps strongly with course and teacher reviews.
- **Course** reviews are the hardest target domain (~75%), indicating course-specific vocabulary is more domain-unique.
- **Neutral** sentiment remains the weakest class across all experiments due to lower support and more ambiguous language.
- LoRA with 4-bit quantisation significantly reduces memory usage compared to full fine-tuning while maintaining competitive performance.

---

## Comparison with Full Fine-Tuning (LLaMA only)

| Train → Test | LLaMA Full FT | LLaMA + LoRA |
|---|---|---|
| Course + Uni → Teacher | 85.91% | 77.37% |
| Course + Teacher → Uni | 93.36% | 89.57% |
| Teacher + Uni → Course | 74.17% | 74.76% |

> LoRA achieves comparable results with far less compute and memory, making it more practical for resource-constrained environments.

---

## Data Files

| File | Description |
|---|---|
| course_balanced-new.csv | Balanced training data – Course domain |
| teacher_balanced-new.csv | Balanced training data – Teacher domain |
| university_balanced-new.csv | Balanced training data – University domain |
| course_test.csv | Shared fixed test set – Course |
| teacher_test.csv | Shared fixed test set – Teacher |
| university_test.csv | Shared fixed test set – University |

Each CSV contains review, aspect, and sentiment columns.

---

## Output Files (per experiment)

- ***_predictions.csv** – per-sample true and predicted sentiments
- ***_metrics.csv** – accuracy, macro F1, weighted F1
- ***_model/** – saved LoRA adapter weights and tokenizer

---

## Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets peft bitsandbytes accelerate scikit-learn -U
```

A Hugging Face account with access to meta-llama/Meta-Llama-3-8B-Instruct is required.

---

## Running

Each notebook corresponds to one train/test split:

| Notebook | Train Domains | Test Domain |
|---|---|---|
| Course + Uni → Teach (llama+lora) | Course, University | Teacher |
| Course + Teacher → Uni (llama+lora) | Course, Teacher | University |
| Teacher + Uni → Course (llama+lora) | Teacher, University | Course |

---

## Environment

- GPU: NVIDIA RTX A6000
- CUDA: enabled
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True