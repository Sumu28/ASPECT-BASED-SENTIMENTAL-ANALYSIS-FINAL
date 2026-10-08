# Multidomain Aspect-Based Sentiment Analysis using LLaMA 3.2 3B + LoRA

## Project Overview

This project investigates **Multidomain Aspect-Based Sentiment Analysis (ABSA)** using **LLaMA 3.2 3B Instruct** with **LoRA (Low-Rank Adaptation)** fine-tuning.

The objective is to determine whether multidomain training can outperform traditional cross-domain training across multiple educational review domains.

The project forms part of an MSc Data Science dissertation and includes:

- Multidomain LoRA fine-tuning
- Hyperparameter tuning
- Cross-domain comparison
- Agreement and disagreement analysis
- Winner analysis
- Confusion matrix evaluation
- Dissertation-ready visualisations and tables

---

# Domains

The study uses three educational review domains:

1. Course Reviews
2. Teacher Reviews
3. University Reviews

Each review contains:

- Review text
- Aspect term
- Sentiment label
- Unique row identifier

Sentiment classes:

```text
positive
negative
neutral
```

---

# Dataset Files

Required datasets:

```text
course_balanced.csv
teacher_balanced.csv
university_balanced.csv
```

Required shared test sets:

```text
shared_test_sets/
├── course_test.csv
├── teacher_test.csv
└── university_test.csv
```

---

# Model

Base Model:

```text
Meta-Llama-3.2-3B-Instruct
```

Training Method:

```text
LoRA Fine-Tuning
```

Framework:

```text
Transformers
PEFT
BitsAndBytes
PyTorch
```

---

# Final Selected Configuration

The final multidomain LoRA configuration selected for the dissertation:

| Parameter | Value |
|------------|--------|
| Learning Rate | 5e-5 |
| Epochs | 5 |
| LoRA Rank | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Batch Size | 1 |
| Gradient Accumulation | 16 |
| Quantization | 4-bit NF4 |
| Optimizer | paged_adamw_8bit |
| Precision | BF16 |
| Gradient Checkpointing | Enabled |

---

# Project Structure

```text
Dissertation/
│
├── shared_test_sets/
│
├── multidomain_llama_lora_hypertuning/
│   │
│   ├── lr_1e5/
│   ├── lr_3e5/
│   ├── lr_5e5/
│   ├── epoch_7/
│   ├── rank_32/
│   └── comparison_analysis/
│
├── final_model/
│
└── predictions/
```

---

# Installation

Create a virtual environment:

```bash
python -m venv venv

source venv/bin/activate
```

Install required packages:

```bash
pip install torch transformers peft accelerate bitsandbytes
pip install datasets sentencepiece
pip install pandas numpy scikit-learn matplotlib tqdm
```

---

# Requirements

Create a file named:

```text
requirements.txt
```

Contents:

```text
torch
transformers
peft
accelerate
bitsandbytes
datasets
sentencepiece
pandas
numpy
scikit-learn
matplotlib
tqdm
```

Install using:

```bash
pip install -r requirements.txt
```

---

# Hardware Requirements

Recommended:

```text
GPU: NVIDIA RTX 3090 (24GB)
RAM: 32GB+
Disk Space: 100GB+
```

Training in this dissertation was performed using:

```text
RTX 3090
4-bit Quantization
BF16 Training
```

---

# Running the Project

## Step 1: Load Datasets

Place the following files in the working directory:

```text
course_balanced.csv
teacher_balanced.csv
university_balanced.csv
```

Load datasets:

```python
course_df = pd.read_csv("course_balanced.csv")
teacher_df = pd.read_csv("teacher_balanced.csv")
university_df = pd.read_csv("university_balanced.csv")
```

---

## Step 2: Create Shared Test Sets

Generate and save shared test sets for:

```text
Course
Teacher
University
```

These test sets must be reused for:

- Multidomain experiments
- Cross-domain experiments

to ensure fair comparison.

---

## Step 3: Create Multidomain Training Dataset

Combine training splits:

```python
multidomain_train = pd.concat([
    course_train,
    teacher_train,
    university_train
])
```

---

## Step 4: Load LLaMA Model

Load:

```text
Meta-Llama-3.2-3B-Instruct
```

using:

```python
AutoModelForCausalLM.from_pretrained()
```

with:

```text
4-bit NF4 Quantization
BF16
BitsAndBytes
```

---

## Step 5: Configure LoRA

```python
from peft import LoraConfig

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
)
```

---

## Step 6: Train Model

Training configuration:

```text
Learning Rate: 5e-5
Epochs: 5
Batch Size: 1
Gradient Accumulation: 16
Optimizer: paged_adamw_8bit
```

Run training:

```python
trainer.train()
```

---

## Step 7: Save Model

```python
model.save_pretrained(
    "./final_model"
)

tokenizer.save_pretrained(
    "./final_model"
)
```

---

## Step 8: Generate Predictions

Generate predictions separately for:

```text
Course
Teacher
University
```

Outputs:

```text
course_predictions.csv
teacher_predictions.csv
university_predictions.csv
```

---

## Step 9: Evaluation

Metrics generated:

- Accuracy
- Weighted Precision
- Weighted Recall
- Weighted F1
- Macro Precision
- Macro Recall
- Macro F1

Output:

```text
evaluation_summary.csv
```

---

## Step 10: Confusion Matrix Analysis

Generated files:

```text
course_confusion_matrix.png
teacher_confusion_matrix.png
university_confusion_matrix.png
```

---

## Step 11: Cross-Domain Comparison

Compare:

```text
Multidomain LoRA
vs
Cross-Domain LoRA
```

Generated files:

```text
multidomain_vs_crossdomain_results.csv
multidomain_vs_crossdomain_f1.png
```

---

## Step 12: Agreement Analysis

Calculate:

```text
Agreement Rate
Disagreement Rate
```

Outputs:

```text
agreement_summary.csv
```

---

## Step 13: Disagreement Analysis

Generated files:

```text
course_disagreements.csv
teacher_disagreements.csv
university_disagreements.csv
```

---

## Step 14: Winner Analysis

Generated files:

```text
course_multi_correct.csv
course_cross_correct.csv

teacher_multi_correct.csv
teacher_cross_correct.csv

university_multi_correct.csv
university_cross_correct.csv
```

These files identify:

- Cases where Multidomain was correct and Cross-Domain was wrong
- Cases where Cross-Domain was correct and Multidomain was wrong

---

# Experimental Results

## Baseline LoRA (LR=1e-5)

| Domain | Weighted F1 |
|----------|-----------:|
| Course | 0.6617 |
| Teacher | 0.7388 |
| University | 0.8798 |

Average F1:

```text
0.7601
```

---

## LR = 3e-5

| Domain | Weighted F1 |
|----------|-----------:|
| Course | 0.7569 |
| Teacher | 0.8056 |
| University | 0.9169 |

Average F1:

```text
0.8265
```

---

## Final Selected Model (LR=5e-5)

| Domain | Weighted F1 |
|----------|-----------:|
| Course | 0.7946 |
| Teacher | 0.8300 |
| University | 0.9556 |

Average F1:

```text
0.8601
```

---

## Epoch 7 Ablation

| Domain | Weighted F1 |
|----------|-----------:|
| Course | 0.8098 |
| Teacher | 0.8209 |
| University | 0.9558 |

Average F1:

```text
0.8622
```

Conclusion:

```text
The Epoch 5 configuration remained the preferred model
because it achieved nearly identical performance while
requiring less training time.
```

---

# Key Findings

- Multidomain LoRA outperformed Cross-Domain LoRA across all domains.
- Hyperparameter tuning improved average Weighted F1 from 0.7601 to 0.8601.
- University domain achieved the strongest performance.
- Winner analysis consistently favoured multidomain training.
- Additional training beyond five epochs produced minimal gains.
- LR=5e-5, Epoch=5 was selected as the final LoRA configuration.

---

# Author

**Sumukha Sagar**

MSc Data Science Dissertation

**Multidomain Aspect-Based Sentiment Analysis using LLaMA 3.2 3B and LoRA Fine-Tuning**