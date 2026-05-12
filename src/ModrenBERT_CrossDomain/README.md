# Cross-Domain Aspect-Based Sentiment Analysis using Transformer Models

This project investigates cross-domain Aspect-Based Sentiment Analysis (ABSA) using transformer-based architectures including ModernBERT and LLaMA 3 with LoRA fine-tuning.

The work evaluates model generalisation across educational review domains:
- course
- teacher
- university

Cross-domain experiments are conducted by training on combinations of two domains and testing on an unseen target domain.

The project includes:
- hyperparameter tuning,
- weighted loss experiments,
- error analysis,
- confusion matrix analysis,
- shared fixed test sets with row-level comparison using unique row identifiers,
- and comparison between multidomain and cross-domain performance.

All source code is available in the `src` directory and documentation is stored in the `docs` directory.