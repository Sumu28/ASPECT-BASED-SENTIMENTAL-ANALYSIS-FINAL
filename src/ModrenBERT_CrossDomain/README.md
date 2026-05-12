# Cross-Domain Aspect-Based Sentiment Analysis using ModernBERT

This project investigates cross-domain Aspect-Based Sentiment Analysis (ABSA) using transformer-based architectures including ModernBERT.

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

Dataset and Results File link :
1. Course + Teacher on University : https://drive.google.com/drive/folders/1vz9Bp-C-r_jlT81tEyqLVgoLnTXygK1s?usp=drive_link

2. Course + University on Teacher : https://drive.google.com/drive/folders/1dM1lyxlaYUHVYusFLzG0Wm4EBZU0KAJc?usp=sharing

3. Teacher + University on Course : https://drive.google.com/drive/folders/1Szq07oNr4XZO80OFoPeZVzW4zYp0VSzw?usp=sharing