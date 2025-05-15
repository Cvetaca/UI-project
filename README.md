# Fake News Detection with BERT

This project implements a fake news detection system using BERT and PyTorch. The model classifies news articles as either **FAKE** or **REAL**.

## Features

- Utilizes the `bert-base-uncased` model from HuggingFace Transformers.
- Custom `NewsDataset` class for efficient data loading and tokenization.
- 5-Fold Cross Validation for robust evaluation.
- Training and validation loops with accuracy reporting.

## Project Structure

- `ui_bert.py`: Main script for data loading, model definition, training, and evaluation.
- `full_dataset.csv`: CSV file containing news articles and labels.

## Usage

1. **Install dependencies:**
    ```bash
    pip install pandas torch scikit-learn transformers
    ```

2. **Prepare your dataset:**
    - Ensure `full_dataset.csv` is in the project directory.
    - The CSV should have columns: `text` (news content) and `label` (`FAKE` or `REAL`).

3. **Run the script:**
    ```bash
    python ui_bert.py
    ```

## Model Overview

- **Tokenizer:** BERT tokenizer (`bert-base-uncased`)
- **Model:** BERT with a frozen encoder and a linear classification head
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam

## Cross Validation

The script uses 5-fold cross validation to split the dataset and reports accuracy for each fold.

## Custom Classes

- **NewsDataset:** Handles tokenization and formatting for BERT input.
- **FakeNewsClassifier:** Wraps BERT and adds a linear layer for classification.

## Notes

- BERT parameters are frozen during training for efficiency.
- Adjust `n_epochs` and `batch_size` in `ui_bert.py` as needed.

## License

This project is for educational purposes.
