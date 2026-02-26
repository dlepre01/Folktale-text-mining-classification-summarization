# Folktale Text Mining: Classification and Summarization

## Overview

This project applies text mining and machine learning techniques to classify and summarize folktales based on the Aarne–Thompson–Uther (ATU) index.

The study combines traditional NLP pipelines, supervised classification models, transformer-based abstractive summarization, and graph-based extractive summarization.

Developed as part of the course "Text Mining and Search" (a.y. 2024/2025).

---

## Objectives

1. Classify folktales into predefined ATU genres.
2. Compare TF-IDF and embedding-based representations.
3. Generate abstractive summaries using a transformer model.
4. Generate extractive summaries using TextRank.
5. Evaluate summarization quality using ROUGE metrics.

---

## Dataset

Two structured datasets were used:

- ATU dataset (2247 rows): genre (chapter), tale ID, plot summary.
- AFT dataset (1518 rows): tale ID, title, provenance, full narrative text.

Datasets were merged via `atu_id` to associate each tale with its genre.

For classification, classes were consolidated into 5 genres:
- Anecdotes and Jokes
- Animal Tales
- Formula Tales
- Religious Tales
- Tales of Magic

Stratified split:
- 90% training
- 10% validation

---

## Text Preprocessing

Text normalization performed using spaCy (`en_core_web_lg`):

- Tokenization
- Lemmatization
- Stopword removal
- POS tagging
- Named Entity Recognition
- PyTextRank integration (for extractive summarization)

TF-IDF representation:
- Unigrams and bigrams
- GridSearchCV for hyperparameter tuning
- min_df tuning
- Class imbalance handling

---

## Classification Models

Two representation strategies were tested:

### 1. TF-IDF

Models:
- LinearSVC
- ComplementNB

Best Model: LinearSVC

Validation results:
- Accuracy: 0.81
- Weighted F1: 0.81

TF-IDF outperformed embedding-based approaches.

### 2. Non-contextual Word Embeddings

Document vectors (300-d) from spaCy tok2vec.

Models:
- Linear SVM
- k-NN
- Gaussian SVM
- Random Forest

Best accuracy: 0.67 (Linear SVM)

Conclusion: TF-IDF performed significantly better than averaged embeddings.

---

## Text Summarization

Two-step approach:

### Abstractive Summarization

Model:
- BART (facebook/bart-large-cnn)

Pipeline:
- Tokenization
- Chunking for long texts (512-token limit)
- Beam search (num_beams=4)
- max_length=128
- min_length=64

GPU recommended for performance.

### Extractive Summarization

Algorithm:
- TextRank (PyTextRank integration)

Procedure:
- Sentence ranking
- Number of sentences matched to abstractive summary
- Hyperparameter tuning for `limit_phrases`

---

## Evaluation

Summarization evaluated using ROUGE metrics:

Average scores (22 Italian folktales):

- ROUGE-1: 0.564
- ROUGE-2: 0.360
- ROUGE-L: 0.390

Extractive summaries showed moderate lexical overlap with abstractive references.

---

## Tech Stack

- Python
- spaCy
- PyTextRank
- scikit-learn
- Hugging Face Transformers
- BART
- ROUGE
- Pandas / NumPy

---

## Key Insights

- TF-IDF with LinearSVC outperforms embedding-based classifiers in this narrative genre classification task.
- Averaging word embeddings may dilute semantic signals in long narrative texts.
- Transformer-based abstractive summarization provides strong semantic condensation but requires significant computational resources.
- Extractive summarization remains competitive and interpretable.

---

## Possible Improvements

- Contextual embeddings (BERT)
- Fine-tuned transformer classification
- Larger summarization dataset
- Human evaluation of summary coherence
