# Fake Job Posting Classifier

A multi-modal deep learning system that detects fraudulent job postings using fine-tuned RoBERTa combined with structured feature embeddings. Built on the EMSCAD dataset — 17,880 real-world job ads with severe class imbalance (~4.8% fraudulent).

---

## Problem

Fraudulent job postings steal personal data and money from job seekers at scale. Traditional ML approaches using TF-IDF or Bag-of-Words fail here — they're non-contextual and collapse under extreme class imbalance. This project addresses both limitations with a transformer-based multi-modal architecture.

---

## Architecture

```
Text Input (title + description + requirements)
        │
   RoBERTa-base (fine-tuned, selective layer freezing)
        │
   Text Embeddings [768-dim]
        │
        ├──────────────────────────────┐
                                       │
Numerical Features         Categorical Features
(salary, has_logo, etc.)   (employment_type, education)
        │                              │
   Linear layer              nn.Embedding layers
        │                              │
        └──────────┬───────────────────┘
                   │
           Fusion / Concat
                   │
         Classification Head
                   │
            Fake / Legitimate
```

---

## Key Design Decisions

| Decision | Why |
|---|---|
| Selective layer freezing | Preserve RoBERTa's pretrained language knowledge; only fine-tune upper layers for fraud-specific patterns |
| Focal Loss (tuned α, γ) | Shifts learning focus to rare fraudulent samples; outperforms CrossEntropy on imbalanced data |
| `nn.Embedding` for categoricals | Avoids false ordinal assumptions from label encoding |
| Missing-value indicator features | Converts null fields into explicit fraud signals — deliberate omission is itself a fraud pattern |
| Layer-wise learning rates + warmup | Lower LR for frozen layers, higher for classification head; warmup prevents early instability |
| Gradient clipping + early stopping | Stabilizes training; prevents overfitting on minority class |

---

## Results

| Metric | Score |
|---|---|
| Precision (fraud class) | [your value] |
| Recall (fraud class) | [your value] |
| F1-Score (fraud class) | [your value] |
| ROC-AUC | [your value] |
| PR-AUC | [your value] |

> Metrics reported on held-out test set. PR-AUC is the primary metric given class imbalance.

---

## Dataset

**EMSCAD** — Employment Scam Aegean Dataset  
Source: [Kaggle — Real or Fake Job Posting Prediction](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)

- 17,880 job postings, 18 features
- ~866 fraudulent (~4.8%) — severe imbalance
- Mix of text fields (description, requirements, benefits) and structured fields (salary range, employment type, education level)

---

## Tech Stack

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)

- **Model:** `roberta-base` via HuggingFace Transformers
- **Training:** PyTorch, Focal Loss, AdamW optimizer
- **Hyperparameter tuning:** Optuna (TPE sampler)
- **Data:** Pandas, NumPy, Scikit-learn

---

## Project Structure

```
fake-job-posting-classifier/
│
├── fake_job_posting_classifier.ipynb   # Full pipeline notebook
├── README.md
└── requirements.txt
```

---

## How to Run

1. Clone the repo and open the notebook in Google Colab or Jupyter
2. Install dependencies:
   ```bash
   pip install torch transformers scikit-learn optuna pandas numpy
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) and place `fake_job_postings.csv` in the working directory
4. Run all cells sequentially

---

## What I Learned

- Transformer fine-tuning is significantly more effective than TF-IDF on fraud detection tasks where subtle linguistic cues matter
- Missing value patterns carry strong signal in fraud detection — deliberate omission of salary, company profile, or location correlates with fake postings
- Focal Loss with tuned γ meaningfully improves minority-class recall vs. standard CrossEntropy on 95:5 imbalanced datasets
