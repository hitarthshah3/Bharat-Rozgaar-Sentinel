# Bharat Rozgar Sentinel architecture

## Full project architecture

```text
Employment Scam Aegean CSV (the one approved dataset)
  → schema checks, missing-value handling, seeded stratified train/validation/test split
  → NLP inputs: field-aware posting text; structured inputs: categorical and binary fields
  → baseline: TF-IDF word uni/bi-grams + Logistic Regression
  → deep-learning comparisons:
      Word2Vec + BiLSTM (text)
      Hybrid Word2Vec + BiLSTM (text) + metadata neural branch (new experiment)
      DistilBERT fine-tuning (planned transformer comparison)
  → validation checkpoint and threshold selection; final held-out evaluation
  → metrics, error analysis, risk indicators and explainable evidence
  → FastAPI inference service
  → React user interface / local prototype
```

All model fitting and threshold decisions use the training/validation partitions. The held-out test partition is used for final reporting. Inference returns a probability-based AI risk assessment with a selected model and readable indicators; it does not establish proof of fraud.

## Current prototype architecture

```text
Browser form
  → FastAPI page and prediction endpoint
  → saved TF-IDF + Logistic Regression pipeline
  → fraud/genuine probabilities, risk band and text indicators
  → browser displays a cautious “potentially fraudulent” assessment
```

The prototype currently uses the trained baseline and serves at `http://127.0.0.1:8000/` when launched with `Launch Demo.bat`. It is self-contained and does not require the React development server. The new hybrid model is a research/training experiment and is not automatically selected by the prototype.

## Work completed through the prototype

- Dataset discovery, validation, EDA, and missing-value summary.
- TF-IDF + Logistic Regression baseline and Word2Vec + BiLSTM implementation/training.
- Metrics, comparison outputs, explainability indicators, FastAPI service, React client, and local demonstration page.
- Baseline-powered prototype launched and checked with a sample request.
- Literature review and hybrid text-plus-metadata BiLSTM experiment added; training results will be filled in after execution.

## Future prospects

- Fine-tune DistilBERT and compare cost, inference latency, PR-AUC and fraud-class recall against the baseline and BiLSTM.
- Run repeated stratified seeds or cross-validation to measure result variability, while preserving an untouched final test set.
- Add temporal and platform-shift evaluation when a separately approved, labeled dataset becomes available.
- Calibrate probabilities and choose thresholds according to the review team's tolerance for missed scams versus false alarms.
- Expand human-readable explanations, confidence/uncertainty display and a human-review feedback loop.
- Add multilingual support for Indian job listings, with careful evaluation per language and regional terminology.
- Explore privacy-preserving and secure platform workflows, such as recruiter verification and suspicious-contact reporting, without treating those signals as proof.
- Measure model size, CPU/GPU latency and operational cost before selecting a deployable model.

These are future directions, not capabilities claimed by the current prototype.
