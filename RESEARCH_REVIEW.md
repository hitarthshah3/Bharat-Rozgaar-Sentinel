# Research review: online recruitment fraud detection

**Review date:** 18 October 2026  
**Scope:** Research directly about fraudulent job postings / online recruitment fraud. The sources below are academic publications. Results are those reported by the papers and are not directly comparable because data splits, preprocessing, labels, and metrics differ.

## Selected papers

| Paper | What it uses and reports | Limitation or gap |
|---|---|---|
| Ullah & Jamjoom (2023), [A smart secured framework for detecting and averting online recruitment fraud using ensemble machine learning techniques](https://doi.org/10.7717/peerj-cs.1234), *PeerJ Computer Science* | Uses the Employment Scam Aegean Dataset (17,880 postings; 866 fraudulent). Compares AdaBoost, XGBoost, voting and Random Forest after mode imputation, oversampling, label encoding and scaling. AdaBoost reports 98.374% accuracy. | It is a useful classical-ensemble benchmark, but the reported headline is accuracy on a very imbalanced benchmark. It does not establish how well the detector transfers to current postings or different platforms. The latter is a deployment/generalization gap inferred from the dataset and evaluation described, not a stated author conclusion. |
| Chapman, Skulmowski & Haan (2023), [A machine learning approach to detecting fraudulent job types](https://doi.org/10.1007/s00146-022-01469-0), *AI & Society* | Goes beyond real/fake classification and labels fraud types. Compares TF-IDF/BoW, empirical warning rules, POS counts, Word2Vec and BERT/RoBERTa/XLNet. Combining rules, POS and BoW with Gradient Boosting reports F1 0.88. | Authors explicitly note anonymization may alter language, the EMSCAD data is time/platform specific (collected 2012–2014), fraud types are homogeneous/imbalanced, and a single holdout does not show variability across runs. Corporate identity fraud was especially difficult to recall. |
| Akram et al. (2024), [Online Recruitment Fraud (ORF) Detection Using Deep Learning Approaches](https://doi.org/10.1109/ACCESS.2024.3435670), *IEEE Access* | Evaluates BERT and RoBERTa and several SMOTE-family balancing methods. On the imbalanced data, BERT reports 98.42% accuracy but only 65.50% fraud recall. BERT with SMOBD reports 89.84% recall (with lower 97.03% accuracy). | Their results demonstrate why accuracy alone hides missed scams and why balancing/threshold choices matter. The paper introduces/uses a combined multi-source dataset in parts of its description; that data is not adopted in this project. Its oversampling experiment is a method comparison, not proof that synthetic samples generalize to real new scams. |
| Sun et al. (2026), [Detecting cyber employment fraud in platform-based space: A multi-agent large language model approach](https://doi.org/10.1016/j.jcrimjus.2026.102692), *Journal of Criminal Justice* | Tests ChatGPT-4o single-pass scoring and a multi-agent flow (consistency, legitimacy and contextual validation), then feeds generated 0–10 risk scores into LR, Random Forest, XGBoost and MLP. Uses EMSCAD with Monte Carlo cross-validation. Reports improved metrics and a relationship between risk scores and fraud prevalence. Single-pass is cheaper; multi-agent is more explainable. | This is an LLM-assisted, API-dependent design, with added contextual validation through QCEW. Cost, latency, reproducibility and access to outside context make it a different deployment profile. Those operational concerns are our assessment; they are not presented here as a measured failure of the paper. This project will not call an LLM or use QCEW as a training source. |

## What these papers suggest

1. A high accuracy score can coexist with poor detection of the rare fraudulent class. Fraud recall, precision, F1, PR-AUC and the confusion matrix should be reported alongside accuracy and ROC-AUC.
2. Text-only and metadata/rule features capture different evidence. The 2023 fraud-type study found that combining linguistic/rule features helped; EMSCAD itself contains binary and categorical posting metadata as well as text.
3. The public benchmark is historical, imbalanced and limited in platform coverage. Results on it cannot establish current real-world performance. Temporal or platform-based external validation is a meaningful future step, subject to the project’s current one-dataset restriction.
4. Explanations and human review matter. Risk indicators should support a reviewer and job seeker, not state that fraud has been proven.

## Proposed project contribution

Add a **hybrid Word2Vec + BiLSTM model with metadata fusion**. The text branch learns sequence patterns from the posting. A separate small neural branch encodes the dataset’s categorical and binary fields, and the two learned representations are joined before the fraud classifier. This creates a direct text-only vs text-plus-metadata ablation on the same rows and split.

Also select a decision threshold on the validation split to maximize fraud-class F1, then evaluate the untouched test split at that threshold. Save a second report at the fixed 0.5 threshold so the effect of threshold selection remains visible. Report PR-AUC because the positive class is only 4.84% of this supplied dataset. Do not use the test split to choose epochs, features or threshold.

### Constraints and interpretation

- Training remains restricted to the one user-provided Employment Scam Aegean CSV.
- No paper’s benchmark numbers should be compared as if they came from this project’s identical split.
- Any future comparison with current postings requires an approved and properly labeled evaluation dataset; it is not part of this training run.
- A model score is an AI-based risk assessment, not definitive proof of fraud.

## References

- Ullah, Z. & Jamjoom, M. (2023). DOI: [10.7717/peerj-cs.1234](https://doi.org/10.7717/peerj-cs.1234).
- Chapman, S., Skulmowski, A. & Haan, P. (2023). DOI: [10.1007/s00146-022-01469-0](https://doi.org/10.1007/s00146-022-01469-0).
- Akram, N. et al. (2024). DOI: [10.1109/ACCESS.2024.3435670](https://doi.org/10.1109/ACCESS.2024.3435670).
- Sun, Q. et al. (2026). DOI: [10.1016/j.jcrimjus.2026.102692](https://doi.org/10.1016/j.jcrimjus.2026.102692).
