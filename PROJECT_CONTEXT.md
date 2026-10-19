# Bharat Rozgaar Sentinel — Project Context

## Project identity

- **Name:** Bharat Rozgaar Sentinel — AI-Powered Fraudulent Employment Detection System
- **Objective:** Provide an AI-based risk assessment of a job posting as genuine or potentially fraudulent, with probabilities, a risk level, selected model, and explainable indicators. Predictions are not proof of fraud.
- **Subjects:** Natural Language Processing; Deep Learning.
- **Problem:** Binary classification of job postings using the `fraudulent` label.
- **Dataset:** `data/fake_job_postings.csv`, the CSV supplied by the user.
- **Source:** Kaggle Employment Scam Aegean Dataset, https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction (source supplied by user; no other dataset used).
- **Current status:** Dataset discovery/validation, EDA, four trained classifier artifacts, evaluation reports, explainability, document extraction, and FastAPI inference are complete. The full responsive React website is built with Vite and served by FastAPI at `http://127.0.0.1:8000/`. Uploads support PDF/PNG/JPEG and editable extracted text. Empty assessments are rejected. The classifier was not retrained for document upload.

## Architecture

Dataset → preprocessing → field-aware text + structured features → TF-IDF + Logistic Regression baseline → Word2Vec + text-only BiLSTM and metadata-fusion BiLSTM → DistilBERT fine-tuning → evaluation and explainability → FastAPI → responsive React website (Vite production build). The site provides grouped text input, PDF/PNG/JPEG extraction with editable review, results, model metrics, methodology, dataset source, and limitations. `POST /extract` uses PyMuPDF and RapidOCR/ONNX Runtime; reviewed text then goes through `POST /predict`.

All splits are stratified and seeded; the held-out test split is reserved for final comparisons. The small positive class is handled with training-only class weighting where supported. Model artifacts are loaded explicitly at inference.

## Dataset information (measured from supplied CSV)

- Filename: `fake_job_postings.csv` (50,061,601 bytes)
- Records: 17,880
- Columns: `job_id`, `title`, `location`, `department`, `salary_range`, `company_profile`, `description`, `requirements`, `benefits`, `telecommuting`, `has_company_logo`, `has_questions`, `employment_type`, `required_experience`, `required_education`, `industry`, `function`, `fraudulent`
- Target: `fraudulent`; class 0 = 17,014, class 1 = 866 (4.84% positive)
- Missing/blank values (measured by `src.eda` on 2026-09-29): `salary_range` 15,012; `benefits` 7,220; `required_education` 8,105; `industry` 4,903; `department` 11,553; `employment_type` 3,471; `required_experience` 7,050; `function` 6,455; `company_profile` 3,308; `requirements` 2,698; `location` 346; `description` 2. Other columns have no blank values.
- Exact duplicate records: 0 (full row content compared)
- Preprocessing decisions: combine text fields without `job_id` or target, preserve field names as readable prefixes, normalize missing text to empty strings, keep label numeric, and exclude identifiers from model input. Split before fitting tokenizers/vectorizers to avoid vocabulary leakage. No resampling or synthetic examples.

## Models

| Model | Implementation | Training | Configuration / metrics / artifact / issues |
|---|---|---|---|
| TF-IDF + Logistic Regression | Implemented (`src/models/baseline/model.py`, `src/train.py`) | Complete | Word uni/bi-grams, 100k features, `min_df=2`, balanced class weights, seed 42; test metrics below; artifact `models/tfidf_lr/model.joblib`. |
| Word2Vec + BiLSTM | Implemented (`src/models/bilstm/model.py`, `src/train.py`) | Trained (parked) | Word2Vec trained only on training partition; 60k vocab, 256 tokens, 128d embeddings/hidden, dropout .35, batch 16, 3 epochs, positive weighted BCE, best validation F1 checkpoint; CPU; artifacts `models/word2vec_bilstm/model.pt` and `word2vec.model`; test metrics below. |
| Hybrid Word2Vec + BiLSTM + metadata | Implemented (`src/models/bilstm/model.py`, `src/train.py`, `src/inference/predict.py`) | Trained | Seed 42, CPU, 3 epochs, batch 32, max length 160, 128d embeddings/hidden, dropout .35, weighted BCE. Train-only one-hot schema for five categorical fields and three binary fields. Validation-selected threshold 0.610729; metrics below; artifact `models/hybrid_bilstm/model.pt` and `word2vec.model`. Inference and API route handler smoke checks passed. |
| Matched text-only BiLSTM ablation | Same BiLSTM architecture, separate artifact | Trained | Same split and configuration as hybrid, threshold 0.5; artifact under `models/word2vec_bilstm_ablation/`; report under `outputs/metrics/word2vec_bilstm_ablation.json`. |
| DistilBERT | Implemented (`src/train.py`, `src/inference/predict.py`) | Complete (one full-data epoch) | `distilbert-base-uncased` checkpoint cached locally. Seed 42, CPU, all 12,873 training rows, batch 32, max length 64, 2 CPU threads, learning rate 2e-5, weighted cross-entropy; full 1,431-row validation and 3,576-row test partitions. Validation F1 at 0.5 = 0.7186; validation-selected threshold = 0.984093 (validation F1 0.8254). Test metrics below at both selected and fixed thresholds. Artifact `models/distilbert/`; metadata `models/distilbert/training_metadata.json`. One epoch and 64-token truncation are compute constraints; multi-epoch/longer-context experiments remain future work. |

## Current best results

Only measured evaluation values belong here.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + Logistic Regression | 0.9899 | 0.8914 | 0.9017 | 0.8966 | 0.9934 |
| Word2Vec + BiLSTM | 0.9318 | 0.3489 | 0.4740 | 0.4020 | 0.8558 |
| Hybrid BiLSTM, validation-selected threshold | 0.9555 | 0.5625 | 0.3642 | 0.4421 | 0.9081 |
| Hybrid BiLSTM, threshold 0.5 | 0.9004 | 0.2752 | 0.6474 | 0.3862 | 0.9081 |
| Matched text-only BiLSTM ablation, threshold 0.5 | 0.9181 | 0.2887 | 0.4740 | 0.3589 | 0.8857 |
| DistilBERT, validation-selected threshold | 0.9824 | 0.9044 | 0.7110 | 0.7961 | 0.9806 |
| DistilBERT, threshold 0.5 | 0.9678 | 0.6239 | 0.8439 | 0.7174 | 0.9806 |

Measured average precision (PR-AUC): TF-IDF 0.9542; text BiLSTM 0.4241; hybrid BiLSTM 0.4130; matched text-only ablation 0.2854; DistilBERT 0.8633. TF-IDF remains best by test F1 and ROC-AUC. DistilBERT's validation-selected threshold gives higher test precision and F1 than its fixed 0.5 setting, while fixed 0.5 yields higher fraud recall. These are results on one historical dataset split, not deployment validation.

## Decisions log

- 2026-09-28 — Use the user-supplied CSV only; do not fetch other datasets. Reason: dataset restriction in the project brief.
- 2026-10-19 — Reject prediction requests with no job-posting text in both the API and prototype UI. Reason: avoid returning an intercept-only score that could be mistaken for an evidence-based assessment.
- 2026-10-19 — Let the guarded stop script recognize the project virtual-environment Python and its installed `C:\Codex\Python312\python.exe` runtime. Reason: safely restart a stale project API regardless of which approved Python executable launched it.
- 2026-09-28 — Use a stratified train/validation/test split and report class-sensitive metrics. Reason: only 866/17,880 rows are positive, so accuracy alone would be misleading.
- 2026-09-28 — Dataset missing values become empty strings in text; no rows are dropped for optional fields. Reason: fields are sparsely populated and this preserves available evidence.
- 2026-09-28 — Share one seeded stratified 72/8/20 train/validation/test partition across models; use validation F1 for deep-model checkpoint selection and report test metrics at threshold 0.5. Reason: make comparisons reproducible while keeping test data out of model selection.
- 2026-09-28 — Initialize DistilBERT from the named pretrained checkpoint and fine-tune it only with the supplied CSV. Reason: demonstrate transformer transfer learning while honoring the no-additional-training-dataset constraint.
- 2026-09-28 — Include rule-based warning patterns separately from learned model evidence. Reason: distinguish transparent text indicators from learned probability and prevent either from being presented as proof.
- 2026-09-29 — Install Python 3.12.10 and project dependencies under `C:\Codex`. Reason: make the project runnable while keeping the runtime and package environment outside the Windows user folders.
- 2026-09-29 — Pass a NumPy array to `ConfusionMatrixDisplay`. Reason: scikit-learn 1.9 rejects a nested Python list in its plotting API; this fixed evaluation after baseline fitting.
- 2026-09-29 — Use the baseline as the classroom prototype model and park further DL training. Reason: the CPU-only BiLSTM run was lengthy and the user prioritized a showable prototype for tomorrow.
- 2026-10-18 — Add a hybrid BiLSTM metadata branch and validation-selected threshold after reviewing recruitment-fraud papers. Reason: research points to imbalance, old benchmark coverage, and complementary text/metadata signals; compare without adding data.
- 2026-10-18 — Save threshold-selected and fixed-0.5 reports. Reason: make the precision/recall tradeoff visible and keep threshold selection away from the test set.
- 2026-10-18 — Preserve the previous BiLSTM artifact and save a matched-config text-only ablation separately. Reason: isolate metadata fusion while preserving existing work.
- 2026-10-18 — Use one full-data DistilBERT epoch, max length 64, batch 32, and two CPU threads after the original longer CPU run proved impractical. Reason: complete actual transformer fine-tuning and full held-out evaluation within available compute, without changing the dataset or test split.
- 2026-10-18 — Retain both fixed-0.5 and validation-selected DistilBERT test reports. Reason: show the high-precision/lower-recall versus higher-recall tradeoff; threshold was selected on validation data only.
- 2026-10-19 — Group the demo inputs into four free-text areas and map them into existing trained text fields; keep the browser payload compatible with the legacy API schema. Reason: make the demo easier to use while preserving the existing NLP model contract and support for the already-running server.
- 2026-10-19 — Add document extraction as a separate pre-inference step using PyMuPDF and RapidOCR, with an editable preview and no classifier retraining. Reason: accept PDF and screenshot text while retaining the existing dataset-trained prediction pipeline; OCR is a text extraction dependency, not another fraud dataset.
- 2026-10-19 — Bound uploads to 15 MB and 12 pages and spool multipart temporary files under `C:\Codex\tmp`. Reason: constrain local demo resource use and follow the user's requested project/runtime location.

## Current TODO

- [x] Implement preprocessing and dataset discovery/validation.
- [x] Implement training for TF-IDF + Logistic Regression, Word2Vec + BiLSTM, and DistilBERT.
- [x] Implement evaluation reports, confusion-matrix plots, EDA, and metric comparison.
- [x] Implement explainable indicators and prediction API.
- [x] Implement React client and deployment documentation.
- [x] Install declared dependencies in an isolated environment under `C:\Codex` and run EDA plus baseline training/evaluation.
- [x] Train/evaluate the baseline and BiLSTM; preserve their measured metrics.
- [x] Build and launch the baseline-powered classroom prototype; verify its health and prediction flow.
- [x] Review related research and document current/full architecture, prototype architecture, and future prospects.
- [x] Train hybrid BiLSTM and same-config text-only ablation; record actual test metrics.
- [x] Add baseline, BiLSTM, hybrid BiLSTM and DistilBERT loading/prediction paths; use saved validation threshold for hybrid and DistilBERT.
- [x] Train DistilBERT for one epoch on the full training partition with the available CPU budget; evaluate the untouched full validation and test partitions.
- [x] Smoke-check baseline, text BiLSTM, hybrid BiLSTM, and DistilBERT through inference and FastAPI route handlers; refresh the model comparison table.
- [x] Verify the live HTTP demo page, health endpoint, and prediction endpoint.
- [x] Replace dataset-column form fields with four free-text sections and adapt them to the existing NLP inference inputs; retain legacy API field compatibility and keep the result view unchanged.
- [x] Add PDF, PNG, and JPEG upload UI, text extraction endpoint, editable OCR preview, and assessment of user-reviewed document text.
- [x] Add multipart, PDF, and OCR dependencies to `requirements.txt` and install them into the `C:\Codex` virtual environment using `C:\Codex\tmp` for pip's temporary work.
- [x] Reload the active FastAPI process and verify native-text PDF extraction and assessment end to end with the supplied sample.
- [x] Verify screenshot OCR and image-only PDF OCR through the live `/extract` endpoint using temporary synthetic samples; both returned readable English text.
- [x] Reject empty prediction submissions with HTTP 422 and a clear message.
- [x] Replace the single-screen demo with the responsive React website and connect FastAPI to serve the Vite build and assets.
- [x] Build the frontend, launch the combined site/API, and verify the page, compiled assets, API health, and grouped-input prediction through localhost.
- [ ] Check edited extracted text through a real browser interaction and measure OCR quality on representative real screenshots/scanned documents.

## Known issues

- The installed PyTorch wheel is CPU-only. The completed DistilBERT run is intentionally limited to one epoch and 64-token input length; run longer experiments on CUDA-capable hardware if additional model quality/robustness is required.
- Hybrid F1 is threshold-sensitive: test F1 0.386 at 0.5 and 0.442 at validation-selected 0.6107; recall changes from 0.647 to 0.364. TF-IDF remains strongest overall by F1 and ROC-AUC.
- DistilBERT is also threshold-sensitive: test F1 is 0.717 at 0.5 (recall 0.844) and 0.796 at the validation-selected 0.9841 (recall 0.711). Probability calibration is not established; users should interpret thresholded predictions as risk assessments, not definitive determinations.
- The demo runs locally at `http://127.0.0.1:8000/` while its API process is active. The launcher reports the URL if it cannot open a browser automatically.
- DistilBERT fine-tuning may require access to the configured pretrained checkpoint. No external dataset is used by the project; document the pretrained model separately from training data.
- EDA emits a pandas warning about `select_dtypes(include="object")` behavior under pandas 3; the measured summary was generated successfully.
- The API defaults to the baseline. Set `BRS_MODEL` to select another trained artifact. Live HTTP health, `/extract`, `/predict`, and empty-input rejection were verified during the 2026-10-19 audit; all four model artifacts also passed direct inference smoke checks.
- The four-section form is a friendlier text-only input path. Its API adapter places job/location into `title`, company/recruiter details into `company_profile`, and the two remaining sections into a headed `description`; metadata fields stay empty unless a legacy client supplies them. This does not retrain any model or add new training features.
- Upload extraction accepts PDF, PNG, and JPEG files up to 15 MB and PDFs up to 12 pages. Native PDF text uses PyMuPDF; image OCR uses RapidOCR/ONNX Runtime with bundled pretrained English OCR models. The upload is not persisted by application code; FastAPI multipart spool files are directed to `C:\Codex\tmp`. Review OCR output before assessment.
- Fraud-classifier training still uses only the supplied Employment Scam Aegean dataset. The OCR engine's pretrained text-recognition weights are used solely for extraction. The classifier has not been validated on offer letters, and OCR accuracy has not yet been measured on a document sample.
- `start-demo.ps1` checks the OpenAPI schema for `/extract`. `stop-demo.ps1` only stops a process after checking its approved Python executable and Uvicorn app command. It supports both the venv Python and `C:\Codex\Python312\python.exe`.

## 2026-10-19 upload route fix and verification

- Replaced the stale baseline API on port 8000 after verifying its PID, executable, Uvicorn route, and API identity. The old API lacked `/extract`.
- Started the updated upload-enabled API with `start-demo.ps1`; `/health` reports `ready` with the baseline model and OpenAPI includes `/extract`.
- Posted the supplied one-page sample PDF through `/extract`: native PDF extraction returned 1,418 characters. Sent the extracted text through `/predict`; the baseline returned 0.351015 potentially-fraudulent probability, 0.648985 genuine probability, moderate risk. This is an inference example, not an evaluation metric.
- The API is currently running at `http://127.0.0.1:8000/`. Refresh the prototype page before using upload.

## 2026-10-19 full website build

- Replaced the one-screen form with a responsive website: landing section, grouped posting analyzer, PDF/image extraction and editable preview, model assessment, model comparison, process explanation, dataset source, and limitations.
- FastAPI serves `frontend/dist/index.html` and compiled `/assets` when the Vite build exists; the old prototype remains a development fallback.
- Built with Vite 5.4.21, React 18.3.1, and pnpm. The package store is under `C:\Codex\cache\pnpm`; `frontend/dist` is ignored build output.
- Verified the page and compiled JS/CSS return HTTP 200, API health is ready, and grouped-input assessment reaches the baseline model. The combined website/API is running on localhost.

## 2026-10-19 comprehensive backend audit

- `pip check`, imports for FastAPI/PyMuPDF/Pillow/RapidOCR/ONNX Runtime/scikit-learn/PyTorch/Transformers, and Python source compilation passed.
- Local HTTP checks passed for `/health`, `/extract`, and `/predict`. The supplied native-text PDF extracted 1,418 characters and reached the baseline classifier.
- Live `/extract` checks on temporary synthetic screenshot and scanned-PDF uploads both returned readable text; temporary files were removed. These are functionality checks, not OCR accuracy measurements on real postings.
- Direct inference smoke checks passed for baseline, text BiLSTM, hybrid BiLSTM, and DistilBERT artifacts.
- The audit found that empty predictions were scored. Added a clear HTTP 422 response and matching browser-side validation; confirmed the API returns 422 for an empty request.
- Updated the guarded stop script to recognize both approved `C:\Codex` Python executables after identifying the stale-server restart failure. Restarted the API with the updated code; it is currently serving the upload-enabled prototype.

## How to continue

Read this file first and inspect the repository before continuing. Use `C:\Codex\venvs\bharat-rozgaar-sentinel\Scripts\python.exe` from this project directory. Read `RESEARCH_REVIEW.md` and `ARCHITECTURE.md`. The demo defaults to the baseline; model selection uses `BRS_MODEL=baseline|bilstm|hybrid_bilstm|distilbert`. All four artifacts are trained. Use `Launch Demo.bat` to start the local upload-enabled API and `stop-demo.ps1` to stop it. Remaining work is browser-level review of OCR edits and measurement on representative real documents; the classifier has not been validated on offer letters. Keep temporary work under `C:\Codex\tmp`. Preserve AI-based risk-assessment / “potentially fraudulent” language; never imply definitive proof.

## 2026-09-28 follow-up (historical)

- Copied the project to `C:\Codex\Projects\Bharat Rozgar Sentinel` at the user's requested destination.
- At that point dependency installation was not yet available; setup was completed on 2026-09-29 as recorded below.

## 2026-09-29 environment and run log

- Installed and signature-verified official Python 3.12.10 x64 at `C:\Codex\Python312`.
- Installed `requirements.txt` into `C:\Codex\venvs\bharat-rozgaar-sentinel`; `pip check` and core imports passed. PyTorch is CPU-only.
- Ran EDA on the supplied CSV: 17,880 rows, 18 columns, class counts 17,014 genuine and 866 potentially fraudulent, and 0 exact duplicate rows. EDA outputs are under `outputs/eda/` and `outputs/plots/`.
- Trained/evaluated the baseline on the seeded stratified 20% test split (3,576 rows): accuracy 0.9899328859, precision 0.8914285714, recall 0.9017341040, F1 0.8965517241, ROC-AUC 0.9933686530. Artifact: `models/tfidf_lr/model.joblib`; report and plot under `outputs/metrics/` and `outputs/plots/`.
- Trained/evaluated BiLSTM for 3 epochs on CPU using its default config: accuracy 0.9317673378, precision 0.3489361702, recall 0.4739884393, F1 0.4019607843, ROC-AUC 0.8558276529. Artifacts: `models/word2vec_bilstm/model.pt` and `word2vec.model`; report and plot under `outputs/metrics/` and `outputs/plots/`.
- Fixed evaluation plotting to pass a NumPy array to scikit-learn's `ConfusionMatrixDisplay`.
- Added a self-contained prototype page at `/`, served by FastAPI, plus `start-demo.ps1`, `stop-demo.ps1`, and `Launch Demo.bat`.
- Verified the demo page returns HTTP 200, `/health` reports `ready` with the baseline model, and the example returns fraud/genuine probabilities, high risk, three indicators, and model token evidence. The example prediction is a UI smoke check, not an evaluation result.
- Further DL training was parked while the classroom prototype was prepared; this was superseded by the 2026-10-18 request to review research and resume DL experimentation.

## 2026-10-18 research and hybrid model run

- Reviewed papers from 2023–2026 on ensemble learning, fraud-type classification, transformer/imbalance methods, and LLM-assisted risk scoring. Paper methods, reported limitations, citations, and the proposed project contribution are in `RESEARCH_REVIEW.md`.
- Added `ARCHITECTURE.md` with plain-text full-system and prototype flows plus future prospects.
- Added hybrid metadata-fusion BiLSTM: text embedding/BiLSTM branch fused with a small MLP over five categorical fields (`employment_type`, `required_experience`, `required_education`, `industry`, `function`) and three binary fields (`telecommuting`, `has_company_logo`, `has_questions`). Metadata vocabulary/schema is fitted on training rows only. No external dataset or synthetic training samples were used.
- Hybrid configuration: seed 42, stratified 72/8/20 split, CPU, 3 epochs, batch 32, max sequence length 160, Word2Vec/embedding 128, hidden 128, dropout .35, weighted BCE. Best epoch uses validation F1 at threshold 0.5; threshold is then chosen on validation probabilities by F1: 0.610729 (validation F1 0.504065). Test metrics at selected threshold: accuracy 0.9555369, precision 0.5625, recall 0.3641618, F1 0.4421053, ROC-AUC 0.9081463, average precision 0.4130132. At fixed 0.5: accuracy 0.9004474, precision 0.2751843, recall 0.6473988, F1 0.3862069, same ROC-AUC and average precision. Test has 173 positive rows; selected threshold confusion matrix: 63 TP, 110 FN, 49 FP, 3,354 TN.
- Trained a separate matched text-only ablation on the same split/config. At threshold 0.5: accuracy 0.9180649, precision 0.2887324, recall 0.4739884, F1 0.3588621, ROC-AUC 0.8857248. Metadata fusion improved ROC-AUC and fraud recall in this one run, but precision remains low and TF-IDF baseline remains stronger overall.
- Hybrid artifacts: `models/hybrid_bilstm/model.pt`, `models/hybrid_bilstm/word2vec.model`; reports/plots under `outputs/metrics/` and `outputs/plots/`. Matched ablation artifact is under `models/word2vec_bilstm_ablation/`.
- At the time of this run, comparison reports included PR-AUC where available and DistilBERT had not yet been trained. This status was superseded by the later completion entry below. The prototype and API remained baseline-powered.

## 2026-10-18 model backend follow-up

- Added inference loading for `hybrid_bilstm` and saved decision thresholds for hybrid/DistilBERT in `src/inference/predict.py`; DistilBERT readiness now checks for a real config and weight file instead of treating an empty folder as trained. API selection uses `BRS_MODEL`.
- Added batched DistilBERT validation/test inference to bound evaluation memory and reports both fixed 0.5 and validation-selected threshold results after training.
- Downloaded `distilbert-base-uncased` from Hugging Face into `work/huggingface_cache` (the cache is outside training data). The first cache attempt at `C:\Codex\model_cache` failed due Windows ACLs; moving the cache into the project succeeded.
- Attempt 1: CPU, 3 epochs, batch 16, sequence length 128. Stopped after about 30 minutes without an epoch report; no artifact/metrics saved.
- Attempt 2: CPU, full training partition, 1 epoch, batch 32, sequence length 96, learning rate 2e-5. Stopped after about 45 minutes without an epoch report; no artifact/metrics saved. This is an incomplete run, not a model result.
- Existing TF-IDF, text-only BiLSTM, hybrid BiLSTM and matched ablation metrics remain the only measured results. PyTorch reports CPU-only and no `nvidia-smi` command is available. DistilBERT needs GPU-capable compute or a deliberately smaller study.

## 2026-10-18 DistilBERT completion and backend verification

- The earlier 1,600-row stratified DistilBERT pilot ran for one epoch with batch 16, max length 64, and two CPU threads. It produced a test F1 of 0.4820 at its validation-selected threshold and was kept only as a pilot; the artifact was subsequently replaced by the full-training-partition run.
- Completed DistilBERT fine-tuning for one epoch on all 12,873 training rows. Configuration: `distilbert-base-uncased`, seed 42, CPU, batch 32, max length 64, learning rate 2e-5, weighted cross-entropy, two CPU threads. Full validation rows: 1,431; full test rows: 3,576. Final train mean loss: 0.4446; validation F1 at 0.5: 0.7186; validation-selected threshold: 0.984093; validation F1 at selected threshold: 0.8254.
- Test results at validation-selected threshold: accuracy 0.9823826, precision 0.9044118, recall 0.7109827, F1 0.7961165, ROC-AUC 0.9806172, PR-AUC 0.8632896. Confusion matrix (genuine/fraud rows, predicted genuine/fraud columns): `[[3390, 13], [50, 123]]`.
- Test results at fixed threshold 0.5: accuracy 0.9678412, precision 0.6239316, recall 0.8439306, F1 0.7174447, ROC-AUC 0.9806172, PR-AUC 0.8632896. Confusion matrix: `[[3315, 88], [27, 146]]`.
- Saved final artifact under `models/distilbert/`; reports `outputs/metrics/distilbert.json` and `distilbert_0_5.json`; confusion matrices under `outputs/plots/`. Refreshed `outputs/metrics/model_comparison.csv` with actual reports.
- Added stratified `--train-limit` for optional pilot runs and `--cpu-threads` configuration to `src/train.py`, with training progress output and metadata recording the actual/full training row counts. README documents the reproducible full-data CPU command.
- Smoke-checked inference and FastAPI health/predict route handlers for baseline, text BiLSTM, hybrid BiLSTM, and DistilBERT. `pip check`, Python compilation, and trainer CLI help passed. The sandbox denied local socket binding and in-process HTTP client requests stalled, so a live HTTP server round-trip remains unverified here; direct handler calls succeeded.
- Re-evaluated the saved baseline and text BiLSTM artifacts on the same held-out test rows to add PR-AUC to their reports (0.9542049 and 0.4240506 respectively); their previously recorded accuracy, precision, recall, F1, and ROC-AUC were unchanged. Rebuilt the comparison CSV with PR-AUC for all available reports.

## 2026-10-19 demo launch and HTTP verification

- Started the baseline-backed FastAPI app at `http://127.0.0.1:8000/` using the installed `C:\Codex` virtual environment.
- Verified the prototype page returns HTTP 200, `/health` returns `ready` with model `baseline`, and `/predict` returns an assessment with fraud/genuine probabilities, risk level, model name, and risk indicators. The illustrative request returned `genuine`, fraud probability 0.439403, genuine probability 0.560597, and moderate risk; this is an inference smoke check, not an evaluation metric.
- The launcher initially stopped after browser auto-open was denied by the desktop sandbox. Updated `start-demo.ps1` so browser launch failure is reported while the API remains running. Retested the launcher against the active server; it now exits successfully and keeps the server healthy.
- Detached processes are reaped after a shell command exits in this execution environment, so the active server was launched in a persistent foreground execution session. The app remains available at the URL while that session is active; from a normal desktop PowerShell session, `Launch Demo.bat` can be used to start it.

## 2026-10-19 grouped text input update

- Replaced the prototype's dataset-shaped form with four text areas: job and location; company and recruiter; work and qualifications; pay, benefits, and next steps. Each accepts free-form pasted text, and the page explains that users can leave unavailable details out.
- Added four grouped text properties to the FastAPI request schema and `to_model_posting` adapter. Role/location maps to `title`, company/recruiter to `company_profile`, and the two longer sections are added to `description` with headings. Existing dataset-shaped API fields remain supported. The browser serializes its four grouped values into the legacy title/company-profile/description request contract so an already-loaded older server can still process the form.
- Kept the assessment/results panel and model behavior unchanged. The existing TF-IDF model still produces probability and token evidence from its learned text features; this input change does not establish new performance or train a new model.
- Refreshed the demo example to populate the four grouped fields. The launcher reported an already-running baseline server; its OpenAPI schema still showed only the original request fields, and its recorded PID was already stopped. The form's prediction request was not sent in this edit session.
- Next: open `http://127.0.0.1:8000/`, use the demo example, and check the four-field input and assessment once before presenting. The current browser request uses the legacy field contract for compatibility. Restart the server from a normal desktop session if direct requests using the new grouped API properties are needed.

## 2026-10-19 PDF and image upload update

- Added `src/inference/documents.py` for in-memory PDF, PNG, and JPEG validation and extraction. Text-based PDF pages use PyMuPDF text extraction; pages with little embedded text and uploaded images are rendered/read and processed by RapidOCR. Limits are 15 MB, 12 document pages, 25 megapixels per image, and 60,000 extracted characters. Password-protected PDFs and unreadable uploads return clear errors.
- Added FastAPI `POST /extract` using `UploadFile`; CPU extraction runs in Starlette's thread pool. The endpoint returns extracted text, source filename, page count, extraction method, and a review warning when OCR ran. It never writes a persistent upload copy. FastAPI's multipart temporary spool directory is set to `C:\Codex\tmp`.
- Added upload controls for PDF/PNG/JPG, an editable text preview, extraction method/page count display, and an **Assess extracted text** action that sends the reviewed document text through the existing classifier. The four-section manual entry path and result panel remain available.
- Added `python-multipart`, `pymupdf`, `Pillow`, `rapidocr`, and `onnxruntime` to `requirements.txt`. Installed those packages successfully into `C:\Codex\venvs\bharat-rozgaar-sentinel`; pip temporary files for that successful install were directed to `C:\Codex\tmp`. The OCR models are bundled with the RapidOCR package; no additional fraud-training dataset or model retraining was used.
- A first direct Tesseract installer download was blocked with HTTP 403. Switched to RapidOCR, which is packaged for CPU inference, and did not install any OCR executable outside `C:\Codex`.
- Updated the launcher to inspect OpenAPI for `/extract`, restart only a verified project process if the API is stale, and set `BRS_TMP_DIR=C:\Codex\tmp`. Updated the stop script to search the port listener only when its executable and Uvicorn route match the project server.
- Attempted to launch the updated demo. It found the old API schema, then Windows CIM process inspection was denied by the execution environment; the stop script left the process untouched and launcher exited without starting a replacement. No tests or live upload requests were run. Verify native PDF text, scanned PDF OCR, PNG/JPEG OCR, preview edits, and final prediction after restarting from a normal desktop PowerShell session.

## 2026-10-19 full website completion

- Replaced the classroom prototype UI with a complete responsive React/Vite website served by FastAPI at `/`. The experience includes project overview, grouped posting analysis, document upload and editable OCR preview, prediction results, model comparison, workflow, dataset source, and limitations.
- Built frontend assets with Vite (`pnpm build`; Vite 5.4.21). Build completed successfully. `start-demo.ps1` now requires the built React site and reports the website status.
- FastAPI serves `frontend/dist/index.html` and `/assets/*` when the build exists; the prototype page remains a fallback. No third-party runtime font/CDN dependency was added.
- Verified on 2026-10-19: `http://127.0.0.1:8000/` returned HTTP 200 with the website title; `/health` returned `ready`, model `baseline`; OpenAPI includes `/extract`; both generated JavaScript and CSS assets returned HTTP 200. A grouped-input `/predict` smoke request returned fraud probability 0.353909, genuine probability 0.646091, moderate risk, baseline model, and token evidence. This is an inference smoke check and not a model evaluation metric.
- The local API binds to `127.0.0.1` on port 8000. It was live at the end of this session. Open or refresh `http://127.0.0.1:8000/` to view the website. `Launch Demo.bat` / `start-demo.ps1` can start or reuse it; `stop-demo.ps1` stops it.
- Model artifacts and reported evaluation metrics were not changed during the website work. The live website currently defaults to the baseline classifier; document extraction/OCR feeds extracted text into the selected classifier and does not itself alter model accuracy.
- Next: continue backend/model work from this implementation and use real evaluation reports for any model-performance claims. Before deployment beyond local demo use, review production security, privacy, and validation needs.

## 2026-10-19 blank website screen fix

- User screenshot showed a blank page despite HTTP 200 responses. Reproduced it in headless Chrome and captured the browser runtime exception: `ReferenceError: React is not defined` in the built bundle. JSX was compiled as `React.createElement`, while `App.jsx` imported only hooks.
- Fixed `frontend/src/App.jsx` to import the React namespace, then rebuilt the Vite site successfully. The bundle hash changed to `index-vfPN-Nru.js`.
- Verified with headless Chrome after rebuild: the React root now contains the full page (hero, analyzer, model table and other sections), the API badge reaches “System online”, and the browser console contains no React startup/runtime error. The API remains available at `http://127.0.0.1:8000/` and the assessment route responds.
- The existing localhost server serves files from `frontend/dist`, so it did not need a backend restart. If the browser retains the older blank bundle, open `http://127.0.0.1:8000/?v=react-fix` to fetch the current HTML and hashed JavaScript bundle.

## 2026-10-19 GitHub repository preparation

- Prepared the project for source control: `.gitignore` excludes the training CSV, model weights, pretrained cache, temporary files, frontend dependencies/build output and generated predictions, while including actual small evaluation reports, EDA summaries and plots. README explains how to download the dataset and regenerate model files.
- Scanned source candidates for common API/private-key credential patterns; none were found. The dataset, models and work cache were excluded from Git.
- Initialized local Git on `main` and created initial commit `6dc42a6` (`Initial Bharat Rozgaar Sentinel project`), containing 63 reviewed files (about 330 KB packed loose Git objects). The local working tree is clean; no remote is configured.
- Authenticated GitHub connector account is `hitarthshah3`. The private target `hitarthshah3/Bharat-Rozgaar-Sentinel` was populated through the connected GitHub Git Data API because local Git Credential Manager access is denied in this environment. The project tree was established by commit `4892706dbcfd192022bdb914e038810fef7a9cbb` on `main`, followed by context-recording commits. The verified tree contains 63 source, documentation, evaluation-report, and plot files. Verified `README.md`, the React import fix, and absence of the dataset and model weight files.
- The GitHub-side commit tree contains the prepared project files; its commit hash differs from the locally prepared commits because upload used the GitHub API instead of `git push`. Do not assume local `main` is synchronized with the GitHub remote. For future command-line pushes, first configure GitHub authentication and synchronize from the remote branch. Keep the dataset and large generated weights/caches excluded.
- Next: continue from the verified GitHub `main` tree and keep local and remote commit histories synchronized before making future pushes.



## 2026-10-19 Stitch UI handoff

- Added `STITCH_UI_HANDOFF.md` to the project and GitHub `main`. It documents the existing React/FastAPI flows, API contract, redesign states, accessibility/visual direction, prediction language constraints, a tailored Stitch prompt, and import steps.
- Google's May 19, 2026 announcement says Stitch supports importing existing codebases and design files. At the time this handoff was first written, direct Stitch access was unavailable; a private Stitch project was later created and populated as recorded below.
- Next: review the imported screen in the private Stitch project, then bring selected redesigned screens/code back for integration. Preserve API behavior and do not treat generated metrics or unsupported states as real capabilities.


## 2026-10-19 Stitch upload and design system

- Created the private Google Stitch project **Bharat Rozgaar Sentinel** (project ID `7576516158361421257`).
- Uploaded `work/STITCH_DESIGN.md` as the project design-system source (screen instance `4071015268694958488`) and created the Stitch design system (asset ID `54440815ea984f9cb4ea9407ce9e1abc`).
- Uploaded `work/stitch-home.html`, a full-page `/home` capture of the local React website, as a desktop design (screen instance `2621135639633319397`, 1280×3102).
- The HTML capture and design tokens came from the existing site; no application code or classifier/model artifacts were changed by this handoff.
- Next: review the imported screen in Stitch, request the desired redesign there, then export the selected design/code and integrate it while preserving the FastAPI prediction and document-extraction contracts.
