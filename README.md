# Bharat Rozgaar Sentinel

AI-based risk assessment for fraudulent employment postings. Predictions are probabilistic signals, not proof that a posting is fraudulent.

## Dataset

Download the official Employment Scam Aegean CSV from [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) and place it in `data/`. The application discovers the sole CSV there and stops with a clear message if it is missing or ambiguous. The dataset is not committed to this repository; no other training dataset is used.

## Setup and training

Use Python 3.10+ in a virtual environment, then install `requirements.txt`.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m src.train --model baseline
python -m src.train --model bilstm
python -m src.train --model hybrid_bilstm
python -m src.train --model distilbert --epochs 1 --batch-size 32 --max-length 64 --cpu-threads 2
```

Each trainer uses the same reproducible, stratified partitions and writes actual metrics to `outputs/metrics/`, plots to `outputs/plots/`, and the deployable artifact to its `models/` directory. The held-out test partition is evaluated after training. Deep-learning training can be configured with CLI options; run `python -m src.train --help` for options.

The hybrid BiLSTM fuses the text sequence representation with train-fitted categorical and binary posting metadata. It selects its checkpoint and fraud threshold using validation data; its report also includes the fixed 0.5 threshold. The text-only BiLSTM accepts `--artifact-name` for a separate ablation run. See [RESEARCH_REVIEW.md](RESEARCH_REVIEW.md) for the paper review and [ARCHITECTURE.md](ARCHITECTURE.md) for the system and prototype flows.

Run `python -m src.eda` to create the dataset summary and missing-value plot. After evaluating multiple models, `python -m src.evaluation.compare` builds a comparison CSV from the reports present on disk.

DistilBERT uses a pretrained language-model checkpoint for initialization, then fine-tunes only on the supplied dataset. It does not download or train on another task dataset. The command above is the completed CPU configuration: one epoch over the full training partition, with the full validation and test partitions retained. Set `--checkpoint` to a locally available checkpoint for offline work. `--train-limit N` optionally runs a faster stratified pilot while still evaluating on the full validation and test partitions; do not report a pilot as a full-data run. `--cpu-threads` controls PyTorch's CPU thread count.

The DistilBERT trainer selects its decision threshold on validation probabilities and also writes a fixed-0.5 report so the precision/recall tradeoff can be compared.

## API

After training a model:

```powershell
$env:BRS_MODEL = "baseline"
uvicorn src.inference.api:app --reload
```

Set `BRS_MODEL` to `baseline`, `bilstm`, `hybrid_bilstm`, or `distilbert` to select an available artifact. For example, use `hybrid_bilstm` to load the metadata-fusion BiLSTM. Open `/docs` for the interactive API. Set `BRS_MODEL_PATH` to override the selected model artifact. Health reports whether that artifact is available.

For DistilBERT, the pretrained checkpoint is cached under `work/huggingface_cache`. The current computer uses CPU-only PyTorch. The recorded model was trained for one epoch on all 12,873 training rows with sequence length 64; longer or multi-epoch CPU runs will take substantially more time.

## Full local website

The website is a responsive React application built with Vite. Rebuild the frontend from `frontend/` with pnpm (`pnpm install`, then `pnpm build`). FastAPI serves the built site together with the assessment API. Double-click `Launch Demo.bat` to open `http://127.0.0.1:8000/`; Node.js is not needed after the build. Use `stop-demo.ps1` from PowerShell to stop it.

The Analyze section accepts four grouped text areas or one PDF, PNG, or JPEG up to 15 MB. Choose **Extract text**, review or correct the editable preview, then assess the posting. Text PDFs use direct extraction; scanned PDFs and images use RapidOCR. Uploads are not saved as files by the app, and multipart temporary files use `C:\Codex\tmp`. The OCR package includes pretrained text-recognition models; it does not train or change the job-fraud classifier or add a training dataset.

The API exposes `POST /extract` for file-to-text extraction and `POST /predict` for model assessment. Empty submissions are rejected. OCR currently targets English text; recognition may make mistakes, so the preview is editable. The classifier was trained on job-posting text, and performance on offer letters has not been established.

## Repository contents

Training data, model weight files, pretrained checkpoints, temporary files, and frontend build/dependency folders are intentionally excluded from Git because they are large local artifacts. Download the dataset as described above, then train the selected model using the commands in **Setup and training**. Actual evaluation reports and plots are included under `outputs/metrics/` and `outputs/plots/`.

## Web client

```powershell
cd frontend
npm install
npm run dev
```

The client expects the API at `http://localhost:8000`; configure `VITE_API_URL` to change it.

## Scope and limitations

The dataset is imbalanced and historical. Metrics must be interpreted with precision, recall, F1, ROC-AUC, and the confusion matrix. A risk assessment can be wrong; users should independently verify employers and never treat model output as a final determination.


