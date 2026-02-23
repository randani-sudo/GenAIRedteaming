# LLM Red Team ML Service

Enterprise-grade binary classifier for detecting successful adversarial attacks on LLMs. Labels each prompt-response pair as **vulnerable** (attack succeeded) or **safe** (defended).

## Features

- **Data**: RedBench (29,362+ samples) and NaviRocker (10,000+) style CSVs; flexible column mapping; synthetic data when no CSV is provided.
- **Features**: Tabular (toxicity proxy, refusal phrases, response/prompt length) + BERT semantic embeddings.
- **Models**: XGBoost (tabular), BERT (response text), or **Ensemble** (XGBoost + BERT, target F1 >85%, recall >90%).
- **Training**: Stratified 70/15/15 split, optional SMOTE, configurable epochs and hyperparameters.
- **API**: FastAPI — `POST /api/train`, `POST /api/predict`, `GET /health`.

## Setup

```bash
cd ml-service
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
cp .env.example .env
# Optionally set DATA_DIR, REDBENCH_CSV, NAVIROCKER_CSV, PERSPECTIVE_API_KEY
```

## Data format

CSV columns (names are normalized): `prompt`, `response`, `label` (vulnerable/safe), optional `category`, `language`. Place files under `data/redbench/` and `data/navirocker/` or set paths in `.env`.

## Training

**CLI (standalone):**

```bash
python train.py --architecture ensemble [--data-csv path/to/data.csv] [--run-id tr-123] [--output-dir ./models]
```

**Via API (from Next.js):**

```bash
uvicorn main:app --reload --port 8000
```

Then from the dashboard, start a training run; the Next.js API will call `POST http://localhost:8000/api/train` with `runId`, `datasetPath`, `architecture`, `hyperparams`. When training finishes, the ML service calls `POST http://localhost:3000/api/ml-callback` to update the run status and metrics.

## Inference

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"prompts":[{"prompt":"...","response":"..."}],"modelId":"ensemble"}'
```

Returns `predictions: [{ label: "vulnerable"|"safe", confidence, toxicityScore }]`.

## Environment

| Variable | Description |
|----------|-------------|
| `DATA_DIR` | Base directory for CSV data |
| `MODEL_DIR` | Where to save/load models |
| `REDBENCH_CSV`, `NAVIROCKER_CSV` | Paths to datasets |
| `BERT_MODEL` | HuggingFace model (default `bert-base-uncased`) |
| `PERSPECTIVE_API_KEY` | Optional; for real toxicity scores (otherwise rule-based proxy) |
| `NEXT_PUBLIC_APP_URL` | Next.js app URL for training callback (default `http://localhost:3000`) |

## Goals

- F1 >85%, recall >90% on binary classification across jailbreaking, bias, toxicity, privacy, misinformation.
- Support for Hindi/Tamil (500+ samples) and extensible to more languages.
- Reusable pipeline for Indian enterprises (Digital India Act compliance, pre-deployment certification, weekly scans).
