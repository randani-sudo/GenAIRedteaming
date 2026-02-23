"""
FastAPI ML Service: training trigger and prediction API for the LLM Red Team classifier.
Run: uvicorn main:app --reload --port 8000
"""
import asyncio
import os
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config

# Optional: callback to Next.js to update training run status
def _callback_training_update(run_id: str, status: str, metrics: dict = None):
    url = os.getenv("NEXT_PUBLIC_APP_URL", "http://localhost:3000")
    try:
        import urllib.request
        import json
        req = urllib.request.Request(
            f"{url}/api/ml-callback",
            data=json.dumps({"runId": run_id, "status": status, "metrics": metrics or {}}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    yield
    # shutdown

app = FastAPI(title="LLM Red Team ML Service", version="1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Schemas ---
class TrainRequest(BaseModel):
    runId: str
    datasetPath: str
    architecture: str = "ensemble"
    hyperparams: dict = {}

class TrainResponse(BaseModel):
    runId: str
    status: str
    message: str | None = None

class PredictItem(BaseModel):
    prompt: str
    response: str
    category: str | None = None

class PredictRequest(BaseModel):
    prompts: list[PredictItem]
    modelId: str = "ensemble"

class PredictItemResponse(BaseModel):
    label: str
    confidence: float
    toxicityScore: float | None = None

class PredictResponse(BaseModel):
    predictions: list[PredictItemResponse]
    modelId: str

# --- Routes ---
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/api/train", response_model=TrainResponse)
async def api_train(req: TrainRequest):
    """Start training in background; returns immediately."""
    def run_sync():
        try:
            from train import main as train_main
            import sys
            import json
            sys.argv = [
                "train.py",
                "--architecture", req.architecture,
                "--run-id", req.runId,
            ]
            if req.datasetPath:
                sys.argv.extend(["--data-csv", req.datasetPath])
            train_main()
            metrics = {}
            mp = config.MODEL_DIR / f"metrics_{req.runId}.json"
            if mp.exists():
                with open(mp) as f:
                    data = json.load(f)
                    metrics = {k: v for k, v in data.items() if k in ("f1", "precision", "recall", "accuracy", "auc") and isinstance(v, (int, float))}
            _callback_training_update(req.runId, "trained", metrics)
        except Exception as e:
            _callback_training_update(req.runId, "failed", {"error": str(e)})
    loop = asyncio.get_event_loop()
    asyncio.create_task(loop.run_in_executor(None, run_sync))
    return TrainResponse(runId=req.runId, status="started", message="Training started in background.")

@app.post("/api/download-redbench")
def api_download_redbench():
    """Download RedBench from Hugging Face (knoveleng/redbench) and export to CSV."""
    def run():
        out_path = config.DATA_DIR / "redbench" / "redbench.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        from datasets import load_dataset
        import pandas as pd
        try:
            ds = load_dataset("knoveleng/redbench", trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(f"Failed to load RedBench from HF: {e}") from e
        parts = []
        for split in list(ds.keys()):
            try:
                parts.append(ds[split].to_pandas())
            except Exception:
                continue
        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if df.empty:
            df = pd.DataFrame(columns=["prompt", "response", "label"])
        for c in list(df.columns):
            c_l = str(c).lower()
            if "prompt" in c_l or "question" in c_l or "input" in c_l or c_l == "prompt":
                df = df.rename(columns={c: "prompt"})
                break
        for c in list(df.columns):
            c_l = str(c).lower()
            if "response" in c_l or "answer" in c_l or "output" in c_l:
                df = df.rename(columns={c: "response"})
                break
        for c in list(df.columns):
            c_l = str(c).lower()
            if "label" in c_l or "target" in c_l:
                df = df.rename(columns={c: "label"})
                break
        if "prompt" not in df.columns and len(df.columns):
            df["prompt"] = df.iloc[:, 0].astype(str)
        if "response" not in df.columns:
            df["response"] = ""
        if "label" not in df.columns:
            df["label"] = "safe"
        keep = [c for c in ["prompt", "response", "label"] if c in df.columns]
        df = df[keep].head(50000) if keep else df.head(50000)
        df.to_csv(out_path, index=False)

    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as ex:
        f = ex.submit(run)
        f.result(timeout=300)
    return {"ok": True, "path": str(config.DATA_DIR / "redbench" / "redbench.csv")}

@app.post("/api/predict", response_model=PredictResponse)
def api_predict(req: PredictRequest):
    """Predict vulnerable/safe for each prompt-response pair."""
    from inference import predict
    prompts = [p.prompt for p in req.prompts]
    responses = [p.response for p in req.prompts]
    results = predict(prompts, responses, model_id=req.modelId)
    return PredictResponse(
        predictions=[PredictItemResponse(**r) for r in results],
        modelId=req.modelId,
    )
