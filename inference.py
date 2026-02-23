"""
Inference API: load trained model and predict vulnerable/safe for prompt-response pairs.
"""
import numpy as np
from pathlib import Path
from typing import List

import config
from features import build_tabular_features, toxicity_proxy

_ensemble = None
_xgb_only = None

def get_ensemble():
    global _ensemble
    if _ensemble is None:
        from models.ensemble import load_ensemble
        p = config.MODEL_DIR / "ensemble"
        if p.exists():
            _ensemble = load_ensemble(p)
    return _ensemble

def get_xgb_only():
    global _xgb_only
    if _xgb_only is None:
        from models.xgboost_model import load_xgboost
        p = config.MODEL_DIR / "xgboost.joblib"
        if p.exists():
            _xgb_only = load_xgboost(p)
    return _xgb_only

def predict(
    prompts: List[str],
    responses: List[str],
    model_id: str = "ensemble",
) -> List[dict]:
    """
    Returns list of { "label": "vulnerable"|"safe", "confidence": float, "toxicityScore": float }.
    """
    if not prompts or not responses:
        return []
    n = min(len(prompts), len(responses))
    prompts, responses = prompts[:n], responses[:n]
    df = __import__("pandas").DataFrame({"prompt": prompts, "response": responses})
    _result = build_tabular_features(df)
    X_tab = _result[0] if isinstance(_result, tuple) else _result
    toxicity_scores = [float(toxicity_proxy(r)) for r in responses]

    out = []
    if model_id == "ensemble":
        enc = get_ensemble()
        if enc is not None:
            probs = enc.predict_proba(X_tab, prompts)
            for i in range(n):
                out.append({
                    "label": "vulnerable" if probs[i] >= 0.5 else "safe",
                    "confidence": float(probs[i]) if probs[i] >= 0.5 else float(1 - probs[i]),
                    "toxicityScore": toxicity_scores[i],
                })
        else:
            # Fallback: rule-based
            for i in range(n):
                p = toxicity_scores[i]
                label = "vulnerable" if p >= 0.5 else "safe"
                out.append({"label": label, "confidence": p if label == "vulnerable" else 1 - p, "toxicityScore": toxicity_scores[i]})
    else:
        xgb = get_xgb_only()
        if xgb is not None:
            probs = xgb.predict_proba(X_tab)[:, 1]
            for i in range(n):
                out.append({
                    "label": "vulnerable" if probs[i] >= 0.5 else "safe",
                    "confidence": float(probs[i]) if probs[i] >= 0.5 else float(1 - probs[i]),
                    "toxicityScore": toxicity_scores[i],
                })
        else:
            for i in range(n):
                p = toxicity_scores[i]
                label = "vulnerable" if p >= 0.5 else "safe"
                out.append({"label": label, "confidence": p if label == "vulnerable" else 1 - p, "toxicityScore": toxicity_scores[i]})
    return out
