"""
Ensemble: XGBoost (tabular) + BERT (semantic) via stacking meta-learner.
Binary classification: vulnerable (1) vs safe (0).
"""
import numpy as np
from pathlib import Path
from typing import Optional
import joblib
import config
from . import xgboost_model as xgb_module
from . import bert_classifier as bert_module
from features import build_tabular_features
from sklearn.linear_model import LogisticRegression


class EnsembleClassifier:
    def __init__(self, xgb_weight: float = 0.4, bert_weight: float = 0.6, threshold: float = 0.35):
        self.xgb_weight     = xgb_weight
        self.bert_weight    = bert_weight
        self.threshold      = threshold
        self.xgb_model      = None
        self.bert_model     = None
        self.bert_tokenizer = None
        self.meta_learner   = None
        self.tfidf          = None   # ← NEW: fitted vectorizer from features.py
        self._device        = None

    def fit(
        self,
        X_tab: np.ndarray,
        train_texts: list,
        y: np.ndarray,
        X_val_tab: np.ndarray,
        val_texts: list,
        y_val: np.ndarray,
        bert_epochs: int = 5,
        xgb_params: Optional[dict] = None,
    ):
        xgb_params = xgb_params or {}

        print("Training XGBoost...")
        self.xgb_model = xgb_module.train_xgboost(
            X_tab, y, X_val_tab, y_val, use_smote=True, **xgb_params
        )

        print("Training BERT...")
        self.bert_model, self.bert_tokenizer = bert_module.train_bert(
            train_texts, y, val_texts, y_val, epochs=bert_epochs
        )

        print("Training stacking meta-learner...")
        xgb_val_proba  = self.xgb_model.predict_proba(X_val_tab)[:, 1]
        bert_val_proba = bert_module.predict_bert(
            self.bert_model, self.bert_tokenizer, val_texts
        )
        meta_X = np.column_stack([xgb_val_proba, bert_val_proba])
        self.meta_learner = LogisticRegression(C=1.0, max_iter=1000)
        self.meta_learner.fit(meta_X, y_val)
        print(
            "Meta-learner weights → XGB: {:.3f} | BERT: {:.3f}".format(
                self.meta_learner.coef_[0][0],
                self.meta_learner.coef_[0][1],
            )
        )
        return self

    def predict_proba(self, X_tab: np.ndarray, texts: list) -> np.ndarray:
        xgb_proba  = self.xgb_model.predict_proba(X_tab)[:, 1]
        bert_proba = bert_module.predict_bert(
            self.bert_model, self.bert_tokenizer, texts
        )
        if self.meta_learner is not None:
            meta_X = np.column_stack([xgb_proba, bert_proba])
            return self.meta_learner.predict_proba(meta_X)[:, 1]
        else:
            return self.xgb_weight * xgb_proba + self.bert_weight * bert_proba

    def predict(self, X_tab: np.ndarray, texts: list) -> np.ndarray:
        return (self.predict_proba(X_tab, texts) >= self.threshold).astype(int)

    def predict_with_confidence(self, X_tab: np.ndarray, texts: list) -> dict:
        proba = self.predict_proba(X_tab, texts)
        return {
            "label":      (proba >= self.threshold).astype(int),
            "confidence": proba,
            "risk_level": np.where(proba > 0.8, "HIGH",
                          np.where(proba > 0.5, "MEDIUM", "LOW")),
        }


def save_ensemble(ensemble: EnsembleClassifier, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    bert_module.save_bert(ensemble.bert_model, ensemble.bert_tokenizer, path / "bert")
    joblib.dump(
        {
            "xgb_model":    ensemble.xgb_model,
            "xgb_weight":   ensemble.xgb_weight,
            "bert_weight":  ensemble.bert_weight,
            "meta_learner": ensemble.meta_learner,
            "threshold":    ensemble.threshold,
            "bert_path":    str(path / "bert"),
            "tfidf":        ensemble.tfidf,   # ← NEW: persist vectorizer
        },
        path / "ensemble_meta.joblib",
    )


def load_ensemble(path: Path) -> EnsembleClassifier:
    path = Path(path)
    meta = joblib.load(path / "ensemble_meta.joblib")
    enc = EnsembleClassifier(
        xgb_weight=meta["xgb_weight"],
        bert_weight=meta["bert_weight"],
        threshold=meta.get("threshold", 0.35),
    )
    enc.xgb_model      = meta["xgb_model"]
    enc.meta_learner   = meta.get("meta_learner")
    enc.tfidf          = meta.get("tfidf")        # ← NEW: restore vectorizer
    enc.bert_model, enc.bert_tokenizer = bert_module.load_bert(path / "bert")
    return enc
