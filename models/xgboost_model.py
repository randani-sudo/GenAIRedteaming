"""XGBoost binary classifier for tabular features. Uses GPU when config.USE_GPU and CUDA available."""
import numpy as np
import joblib
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
import config


def _xgb_gpu_params() -> dict:
    """
    XGBoost 2.x removed gpu_hist. Always use device='cuda' + tree_method='hist'.
    Falls back to CPU if USE_GPU is off or CUDA unavailable.
    """
    if not getattr(config, "USE_GPU", False):
        return {"tree_method": "hist"}
    try:
        import torch
        if torch.cuda.is_available():
            return {"device": "cuda", "tree_method": "hist"}
    except ImportError:
        pass
    return {"tree_method": "hist"}  # CPU fallback


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    use_smote: bool = True,
):
    if len(np.unique(y_train)) < 2:
        raise ValueError(
            "XGBoost needs both classes (0 and 1). The training set has only one label. "
            "Add safe examples (expected: false) or use a dataset with both vulnerable and safe prompts."
        )

    if use_smote:
        from imblearn.over_sampling import SMOTE
        counts = np.bincount(y_train.astype(int))
        minority_count = int(counts.min()) if len(counts) > 1 else 0
        if len(counts) >= 2 and minority_count >= 2:
            k = max(1, min(5, minority_count - 1))
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"SMOTE applied: {X_train.shape[0]} samples after resampling.")

    # scale_pos_weight handles class imbalance in XGBoost natively
    scale_pos_weight = float(np.sum(y_train == 0)) / max(float(np.sum(y_train == 1)), 1)

    gpu_params = _xgb_gpu_params()
    print(f"XGBoost params: {gpu_params}")

    kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        # use_label_encoder removed — deprecated and causes warnings in XGBoost 1.6+
    )
    kwargs.update(gpu_params)

    model = xgb.XGBClassifier(**kwargs)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    return model


def evaluate_xgboost(model: xgb.XGBClassifier, X: np.ndarray, y: np.ndarray, threshold: float = 0.35) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred  = (proba >= threshold).astype(int)
    return {
        "f1":        float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall":    float(recall_score(y, pred, zero_division=0)),
        "accuracy":  float(accuracy_score(y, pred)),
        "auc":       float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else 0.0,
        "threshold": threshold,
    }


def save_xgboost(model: xgb.XGBClassifier, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_xgboost(path: Path) -> xgb.XGBClassifier:
    return joblib.load(path)
