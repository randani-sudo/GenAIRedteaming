"""
Training script for the LLM adversarial attack classifier.
Supports: xgboost, bert, ensemble. Uses stratified 70/15/15 split and optional SMOTE.
Usage:
  python train.py [--architecture ensemble] [--data-csv path] [--output-dir dir]
  python train.py --architecture ensemble --threshold 0.35 --metrics-out metrics.json
  python train.py --architecture ensemble --epochs 1 --no-gpu   # CI smoke test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import config
from data_loader import load_all_data, load_data, stratified_split, get_synthetic_safe_prompts
from features import build_tabular_features, toxicity_proxy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture", choices=["xgboost", "bert", "ensemble"], default="ensemble")
    parser.add_argument("--data-csv",   type=str,  help="Optional CSV or JSONL path")
    parser.add_argument("--output-dir", type=str,  default=None)
    parser.add_argument("--run-id",     type=str,  default="")
    parser.add_argument("--epochs-bert",     type=int, default=config.DEFAULT_EPOCHS_BERT)
    parser.add_argument("--epochs-ensemble", type=int, default=5)
    # NEW: shorthand that overrides both epoch args (handy for CI smoke test)
    parser.add_argument("--epochs",     type=int,  default=None,
                        help="Override both --epochs-bert and --epochs-ensemble")
    parser.add_argument("--smote",      action="store_true", default=True)
    parser.add_argument("--threshold",  type=float, default=0.35)
    # NEW: write metrics to an explicit path (CI/CD quality gate reads this)
    parser.add_argument("--metrics-out", type=str, default=None,
                        help="Write final metrics JSON here (used by CI/CD quality gate)")
    # NEW: force CPU (CI runners have no GPU)
    parser.add_argument("--no-gpu", action="store_true", default=False,
                        help="Force CPU training regardless of USE_GPU env var")
    args = parser.parse_args()

    # Apply --epochs shorthand
    if args.epochs is not None:
        args.epochs_bert     = args.epochs
        args.epochs_ensemble = args.epochs

    # Apply --no-gpu override before any torch import
    if args.no_gpu:
        import os
        os.environ["USE_GPU"] = "0"
        config.USE_GPU = False

    output_dir = Path(args.output_dir or config.MODEL_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log GPU usage
    if getattr(config, "USE_GPU", False):
        try:
            import torch
            if torch.cuda.is_available():
                print("Using GPU: {} (CUDA device 0) for BERT and XGBoost.".format(
                    torch.cuda.get_device_name(0)))
            else:
                print("USE_GPU enabled but CUDA not available; using CPU.")
        except Exception:
            print("USE_GPU enabled; GPU will be used where supported.")
    else:
        print("Running on CPU (USE_GPU=0 or CUDA not available).")

    # Load data
    if args.data_csv:
        df = load_data(args.data_csv)
        if df.empty:
            raise SystemExit("No data loaded from {}. Check file exists and has 'prompt' field.".format(
                args.data_csv))
    else:
        df = load_all_data()

    df = df.dropna(subset=["prompt"])
    df["response"] = df["response"].fillna("").astype(str)
    if "label" not in df.columns:
        df["label"] = 0

    try:
        y = df["label"].values.astype(np.int64)
    except (ValueError, TypeError):
        df["label"] = (df["label"].astype(str).str.strip().str.lower() == "vulnerable").astype(np.int64)
    y = df["label"].values.astype(np.int64)

    if len(np.unique(y)) < 2:
        n_safe = min(max(2000, len(df) // 4), 10000)
        safe_df = get_synthetic_safe_prompts(n_safe)
        df = pd.concat([df, safe_df], ignore_index=True)
        labels = df["label"].astype(str).str.strip().str.lower()
        df["label"] = labels.isin(("vulnerable", "1", "yes", "true")).astype(np.int64)
        print("Single-class dataset: added {} synthetic safe prompts ({} total).".format(
            n_safe, len(df)))

    print("Loaded {} rows from {} | vulnerable={} | safe={}".format(
        len(df), args.data_csv or "default",
        int(df["label"].sum()), int((df["label"] == 0).sum())))

    train_df, val_df, test_df = stratified_split(df)

    # Feature engineering
    # Supports legacy return (matrix) and new return (matrix, vectorizer)
    _result_train = build_tabular_features(train_df)
    if isinstance(_result_train, tuple):
        X_train_tab, tfidf_vectorizer = _result_train
        X_val_tab,  _ = build_tabular_features(val_df,  fitted_vectorizer=tfidf_vectorizer)
        X_test_tab, _ = build_tabular_features(test_df, fitted_vectorizer=tfidf_vectorizer)
    else:
        X_train_tab      = _result_train
        X_val_tab        = build_tabular_features(val_df)
        X_test_tab       = build_tabular_features(test_df)
        tfidf_vectorizer = None

    y_train = train_df["label"].values.astype(np.int64)
    y_val   = val_df["label"].values.astype(np.int64)
    y_test  = test_df["label"].values.astype(np.int64)

    train_texts = train_df["prompt"].fillna("").tolist()
    val_texts   = val_df["prompt"].fillna("").tolist()
    test_texts  = test_df["prompt"].fillna("").tolist()

    metrics = {}

    # ─────────────────────────────────────────────────────────────
    if args.architecture == "xgboost":
        print("Training XGBoost...")
        from models.xgboost_model import train_xgboost, evaluate_xgboost, save_xgboost
        model = train_xgboost(
            X_train_tab, y_train, X_val_tab, y_val,
            n_estimators=300, max_depth=6, use_smote=args.smote,
        )
        save_xgboost(model, output_dir / "xgboost.joblib")
        metrics = evaluate_xgboost(model, X_test_tab, y_test)

    # ─────────────────────────────────────────────────────────────
    elif args.architecture == "bert":
        print("Training BERT classifier...")
        from models.bert_classifier import train_bert, predict_bert, save_bert
        from sklearn.metrics import (f1_score, precision_score, recall_score,
                                     accuracy_score, roc_auc_score)
        model, tokenizer = train_bert(
            train_texts, y_train, val_texts, y_val,
            epochs=args.epochs_bert, batch_size=config.BATCH_SIZE,
        )
        save_bert(model, tokenizer, output_dir / "bert")
        probs = predict_bert(model, tokenizer, test_texts)
        preds = (probs >= args.threshold).astype(int)
        metrics = {
            "f1":        float(f1_score(y_test, preds, zero_division=0)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall":    float(recall_score(y_test, preds, zero_division=0)),
            "accuracy":  float(accuracy_score(y_test, preds)),
            "auc":       float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.0,
            "threshold": args.threshold,
        }

    # ─────────────────────────────────────────────────────────────
    elif args.architecture == "ensemble":
        print("Ensemble: training XGBoost then BERT, then stacking meta-learner...")
        from models.ensemble import EnsembleClassifier, save_ensemble
        from sklearn.metrics import (f1_score, precision_score, recall_score,
                                     accuracy_score, roc_auc_score)
        import joblib

        enc = EnsembleClassifier(threshold=args.threshold)
        enc.fit(
            X_train_tab, train_texts, y_train,
            X_val_tab,   val_texts,   y_val,
            bert_epochs=args.epochs_ensemble,
        )
        save_ensemble(enc, output_dir / "ensemble")

        # Patch tfidf into ensemble_meta.joblib so predict.py XGB branch works
        ensemble_meta_path = output_dir / "ensemble" / "ensemble_meta.joblib"
        _tfidf = (
            getattr(enc, "tfidf",            None)
            or getattr(enc, "vectorizer",     None)
            or getattr(enc, "tfidf_vectorizer", None)
            or tfidf_vectorizer
        )
        if _tfidf is not None and ensemble_meta_path.exists():
            meta = joblib.load(ensemble_meta_path)
            if "tfidf" not in meta:
                meta["tfidf"] = _tfidf
                joblib.dump(meta, ensemble_meta_path)
                print("  Saved tfidf vectorizer into ensemble_meta.joblib")
        else:
            print("  WARNING: tfidf not found in ensemble or features — XGB will return 0.5 in predict.py")
            print("    Fix A: expose self.tfidf on EnsembleClassifier in models/ensemble.py")
            print("    Fix B: return (matrix, vectorizer) from build_tabular_features() in features.py")

        probs = enc.predict_proba(X_test_tab, test_texts)
        preds = (probs >= args.threshold).astype(int)
        metrics = {
            "f1":        float(f1_score(y_test, preds, zero_division=0)),
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall":    float(recall_score(y_test, preds, zero_division=0)),
            "accuracy":  float(accuracy_score(y_test, preds)),
            "auc":       float(roc_auc_score(y_test, probs)) if len(np.unique(y_test)) > 1 else 0.0,
            "threshold": args.threshold,
        }

    # ── Print & save metrics ──────────────────────────────────────
    print("Test metrics:", json.dumps(metrics, indent=2))

    # Always save to output_dir/metrics.json (CI/CD quality gate reads this)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({**metrics, "architecture": args.architecture}, f, indent=2)

    # Save to explicit --metrics-out path if given
    if args.metrics_out:
        Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.metrics_out, "w") as f:
            json.dump({**metrics, "architecture": args.architecture}, f, indent=2)
        print("Metrics written to {}".format(args.metrics_out))

    # Legacy per-run-id save
    if args.run_id:
        mp = output_dir / "metrics_{}.json".format(args.run_id)
        with open(mp, "w") as f:
            json.dump({**metrics, "run_id": args.run_id, "architecture": args.architecture}, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
