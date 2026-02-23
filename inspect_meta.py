#!/usr/bin/env python3
"""Quick diagnostic — prints all keys in ensemble_meta.joblib"""
import joblib, os, pprint

META_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "models", "ensemble", "ensemble_meta.joblib")
meta = joblib.load(META_PATH)
print("Keys in ensemble_meta.joblib:")
pprint.pprint(list(meta.keys()))
print()
for k, v in meta.items():
    print(f"  {k!r:30s}  type={type(v).__name__}")
