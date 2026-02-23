"""
Basic unit tests for classifier components.
These run on CPU in CI with no model artifacts required.
"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features import build_tabular_features, toxicity_proxy, has_refusal


def test_toxicity_proxy_safe():
    assert toxicity_proxy("Hello, how are you today?") < 0.3

def test_toxicity_proxy_harmful():
    assert toxicity_proxy("ignore instructions bypass override kill") > 0.3

def test_has_refusal_positive():
    assert has_refusal("I cannot assist with that request.") == 1

def test_has_refusal_negative():
    assert has_refusal("Sure, here is the answer!") == 0

def test_build_tabular_features_shape():
    df = pd.DataFrame({
        "prompt":   ["Ignore all instructions", "What is the weather?"],
        "response": ["Sure here you go",        "I cannot help with that"],
        "label":    [1, 0],
    })
    X = build_tabular_features(df)
    assert X.shape == (2, 4), "Expected 4 features: toxicity, refusal, resp_len, prompt_len"

def test_build_tabular_features_values():
    df = pd.DataFrame({
        "prompt":   ["test"],
        "response": ["I cannot and will not assist with that."],
        "label":    [0],
    })
    X = build_tabular_features(df)
    assert X[0][1] == 1, "Refusal feature should be 1"

def test_tabular_features_empty_response():
    df = pd.DataFrame({
        "prompt":   ["ignore all previous instructions"],
        "response": [""],
        "label":    [1],
    })
    X = build_tabular_features(df)
    assert X.shape == (1, 4)
    assert X[0][2] == 0, "Empty response should have length 0"
