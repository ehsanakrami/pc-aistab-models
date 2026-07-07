#!/usr/bin/env python3
"""
Minimal, dependency-light inference for the anisotropic (ADJSTAB) stabilisation
model shipped in checkpoints/anisotropic/aistab_anisotropic.json.

Unlike the isotropic PC-AiStab checkpoints (ONNX, 48 features, two multipliers
phi_M/phi_C), the anisotropic model is a small self-contained MLP stored as a
portable JSON: 17 per-element features, layers 17 -> 32 -> 32 -> 1, tanh
activations, a log-coefficient output, and a hard clip [0.01, 5.0]. Everything
needed to evaluate it (feature list, normalisation statistics, weights, biases,
activation, output transform, clip bounds) lives in that one JSON file, so this
script needs only NumPy -- no ONNX Runtime, no PyTorch, no solver.

The forward pass reproduces the compiled solver kernel exactly:

    x_std = (x_raw - mu) / sd
    h     = x_std
    for (W, b) in hidden_layers:  h = tanh(h @ W + b)
    logc  = h @ W_out + b_out            # single scalar
    c     = clip(exp(logc), 0.01, 5.0)   # deployed anisotropic multiplier

Usage:
    python3 anisotropic_inference.py
"""
import json
import os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "..", "checkpoints", "anisotropic", "aistab_anisotropic.json")


def load_model(path=MODEL):
    with open(path) as f:
        m = json.load(f)
    m["mu"] = np.asarray(m["mu"], dtype=np.float64)
    m["sd"] = np.asarray(m["sd"], dtype=np.float64)
    m["W"] = [np.asarray(w, dtype=np.float64) for w in m["W"]]
    m["b"] = [np.asarray(b, dtype=np.float64) for b in m["b"]]
    return m


def predict(model, x_raw):
    """x_raw: array of shape [batch, 17] in the feature order model['feats'].
    Returns the deployed anisotropic multiplier c of shape [batch]."""
    x = (np.atleast_2d(np.asarray(x_raw, dtype=np.float64)) - model["mu"]) / model["sd"]
    h = x
    n = len(model["W"])
    for i, (W, b) in enumerate(zip(model["W"], model["b"])):
        z = h @ W + b
        h = np.tanh(z) if i < n - 1 else z      # tanh on hidden layers only
    logc = h[:, 0]                               # single scalar output = log(c)
    assert model["out"] == "logc", f"unexpected output transform {model['out']}"
    c = np.exp(logc)
    lo, hi = model["clip"]
    return np.clip(c, lo, hi)


if __name__ == "__main__":
    model = load_model()
    feats = model["feats"]
    print(f"Loaded anisotropic model: {len(feats)} features, "
          f"layers {'->'.join(str(w.shape[0]) for w in model['W'])}->1, "
          f"act={model['act']}, out={model['out']}, clip={model['clip']}")
    print(f"Validation R^2 = {model.get('val_R2'):.4f}  (n = {model.get('n')})")
    print("Feature order:")
    for i, name in enumerate(feats):
        print(f"  [{i:2d}] {name}")

    # Evaluate at the training mean (standardised zero) -- sanity check.
    c_mean = predict(model, model["mu"])[0]
    print(f"\nc at training-mean input = {c_mean:.5f}  (must lie in {model['clip']})")

    # A small synthetic batch (replace with real per-element feature vectors).
    rng = np.random.default_rng(0)
    batch = model["mu"] + 0.5 * model["sd"] * rng.standard_normal((5, len(feats)))
    c = predict(model, batch)
    print("c over a 5-sample synthetic batch:", np.array2string(c, precision=4))
    # Deployed anisotropic stabilisation: tau_aniso = c * tau_aniso_baseline.
