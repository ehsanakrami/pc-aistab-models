"""
PC-AiStab: minimal inference example.

Loads a single checkpoint and computes the deployed multiplier (mu_M, mu_C)
for an arbitrary element feature vector. Replace the zero-filled x_raw array
with actual per-element, per-quadrature-point features from your solver.

Requirements: onnxruntime >= 1.16, numpy >= 1.24
"""

import json
import pathlib

import numpy as np
import onnxruntime as rt

# ---------------------------------------------------------------------------
# Configuration — edit these three lines
# ---------------------------------------------------------------------------
CAMPAIGN = "sh_target"       # "sh_target" or "residual_recovery"
VARIANT  = "full"            # "apost", "asymp", "mono", or "full"
SEED     = 0                 # 0 through 4

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
repo_root = pathlib.Path(__file__).parent.parent

if CAMPAIGN == "sh_target":
    model_stem = f"pc_aistab_{VARIANT}"
else:
    model_stem = f"pc_aistab_phaseB_{VARIANT}"

checkpoint_dir = repo_root / "checkpoints" / CAMPAIGN / f"seed_{SEED}"
model_path = checkpoint_dir / f"{model_stem}.onnx"
norm_path  = checkpoint_dir / f"{model_stem}_norm.json"

# ---------------------------------------------------------------------------
# Load model and normalisation statistics
# ---------------------------------------------------------------------------
sess = rt.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
norm = json.loads(norm_path.read_text())
mean = np.array(norm["mean"], dtype=np.float32)   # shape (48,)
std  = np.array(norm["std"],  dtype=np.float32)   # shape (48,)
feature_names = norm["feature_names"]             # list of 48 strings

print(f"Loaded: {model_path.name}")
print(f"  Input:  {sess.get_inputs()[0].shape}")
print(f"  Output: {sess.get_outputs()[0].shape}")

# ---------------------------------------------------------------------------
# Construct a representative feature vector
# (replace with real element-state data from your solver)
# ---------------------------------------------------------------------------
# Each row is one quadrature point; batch inference is supported.
batch_size = 4
x_raw = np.zeros((batch_size, 48), dtype=np.float32)

# Set physically meaningful example values for the asymptotic descriptors
# (indices 0–7; see docs/features.md for the full specification)
Re_h_idx = 0
x_raw[:, Re_h_idx] = np.array([0.5, 1.0, 5.0, 20.0])   # element Reynolds numbers

# ---------------------------------------------------------------------------
# Standardise and run inference
# ---------------------------------------------------------------------------
x_std = (x_raw - mean) / std                          # shape (batch, 48)
phi   = sess.run(["phi"], {"features": x_std})[0]     # shape (batch, 2)

phi_M = phi[:, 0]   # SUPG multiplier (raw network output, pre-kernel)
phi_C = phi[:, 1]   # LSIC multiplier (raw network output, pre-kernel)

# ---------------------------------------------------------------------------
# Apply the PC-AiStab multiplicative kernel
# (Equation (pc_kernel) in the paper)
# ---------------------------------------------------------------------------
phi_min, phi_max = 0.5, 2.0

Re_h  = x_raw[:, Re_h_idx]
alpha = Re_h**2 / (1.0 + Re_h**2)    # analytic envelope α(Re_h)

mu_M = np.clip(1.0 + alpha * (phi_M - 1.0), phi_min, phi_max)
mu_C = np.clip(1.0 + alpha * (phi_C - 1.0), phi_min, phi_max)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
print(f"\n{'Re_h':>8}  {'φ_M':>8}  {'φ_C':>8}  {'μ_M':>8}  {'μ_C':>8}")
print("-" * 48)
for i in range(batch_size):
    print(f"{Re_h[i]:8.2f}  {phi_M[i]:8.5f}  {phi_C[i]:8.5f}"
          f"  {mu_M[i]:8.5f}  {mu_C[i]:8.5f}")

print("\nDeployed stabilisation parameters:")
print("  tau_M_PC = mu_M * tau_M_SH")
print("  tau_C_PC = mu_C * tau_C_SH")
