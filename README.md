# PC-AiStab: Trained ONNX Checkpoints

**Physics-Constrained Neural-Network Stabilisation Parameter for Incompressible Navier–Stokes**

This repository releases the trained ONNX inference models from the paper:

> E. Akrami, *PC-AiStab: a physics-constrained neural-network stabilisation parameter for incompressible Navier–Stokes*, submitted to *Computer Methods in Applied Mechanics and Engineering*, 2026.

The networks correct the Shakib–Hughes–Codina (SH) stabilisation parameter `τ` inside a classical SUPG/PSPG/LSIC stabilised finite-element solver. Each model accepts 48 per-element local features and returns two scalar multipliers `(φ_M, φ_C)` that rescale the momentum and continuity stabilisation parameters respectively.

---

## What is in this repository

```
checkpoints/
  sh_target/                   # SH-target campaign (trivial-target baseline)
    seed_{0..4}/
      pc_aistab_{variant}.onnx
      pc_aistab_{variant}_norm.json
  residual_recovery/            # Residual-recovery campaign (Phase B)
    seed_{0..4}/
      pc_aistab_phaseB_{variant}.onnx
      pc_aistab_phaseB_{variant}.onnx.data
      pc_aistab_phaseB_{variant}_norm.json
examples/
  inference.py                  # Minimal Python inference example
  batch_inference.py            # Batched inference over a set of element states
docs/
  features.md                   # Complete 48-feature specification
```

**Variants** — `{variant}` is one of:

| Variant | Training objective | Constraints active |
|---------|-------------------|-------------------|
| `apost` | A-posteriori loss only | none (unconstrained baseline) |
| `asymp` | A-posteriori + asymptotic penalty | asymptotic consistency |
| `mono`  | A-posteriori + asymptotic + monotonicity penalty | asymptotic + mesh-monotonicity |
| `full`  | A-posteriori + asymptotic + monotonicity + weight decay | all + L2 regularisation |

**Campaigns:**
- **`sh_target`** — the SH stabilisation parameter itself is the regression target; the loss minimum is at `φ = 1` (trivial fixed point). Produces a network that closely reproduces the SH baseline. Useful as an architectural and verification baseline.
- **`residual_recovery`** — the training target is a stabilised residual-recovery indicator; the loss minimum is not at `φ = 1` and the network produces meaningful departures. This is the campaign that yields a measurable AI-versus-SH correction on validation benchmarks.

**Seeds** — each variant is trained five times (`seed_0` through `seed_4`) with independent random initialisation to quantify seed-to-seed variance. For deployment, any single seed is sufficient; ensembling over seeds is not necessary.

---

## Architecture

```
Input:  x ∈ ℝ⁴⁸     (standardised per-element local features)
           ↓
  Linear(48 → 128) + GELU + LayerNorm
           ↓
  Linear(128 → 128) + GELU + LayerNorm
           ↓
  Linear(128 → 128) + GELU + LayerNorm
           ↓
  Linear(128 → 2)              (bias initialised to 0)
           ↓
Output: φ ∈ ℝ²      (φ_M, φ_C) — pre-shift raw network output
```

**Deployed multiplier** (computed outside the network, in the solver kernel):

```
μ(x) = clip( 1 + α(Re_h) · (φ(x) − 1),  φ_min,  φ_max )
```

where `α(Re_h) = Re_h² / (1 + Re_h²)` is the analytic envelope, and `(φ_min, φ_max) = (0.5, 2.0)` are the clip bounds. The stabilisation parameters become:

```
τ_M^PC = μ_M · τ_M^SH
τ_C^PC = μ_C · τ_C^SH
```

This construction guarantees:
1. **Stokes-limit recovery** — as Re_h → 0, μ → 1 and τ^PC → τ^SH regardless of network output.
2. **Two-sided bound** — τ^PC ∈ [0.5 τ^SH, 2.0 τ^SH] for any network output (Proposition 2 in the paper).
3. **Conditional h-monotonicity** — on elements where φ ≥ 1, τ^PC is monotone non-decreasing along mesh-refinement curves in feature space (Theorem 3 in the paper).

---

## Input features

The model input is a standardised 48-component vector of per-element, per-quadrature-point local quantities. Standardisation uses the `mean` and `std` arrays in the companion `*_norm.json` file:

```python
x_std = (x_raw - norm["mean"]) / norm["std"]
```

The 48 features are grouped by physical role:

| Group | Indices | Description |
|-------|---------|-------------|
| Asymptotic descriptors | 0–7   | Re_h, CFL, velocity magnitude, strain rate, vorticity, element size h, Jacobian, mesh quality |
| Residual norms         | 8–10  | Momentum residual, continuity residual, combined residual |
| Gradient norms         | 11–14 | ‖∇u‖, ‖∇p‖, convective acceleration, strain-rate magnitude |
| Tensorial detail       | 15–44 | 9 velocity-gradient components, 6 symmetric-strain components, 3 vorticity components, 6 metric-tensor components, 3 principal element sizes, 3 raw velocity components |
| Position               | 45–47 | Quadrature-point coordinates (x, y, z) |

Full descriptions and units are in [`docs/features.md`](docs/features.md).

---

## Quick-start: Python inference

```python
import json, numpy as np, onnxruntime as rt

# --- Load model and normalisation statistics ---
variant  = "full"          # or: apost, asymp, mono
campaign = "sh_target"     # or: residual_recovery (use phaseB filenames)
seed     = 0

model_path = f"checkpoints/{campaign}/seed_{seed}/pc_aistab_{variant}.onnx"
norm_path  = f"checkpoints/{campaign}/seed_{seed}/pc_aistab_{variant}_norm.json"

sess = rt.InferenceSession(model_path, providers=["CPUExecutionProvider"])
norm = json.load(open(norm_path))
mean = np.array(norm["mean"], dtype=np.float32)
std  = np.array(norm["std"],  dtype=np.float32)

# --- Prepare features (replace with real element state) ---
x_raw = np.zeros((1, 48), dtype=np.float32)   # shape: [batch, 48]
x_std = (x_raw - mean) / std

# --- Run inference ---
phi = sess.run(["phi"], {"features": x_std})[0]   # shape: [batch, 2]
phi_M, phi_C = phi[:, 0], phi[:, 1]               # SUPG and LSIC multipliers

# --- Compute deployed multiplier (kernel equation) ---
Re_h   = x_raw[:, 0]                              # feature index 0
alpha  = Re_h**2 / (1.0 + Re_h**2)               # envelope
phi_min, phi_max = 0.5, 2.0

mu_M = np.clip(1.0 + alpha * (phi_M - 1.0), phi_min, phi_max)
mu_C = np.clip(1.0 + alpha * (phi_C - 1.0), phi_min, phi_max)

# tau_M_PC = mu_M * tau_M_SH
# tau_C_PC = mu_C * tau_C_SH
```

For the **residual-recovery campaign**, the file names include `phaseB`:

```python
model_path = f"checkpoints/residual_recovery/seed_{seed}/pc_aistab_phaseB_{variant}.onnx"
norm_path  = f"checkpoints/residual_recovery/seed_{seed}/pc_aistab_phaseB_{variant}_norm.json"
```

The `.onnx` and `.onnx.data` files must reside in the same directory; ONNX Runtime loads them together automatically.

See [`examples/inference.py`](examples/inference.py) for a self-contained runnable script and [`examples/batch_inference.py`](examples/batch_inference.py) for batched inference over multiple element states.

---

## Requirements

| Package | Tested version | Notes |
|---------|---------------|-------|
| `onnxruntime` | ≥ 1.16 | CPU inference path; GPU not required |
| `numpy` | ≥ 1.24 | Feature preparation and post-processing |

Install via pip:

```bash
pip install onnxruntime numpy
```

No solver or training framework is required at inference time. The `.onnx` files are self-contained (all weights embedded; the `.onnx.data` files for the residual-recovery models contain the external weight tensors).

---

## ONNX model specification

| Property | Value |
|----------|-------|
| ONNX opset | 17 |
| Input name | `features` |
| Input shape | `[batch, 48]` |
| Input dtype | `float32` |
| Output name | `phi` |
| Output shape | `[batch, 2]` |
| Output dtype | `float32` |
| File size (sh_target) | ≈ 163 KB per model |
| File size (residual_recovery) | ≈ 144 KB (`.onnx` + `.onnx.data`) |

The ONNX graphs were exported from PyTorch 2.x (opset 17) and validated against the PyTorch reference to better than 10⁻⁶ maximum absolute difference on a 10 000-sample standard-normal batch.

---

## Numerical verification

Every checkpoint in this repository passed the following automated gates before release:

- **Finite-difference Jacobian gate** — the hand-coded Newton Jacobian of the stabilised FEM residual, with `τ^PC` as a frozen scalar weight, agrees with the finite-difference reference to machine precision (residue < 10⁻⁶) for all three network-output configurations tested (φ ≡ 1, φ ≡ 1.5, trained ONNX).
- **PyTorch–ONNX parity gate** — maximum absolute difference between PyTorch and ONNX Runtime output < 10⁻⁶ on a 10 000-sample standard-normal test batch.

---

## Citation

If you use these models in your research, please cite the following paper:

```bibtex
@article{Akrami2026PCAiStab,
  author  = {Akrami, Ehsan},
  title   = {{PC-AiStab}: a physics-constrained neural-network stabilisation
             parameter for incompressible {Navier--Stokes}},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  year    = {2026},
  note    = {Submitted; preprint available on request}
}
```

Compute resources for training were provided by the Irish Centre for High-End Computing (ICHEC) through project `ICHEC_lxdceng022b`, with access to the EuroHPC MeluXina supercomputer (LuxProvide, Luxembourg) under project `p201309`.

---

## License

The trained model weights and normalisation statistics in this repository are released under the [MIT License](LICENSE).

The underlying stabilised-FEM solver and training infrastructure are part of a separate private repository and are not included here.

---

## Contact

Ehsan Akrami — ehsanakrami1@gmail.com

Issues and questions can be filed via the [GitHub issue tracker](https://github.com/ehsanakrami/pc-aistab-models/issues).
