# Learned Stabilisation-Parameter Models for Incompressible Navier–Stokes

## Current model-only release — 20 September 2026

The current 14-feature causal models are in
[`releases/2026-09-20_mltau_causal`](releases/2026-09-20_mltau_causal).
This version contains twelve frozen models, normalization statistics,
synthetic inference fixtures and explicit validation limitations. It contains
no solver source, training code or training corpus. The new models are
experimental: fine-grid pressure and periodic qualification remain unresolved,
and a general learned wall-clock speedup is not established. Read the model
card before use. The existing MIT licence applies to this model release.

The sections below describe older 48- and 17-feature models and historical
manuscript claims; they are not the specification or validation record of the
new 14-feature release. Historical submission statements and the reserved
Zenodo identifier below do not certify the status or availability of the
current manuscript. Cite the new release by its version and commit permalink.

## Legacy model documentation

**PC-AiStab (isotropic, do-no-harm) and ADJSTAB (anisotropic, adjoint-supervised) neural stabilisation-parameter models**

This repository releases the trained inference models from the paper:

> E. Akrami and Y. Delauré, *A Verified Discrete-Adjoint Framework for Stabilization-Parameter Accuracy in Laminar Incompressible Flow: The Headroom Principle, Parameter-Free Closed Forms, and Certified Deployment*, submitted to *Computer Methods in Applied Mechanics and Engineering*, 2026.

Earlier documentation reserved Zenodo identifier `10.5281/zenodo.21237054` for a broader capsule. It is not the access location for the current release, and its public availability is not asserted here. The current distribution decision is model-only: solver and training source remain private, with no promise of public source release on acceptance.

The models correct the Shakib–Hughes–Codina (SH) stabilisation parameter `τ` inside a classical SUPG/PSPG/LSIC stabilised finite-element solver. The repository contains **two complementary learned multipliers**:

- **PC-AiStab (isotropic)** — a bounded, SH-supervised network that accepts 48 per-element features and returns two multipliers `(φ_M, φ_C)` rescaling the momentum and continuity stabilisation parameters. Designed as a *do-no-harm* correction that provably recovers SH in the Stokes limit. Released as ONNX in `checkpoints/sh_target/` and `checkpoints/residual_recovery/`.
- **ADJSTAB (anisotropic)** — a discrete-adjoint-supervised, orientation-aware model that accepts 17 per-element features and returns a single anisotropic stabilisation multiplier `c`. Supervised on an error-optimal (adjoint) signal for a genuine accuracy gain on stretched, anisotropic elements. Released as a portable JSON in `checkpoints/anisotropic/`.

The sections below document the isotropic PC-AiStab models first (architecture, features, ONNX inference); the anisotropic ADJSTAB model is documented under [The anisotropic model (ADJSTAB)](#the-anisotropic-model-adjstab).

---

## What is in this repository

```
checkpoints/
  sh_target/                    # Isotropic PC-AiStab, SH-target campaign (trivial-target baseline)
    seed_{0..4}/
      pc_aistab_{variant}.onnx
      pc_aistab_{variant}_norm.json
  residual_recovery/            # Isotropic PC-AiStab, residual-recovery campaign (Phase B)
    seed_{0..4}/
      pc_aistab_phaseB_{variant}.onnx
      pc_aistab_phaseB_{variant}.onnx.data
      pc_aistab_phaseB_{variant}_norm.json
  anisotropic/                  # Anisotropic ADJSTAB model (portable, self-contained JSON)
    aistab_anisotropic.json
examples/
  inference.py                  # Minimal ONNX inference (isotropic PC-AiStab)
  batch_inference.py            # Batched ONNX inference (isotropic PC-AiStab)
  anisotropic_inference.py      # NumPy-only inference (anisotropic ADJSTAB)
docs/
  features.md                   # 48-feature specification (isotropic PC-AiStab)
  anisotropic_features.md       # 17-feature specification (anisotropic ADJSTAB)
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

## The anisotropic model (ADJSTAB)

The isotropic PC-AiStab models above are a bounded *do-no-harm* correction to SH. The **anisotropic** model is the paper's second learned multiplier: it is supervised on an **error-optimal discrete-adjoint** signal (rather than the SH parameter) and is **orientation-aware**, so it delivers a genuine accuracy gain on stretched, anisotropic elements where the isotropic SH form is least sharp.

It ships as a single **portable, self-contained JSON** — [`checkpoints/anisotropic/aistab_anisotropic.json`](checkpoints/anisotropic/aistab_anisotropic.json) — that carries the feature list, the normalisation statistics, all weights and biases, the activation, the output transform, and the clip bounds. No ONNX Runtime, PyTorch, or solver is needed to evaluate it; NumPy suffices.

| Property | Value |
|----------|-------|
| Input features | 17 per-element quantities (see [`docs/anisotropic_features.md`](docs/anisotropic_features.md)) |
| Architecture | MLP `17 → 32 → 32 → 1`, `tanh` hidden activations |
| Output | single scalar `logc`; deployed multiplier `c = clip(exp(logc), 0.01, 5.0)` |
| Supervision | discrete-adjoint (error-optimal) target |
| Training samples | 48 224 (multi-seed final model) |
| Validation R² | 0.685 |

**Evaluation** (reproduces the compiled solver kernel exactly):

```python
import json, numpy as np

m = json.load(open("checkpoints/anisotropic/aistab_anisotropic.json"))
mu, sd = np.array(m["mu"]), np.array(m["sd"])          # length-17 normalisation
W = [np.array(w) for w in m["W"]]                      # [(17,32),(32,32),(32,1)]
b = [np.array(v) for v in m["b"]]

def predict(x_raw):                                    # x_raw: [batch, 17]
    h = (np.atleast_2d(x_raw) - mu) / sd
    for i, (Wi, bi) in enumerate(zip(W, b)):
        z = h @ Wi + bi
        h = np.tanh(z) if i < len(W) - 1 else z        # tanh on hidden layers only
    return np.clip(np.exp(h[:, 0]), *m["clip"])        # deployed multiplier c

# tau_aniso = c * tau_aniso_baseline
```

The feature order and per-feature definitions are in [`docs/anisotropic_features.md`](docs/anisotropic_features.md); see [`examples/anisotropic_inference.py`](examples/anisotropic_inference.py) for a runnable script. The same weights are compiled into the paper's solver, so this JSON reproduces the deployed anisotropic parameter bit-for-bit.

---

## Requirements

The isotropic PC-AiStab models are ONNX and need `onnxruntime`; the anisotropic ADJSTAB model is pure JSON and needs only `numpy`.

| Package | Tested version | Notes |
|---------|---------------|-------|
| `onnxruntime` | ≥ 1.16 | CPU inference path for the isotropic ONNX models; GPU not required |
| `numpy` | ≥ 1.24 | Feature preparation, post-processing, and all anisotropic-model inference |

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

If you use these models in your research, please cite the paper and the archived capsule:

```bibtex
@article{AkramiDelaure2026MLTau,
  author  = {Akrami, Ehsan and Delaur\'e, Yan},
  title   = {A Verified Discrete-Adjoint Framework for Stabilization-Parameter
             Accuracy in Laminar Incompressible Flow: The Headroom Principle,
             Parameter-Free Closed Forms, and Certified Deployment},
  journal = {Computer Methods in Applied Mechanics and Engineering},
  year    = {2026},
  note    = {Submitted}
}

@software{AkramiDelaure2026Capsule,
  author    = {Akrami, Ehsan and Delaur\'e, Yan},
  title     = {Reproducibility capsule for ``A Verified Discrete-Adjoint
               Framework for Stabilization-Parameter Accuracy in Laminar
               Incompressible Flow''},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21237054}
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
