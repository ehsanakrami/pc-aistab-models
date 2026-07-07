# Anisotropic model — 17-feature specification

The anisotropic (ADJSTAB) model in
[`checkpoints/anisotropic/aistab_anisotropic.json`](../checkpoints/anisotropic/aistab_anisotropic.json)
takes a **17-component per-element feature vector** and returns a single scalar
anisotropic stabilisation multiplier. The feature list, order, and per-feature
normalisation (`mu`, `sd`) are all stored inside the JSON file; this document
describes what each feature means.

Features are evaluated per element from the local velocity `u`, its gradient
`∇u` (decomposed into the symmetric strain rate `S` and the antisymmetric spin
`W`), the element geometry (size `h`, per-axis sizes `h_long`/`h_short`), and the
kinematic viscosity `ν`. The long/short element axes are the longer/shorter of
the axis-aligned element dimensions.

| # | Name | Definition | Physical role |
|---|------|-----------|---------------|
| 0 | `umag` | \|u\| | local velocity magnitude |
| 1 | `Pe` | \|u\| h / (2ν) | cell Péclet number (isotropic-equivalent) |
| 2 | `Pe_long` | u_long · h_long / (2ν) | Péclet along the long element axis |
| 3 | `Pe_short` | u_short · h_short / (2ν) | Péclet along the short element axis |
| 4 | `Pe_aniso` | max(Pe_long, Pe_short) / min(Pe_long, Pe_short), capped at 100 | directional Péclet anisotropy ratio |
| 5 | `Snorm` | √(S : S) | strain-rate tensor magnitude |
| 6 | `Wnorm` | √(W : W) | spin (vorticity) tensor magnitude |
| 7 | `Q` | ½(Wnorm² − Snorm²) | Q-criterion (rotation vs strain) |
| 8 | `divu` | \|∇ · u\| | absolute velocity divergence (incompressibility residual) |
| 9 | `gradnorm` | ‖∇u‖_F | velocity-gradient Frobenius norm |
| 10 | `vsr` | Wnorm / Snorm | vorticity-to-strain ratio |
| 11 | `h` | element size | characteristic element length |
| 12 | `aspect` | max(hx, hy) / min(hx, hy) | element aspect ratio |
| 13 | `align` | u_long / \|u\| | alignment of the flow with the long axis (cosine) |
| 14 | `xi_coth` | coth(Pe) − 1/Pe | doubly-asymptotic SUPG upwind function at `Pe` |
| 15 | `xi_long` | coth(Pe_long) − 1/Pe_long | upwind function along the long axis |
| 16 | `xi_short` | coth(Pe_short) − 1/Pe_short | upwind function along the short axis |

## Evaluation

```
x_std = (x_raw - mu) / sd            # mu, sd from the JSON (length 17)
h     = x_std
for (W, b) in hidden_layers:         # two hidden layers, 17->32->32
    h = tanh(h @ W + b)
logc  = h @ W_out + b_out            # single scalar output
c     = clip(exp(logc), 0.01, 5.0)   # deployed anisotropic multiplier
```

`c` rescales the anisotropic stabilisation parameter relative to its analytic
baseline. The output transform is a log-coefficient (`out = "logc"`), so the
network predicts `log c` and the deployment exponentiates and clips to
`[0.01, 5.0]`. See [`examples/anisotropic_inference.py`](../examples/anisotropic_inference.py)
for a runnable NumPy-only implementation of exactly this forward pass.

## Relation to the isotropic PC-AiStab model

| | Isotropic (PC-AiStab) | Anisotropic (ADJSTAB) |
|---|---|---|
| Format | ONNX (`checkpoints/sh_target`, `checkpoints/residual_recovery`) | portable JSON (`checkpoints/anisotropic`) |
| Input features | 48 | 17 |
| Output | two multipliers `(φ_M, φ_C)` | one multiplier `c` |
| Supervision | SH-supervised / a-posteriori residual recovery | discrete-adjoint (error-optimal target) |
| Clip | `[0.5, 2.0]` on the enveloped multiplier | `[0.01, 5.0]` on `exp(logc)` |
| Purpose | bounded do-no-harm correction to SH | orientation-aware anisotropic accuracy gain |
