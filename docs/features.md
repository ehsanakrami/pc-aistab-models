# PC-AiStab: 48-Feature Specification

This document describes the 48 input features accepted by every PC-AiStab
checkpoint. The feature vector is computed locally at each quadrature point
of each element by the in-solver feature extractor, standardised with the
per-feature mean and standard deviation stored in the companion `*_norm.json`
file, and passed to the ONNX model.

All quantities are dimensionless unless otherwise noted, and are computed at
the quadrature point of the current Newton iteration.

---

## Group 1 — Asymptotic descriptors (indices 0–7)

These eight quantities are the natural arguments of the closed-form SH
stabilisation parameter `τ_SH`. Including them gives the network direct
access to the information the classical formula uses, so asymptotic limits
reduce to simple linear projections rather than nonlinear reconstructions.

| Index | Name in `norm.json` | Description |
|-------|---------------------|-------------|
| 0 | `Re_h` | Element Reynolds number `Re_h = ‖u‖ h_K / ν` |
| 1 | `CFL` | CFL-like number `‖u‖ Δt / h_K` |
| 2 | `velocity_mag` | Velocity magnitude `‖u‖` |
| 3 | `strain_rate` | Frobenius norm of the strain-rate tensor `‖S‖_F` |
| 4 | `vorticity` | Vorticity magnitude `‖ω‖` |
| 5 | `h_normalized` | Normalised element size `h_K` |
| 6 | `jacobian` | Element Jacobian determinant (volume/area measure) |
| 7 | `quality` | Element quality metric (1 = ideal, 0 = degenerate) |

## Group 2 — Residual norms (indices 8–10)

These three quantities summarise the local discrepancy between the current
discrete solution and the strong form of the incompressible Navier–Stokes
equations. They are the single piece of information that the classical `τ`
formula does not see: an a-posteriori indicator of where the SH baseline
is and is not asymptotically optimal.

| Index | Name in `norm.json` | Description |
|-------|---------------------|-------------|
| 8  | `momentum_residual` | Magnitude of the discrete strong-form momentum residual at the quadrature point |
| 9  | `continuity_residual` | Magnitude of the discrete strong-form continuity residual |
| 10 | `combined_residual` | Combined residual norm (used as the a-posteriori training target in Phase B) |

## Group 3 — Gradient norms (indices 11–14)

Shear-layer and pressure-gradient information that the asymptotic balance
ignores. Where `‖∇u‖` is large the convective term dominates and the SUPG
stabilisation is load-bearing; where `‖∇p‖` is large the PSPG correction
is load-bearing.

| Index | Name in `norm.json` | Description |
|-------|---------------------|-------------|
| 11 | `velocity_grad_norm` | Frobenius norm of the velocity-gradient tensor `‖∇u‖_F` |
| 12 | `pressure_grad_norm` | L2 norm of the pressure gradient `‖∇p‖` |
| 13 | `convective_accel`   | Magnitude of the convective acceleration `‖(u·∇)u‖` |
| 14 | `strain_rate_mag`    | Strain-rate magnitude (scalar invariant) |

## Group 4 — Tensorial detail (indices 15–44)

Thirty components that let the network reconstruct any anisotropic
combination of the local flow and mesh tensors without requiring the
feature engineer to guess which invariants are important. The cost is a
wider input layer; the benefit is that the closure is not limited by a
feature-engineering choice.

| Indices | Name pattern | Description |
|---------|-------------|-------------|
| 15–23  | `vel_grad_ij` | All 9 components of the velocity-gradient tensor `∂u_i/∂x_j` (i,j ∈ {x,y,z}) |
| 24–29  | `strain_ij`   | 6 independent components of the symmetric strain-rate tensor `S_ij = (∂u_i/∂x_j + ∂u_j/∂x_i)/2` |
| 30–32  | `vort_k`      | 3 components of the vorticity vector `ω = ∇ × u` |
| 33–38  | `metric_ij`   | 6 independent components of the element metric tensor `G_ij` (related to the element Jacobian; encodes mesh anisotropy and orientation) |
| 39–41  | `h_x, h_y, h_z` | Principal element sizes along the three coordinate directions |
| 42–44  | `u_raw, v_raw, w_raw` | Raw velocity components `(u, v, w)` (not magnitude) |

## Group 5 — Position (indices 45–47)

Three coordinates of the quadrature point in the physical domain. For the
cavity geometries used in the paper the position is largely redundant with
the rest of the local state, and the trained network is empirically
insensitive to it. For strongly inhomogeneous geometries (e.g. multi-body
configurations, re-entrant corners) position-awareness enables the network
to learn geometry-specific behaviour.

| Index | Name in `norm.json` | Description |
|-------|---------------------|-------------|
| 45 | `position_x` | x-coordinate of the quadrature point |
| 46 | `position_y` | y-coordinate of the quadrature point |
| 47 | `position_z` | z-coordinate of the quadrature point (0 in 2D) |

---

## Standardisation

Before passing the feature vector to the model, each component must be
standardised:

```python
x_std[i] = (x_raw[i] - mean[i]) / std[i]
```

The `mean` and `std` arrays (one value per feature) are stored in the
`*_norm.json` sidecar file alongside each `.onnx` checkpoint. They were
computed over the full training corpus (lid-driven cavity flows at
Re ∈ {100, 400, 1000}, all mesh levels) and are specific to each variant
and campaign. Do not mix norm files across variants or campaigns.

**Important:** If any feature has `std = 0` (a constant feature in the
training corpus — for example `vel_grad_xz` in 2D where the z-components
vanish), the standardisation is replaced by division by 1 to avoid
division by zero. The norm files ship with `std = 1` for these components.

---

## Notes on 2D vs 3D

The feature extractor and ONNX models were trained on two-dimensional
problems. In 2D:
- All z-components of velocity gradient, strain, and vorticity are zero.
- `position_z = 0` at all quadrature points.
- The principal element size `h_z` reflects the unit out-of-plane thickness.

For three-dimensional deployment the same 48-feature vector applies; the
out-of-plane components will carry meaningful information and the network
should be retrained on 3D data for best results.
