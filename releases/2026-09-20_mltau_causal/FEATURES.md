# Fourteen-feature inference specification

All quantities use the solver's consistent nondimensional units; pressure is
kinematic pressure. Inputs are finite, viscosity nu > 0, positive element
lengths, and omega >= 0. Evaluate velocity u=(u,v), its gradient A=grad(u), and
pressure gradient g at the element centroid of the last accepted state.
Features and coefficients are frozen within a Newton solve, not differentiated
as functions of the current Newton state. A full startup, recovery and
relaxation policy belongs to the solver protocol, not to these weights.

Let U=|u|, h=sqrt(hL*hS), q=U+nu/h, hL>=hS>0,
S=| (A+A^T)/2 |_F, W=| (A-A^T)/2 |_F, B=|A|_F.
All logarithms are natural. Features are indexed from zero:

| Index | Value |
|---|---|
| 0 | log(1 + U*h/(2*nu)) |
| 1 | log(1 + uL*hL/(2*nu)) |
| 2 | log(1 + uS*hS/(2*nu)) |
| 3 | log(hL/hS) |
| 4 | log(1 + h*S/q) |
| 5 | log(1 + h*W/q) |
| 6 | 0.5*h^2*(W^2-S^2)/q^2 |
| 7 | log(1 + h*abs(trace(A))/q) |
| 8 | log(1 + h*B/q) |
| 9 | uL/q |
| 10 | log(1 + h*|g|/q^2) |
| 11 | h*(u dot g)/q^3 |
| 12 | log(1 + omega*h/q) |
| 13 | nu/(h*q) |

Two geometry conventions must not be confused:

- Original rectangular convention: hL,hS are the longer/shorter Cartesian
  element side lengths; uL,uS are the absolute velocity components along these
  axes. For equal side lengths the x axis is selected as the long axis.
- Mapped-Q1 extension: for centroid mapping Jacobian J from [-1,1]^2,
  G=J^(-T) J^(-1); hL=2/sqrt(lambda_min(G)),
  hS=2/sqrt(lambda_max(G)); uS^2=max(0,u^T G u/trace(G)),
  uL^2=max(0,U^2-uS^2). This is a metric split, not an eigenvector projection.
  The pressure specialists were fitted with this mapped convention.

The accepted-state rate is global, not elementwise:
omega = ||u_n-u_(n-1)||_L2 / (||u_n||_L2 * dt_last), using accepted finite-element
velocity fields and 2x2 Gauss integration for mapped Q1. Startup without a
usable accepted-state pair requires the solver's separately specified policy;
do not manufacture a feature from rejected iterates or reference solutions.

## Network

z=(x-mean)/scale; h1=tanh(W1*z+b1); h2=tanh(W2*h1+b2);
ell=W3*h2+b3; (c_m,c_c)=exp(clip(ell,log(0.002),log(60))).
Operations on output channels are componentwise. Matrices in JSON use
output-by-input ordering. Width comes from `width`; inputs=14, outputs=2.
No additional training-time gain or centring is applied to the exported
weights. The input-blind controls have an exactly zero first-layer matrix.

c_m multiplies the classical momentum stabilization used jointly in SUPG and
PSPG; c_c multiplies classical LSIC. These are not separate SUPG/PSPG outputs
and are not stabilization times in physical units. Reproducing the stabilized
flow solution additionally requires the matching base operator and solver
protocol; inference parity alone does not certify a discretization.

TXT layout: header `MLTAU_CAUSAL_V1 14 width 2`, followed by eight lines
containing mean, scale, W1, b1, W2, b2, W3, b3. Flatten matrices row-major.
Evaluate in double precision; synthetic fixtures specify absolute and relative
tolerances of 1e-12. Fixtures test arithmetic, not out-of-distribution safety.
