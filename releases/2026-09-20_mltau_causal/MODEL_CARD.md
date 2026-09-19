# Scope and limitations

Release date: 2026-09-20. Intended use: research into stabilization of laminar,
two-dimensional incompressible finite-element flow. Not validated for safety-
critical, turbulent, three-dimensional or arbitrary-geometry deployment.
Bounded positive coefficients are not a proof of stability or convergence.

## Model groups

The six RD/RS parent models are frozen models from three regenerated corpora.
The six pressure specialists share one 86016-row corpus derived from steady
DFG Re=20 on three exposed meshes; seeds do not constitute independent
corpora. The specialists use a 14-16-16-2 tanh network, 900 training epochs,
deployed log-coefficient loss and a 0.01 Jacobian penalty. The blind controls
use zeroed input weights, giving a correctly constant deployed prediction.
Training data and code are private and not included in this release.

## Evidence boundaries at release

- Coarse steady DFG has meaningful velocity/pressure improvements in some
  comparisons. All three blind specialist seeds achieve joint coarse gains
  on Re=12,20,28 with no more linear solves than SH. This is not a general
  learned-adaptivity or wall-clock-speedup claim.
- Fine-grid pressure remains worse than SH. A 64-case coefficient frontier
  did not produce a robust joint pressure/velocity gain across the two
  computed reference resolutions. Computed-reference discrepancies are not
  exact-solution errors.
- In three isolated coarse Re=20 timing blocks, blind specialists do not show
  a reliable wall-clock advantage. Accepted state-conditioned seeds cost
  about 1.54-1.61 times SH. `pressure_state_16003` fails all three timing-block
  acceptance checks and is retained as an experimental failed candidate.
- In the original 64-case periodic campaign, all native runs completed and
  54 passed recurrence, but none passed full frozen qualification. Time
  refinement remained unresolved. Finer-timestep experiments are pending;
  no periodic qualification is claimed for these released weights.
- Classical-initialized BFS recovery is not unassisted learned cold-start
  reliability. Failed or recovered cases must not be presented as ML wins.

The model files enable inspection and independent arithmetic replay, not full
external replication of the private solver study. No journal acceptance,
submission readiness, coauthor approval, general dominance over SH/MRSS,
pressure robustness or guaranteed physical reliability is implied.
