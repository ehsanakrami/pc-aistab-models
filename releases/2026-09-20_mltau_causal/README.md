# Causal stabilization models: 2026-09-20

Experimental weights supporting the working manuscript *Learned Stabilization
of Incompressible Flow: Adaptivity, Reliability and Deployment Cost*.
This is a model-only release, not a claim of journal submission or acceptance.

Twelve frozen models are supplied: three deployed-loss parents (`rdeploy_0..2`),
three secant-regularized parents (`rsecant_0..2`), and three state-conditioned
and three input-blind pressure specialists (seeds 16001, 16002, 16003).
All seeds, including an unreliable specialist, are retained. This release
does not contain every historical model evaluated in the manuscript.

Read [MODEL_CARD.md](MODEL_CARD.md) before deployment and
[FEATURES.md](FEATURES.md) for the exact inference convention.
`MODEL_MANIFEST.json` identifies models and hashes; `SHA256SUMS.json` covers
all other release files. `INFERENCE_FIXTURES.json` contains synthetic inputs
and expected outputs, not samples from the training corpus.

Each JSON contains normalization statistics and inference weights. The
matching TXT is byte-identical to the native deployment file. Weight matrices
are output-by-input. There is no runtime or source implementation in this
release. It is distinct from this repository's older 48- and 17-feature models.

## Availability and licence

The weights, normalization statistics, synthetic fixtures and documentation
in this folder are released under the repository's existing MIT licence.
The finite-element solver, training code, corpus and private raw-field evidence
are not distributed. No permission to redistribute them is granted here.
These files allow independent inference checks, but are not a complete package
for reproducing the flow experiments. There is no promise of future source
release and no reserved DOI is offered as an accessible archive.

Cite this version using its Git commit permalink and the model identifiers.
The associated manuscript is a working manuscript, not a published article.
