"""
PC-AiStab: batched inference over all variants and seeds.

Runs all four variants from a single campaign through a batch of random
feature vectors and reports mean/std of the deployed multiplier mu_M.
Useful for validating that all checkpoints load correctly and for comparing
the output spread between variants (Table 3 in the paper).

Requirements: onnxruntime >= 1.16, numpy >= 1.24
"""

import json
import pathlib

import numpy as np
import onnxruntime as rt

CAMPAIGN    = "sh_target"    # "sh_target" or "residual_recovery"
VARIANTS    = ["apost", "asymp", "mono", "full"]
SEEDS       = [0, 1, 2, 3, 4]
BATCH_SIZE  = 10_000         # number of random feature vectors

repo_root = pathlib.Path(__file__).parent.parent

rng = np.random.default_rng(42)

print(f"Campaign: {CAMPAIGN}   batch_size={BATCH_SIZE}")
print(f"{'Variant':<8}  {'Seed':<4}  "
      f"{'mean(μ_M)':>12}  {'std(μ_M)':>10}  "
      f"{'min(μ_M)':>10}  {'max(μ_M)':>10}")
print("-" * 62)

for variant in VARIANTS:
    results_mu = []

    for seed in SEEDS:
        if CAMPAIGN == "sh_target":
            stem = f"pc_aistab_{variant}"
        else:
            stem = f"pc_aistab_phaseB_{variant}"

        ckpt_dir   = repo_root / "checkpoints" / CAMPAIGN / f"seed_{seed}"
        model_path = ckpt_dir / f"{stem}.onnx"
        norm_path  = ckpt_dir / f"{stem}_norm.json"

        norm = json.loads(norm_path.read_text())
        mean = np.array(norm["mean"], dtype=np.float32)
        std  = np.array(norm["std"],  dtype=np.float32)

        # Random inputs from standard normal (tests the full output range)
        x_std = rng.standard_normal((BATCH_SIZE, 48)).astype(np.float32)
        x_raw = x_std * std + mean

        sess = rt.InferenceSession(str(model_path),
                                   providers=["CPUExecutionProvider"])
        phi   = sess.run(["phi"], {"features": x_std})[0]
        phi_M = phi[:, 0]

        Re_h  = x_raw[:, 0]
        alpha = Re_h**2 / (1.0 + Re_h**2)
        mu_M  = np.clip(1.0 + alpha * (phi_M - 1.0), 0.5, 2.0)
        results_mu.append(mu_M)

        print(f"{variant:<8}  {seed:<4}  "
              f"{mu_M.mean():12.6f}  {mu_M.std():10.6f}  "
              f"{mu_M.min():10.6f}  {mu_M.max():10.6f}")

    # Aggregate across seeds
    all_mu = np.concatenate(results_mu)
    print(f"{'':>14}  {'[all seeds]':<5}  "
          f"{all_mu.mean():12.6f}  {all_mu.std():10.6f}  "
          f"{all_mu.min():10.6f}  {all_mu.max():10.6f}")
    print()
