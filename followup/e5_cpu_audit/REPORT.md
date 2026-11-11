# E5 exploratory CPU audit

Frozen θ=.8, δ=.05. All six original FORGET3 trajectories, full complex, all 49 checkpoints and 48 adjacent bridges per trajectory. Three independent worlds; seeds/checkpoints are repeated observations. No outcome fitting, GPU work or new confirmation evidence.

Failures: 0. Runtime: 22.9 seconds.

κ: induced-map quotient vs independent chain-space quotient; α: historical induced support vs independent chain cycles; rho: H0 MV kernel vs H1 quotient. Delta monotonicity at 0,.05; no b/d monotonicity asserted.

Full per-checkpoint α, historical connecting ranks, every adjacent ρ, all three actual κ source-set values at δ=0/.05, zero locations and failures are in results.json. A zero is a substantive observed value, not omitted or replaced.

## world635_S1

{"alpha_final": 3, "alpha_zero_steps": [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336], "rho_nonzero_bridges": [[480, 496]], "kappa005_nonzero": {"left": [[176, 192], [304, 320], [480, 496], [496, 512]], "right": [[288, 304], [480, 496]], "both": [[480, 496]]}, "zero_counts": {"rho": 47, "kappa005_left": 44, "kappa005_right": 46, "kappa005_both": 47}}

## world635_S2

{"alpha_final": 3, "alpha_zero_steps": [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352], "rho_nonzero_bridges": [[272, 288], [288, 304], [480, 496]], "kappa005_nonzero": {"left": [[192, 208], [304, 320], [496, 512]], "right": [[272, 288], [288, 304], [480, 496]], "both": []}, "zero_counts": {"rho": 45, "kappa005_left": 45, "kappa005_right": 45, "kappa005_both": 48}}

## world636_S1

{"alpha_final": 3, "alpha_zero_steps": [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 240, 256, 272, 288, 304, 320, 336, 352], "rho_nonzero_bridges": [[208, 224], [480, 496]], "kappa005_nonzero": {"left": [[192, 208], [304, 320], [368, 384], [496, 512]], "right": [[288, 304], [480, 496]], "both": []}, "zero_counts": {"rho": 46, "kappa005_left": 44, "kappa005_right": 46, "kappa005_both": 48}}

## world636_S2

{"alpha_final": 3, "alpha_zero_steps": [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336], "rho_nonzero_bridges": [[288, 304]], "kappa005_nonzero": {"left": [[224, 240], [496, 512]], "right": [[288, 304], [480, 496]], "both": []}, "zero_counts": {"rho": 47, "kappa005_left": 46, "kappa005_right": 46, "kappa005_both": 48}}

## world637_S1

{"alpha_final": 3, "alpha_zero_steps": [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320], "rho_nonzero_bridges": [[480, 496]], "kappa005_nonzero": {"left": [[208, 224], [224, 240], [480, 496], [496, 512]], "right": [[288, 304], [480, 496]], "both": [[480, 496]]}, "zero_counts": {"rho": 47, "kappa005_left": 44, "kappa005_right": 46, "kappa005_both": 47}}

## world637_S2

{"alpha_final": 3, "alpha_zero_steps": [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320], "rho_nonzero_bridges": [[480, 496]], "kappa005_nonzero": {"left": [[208, 224], [304, 320], [496, 512]], "right": [[288, 304], [480, 496]], "both": []}, "zero_counts": {"rho": 47, "kappa005_left": 45, "kappa005_right": 46, "kappa005_both": 48}}

## Limits

This checks computation and finite trajectory illustrations. It does not test predictive utility, establish a mechanism, validate sampling coverage, or resolve mathematical originality. The independent chain implementation shares only the complex specification and threshold masks; it does not call CW homology, induced maps or rank. Threshold-mask extraction itself is not independently audited. No perturbation coverage claim is tested. Only κ monotonicity between the two frozen scales is checked; derived birth/death differences need not be monotone.

Run locally with Python and NumPy: python audit.py. Existing repository routines are imported read-only.