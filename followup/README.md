# Follow-up analyses and seven-operator source bundle

This directory contains analyses of the original FORGET3 runs and code,
protocols, and results for the later composition and rule-exchange studies.
E0–E3 are post-hoc analyses of the six original confirmation arms.

## Inputs and execution

Run from the repository root with Python 3 and the dependencies below. Original raw inputs already in this repository are `cloud_return_round4/complex_worlds/runs/RCN12_FORGET3/world{635,636,637}_S{1,2}/eval_*.jsonl.gz` and each run manifest; world specifications, manifests, probes and training streams are under `complex_worlds/data/RCN12_FORGET3/world{635,636,637}`. All 49 saved checkpoints and both seeds are used. Existing modules come from `crosslingual/code`, `stage_ab/code`, and `paper_sec4`; archived comparison JSONs are `paper_sec4/theta_sweep.json` and `paper_sec4/mv_counts.json`. No model training or download is required.

The archived scripts use the original workspace layout. The launcher supplies the current repository path when it loads them. The commands write outputs beside the scripts, so use a working copy to keep the bundled results for comparison.

```sh
python -m pip install -r requirements.txt scipy
python followup/run.py e0_e3/analyze_e0_e3.py
python followup/run.py e0_e3/event_bootstrap.py
python followup/run.py e0_e3/verify_derived.py
python followup/run.py e0_e3/make_figures.py
python followup/run.py behavior/analyze_behavior.py
python followup/run.py behavior/write_report.py
python followup/run.py confidence/audit_design.py
python followup/run.py confidence/certify.py
python followup/run.py confidence/verify_independent.py
python followup/e4_full_invariant_pilot/pilot.py
```

## Manuscript mapping and status

| Directory | Current supplement anchor | Evidence and interpretation |
|---|---|---|
| `e0_e3/` | `supp:barcodes`, `supp:calibration`, `supp:behaviour`, `fig:e0e3`, `fig:sweep` | Post-hoc labelled-complex/history diagnostic, item bootstrap score deviations and prespecified-transition resampling. `derived_results.json` and `event_bootstrap.json` are analysis snapshots. `make_figures.py` produces `e1_e3_evidence.svg/pdf`, the source chain for manuscript `figures/fig_e0e3.pdf`; the third panel is the unmatched marginal comparison, not the matched result. Seed 20260925; 1000 resamples (see scripts for each procedure). |
| `behavior/` | `supp:behaviour` (structural and measurement diagnostics) | Post-hoc matched template/input conjunction and replacement baseline; held-out-world ridge predictive comparison using root-disjoint targets. `results.json` is primary; report describes methods. Seed 20261108; 2000 joint root resamples. No homological-predictor advantage or new training confirmation is claimed. |
| `confidence/` | `supp:calibration` (finite-sample calibration) | Conditional finite-population simultaneous confidence-band diagnostic on the B subcomplex, six arms and 49 checkpoints; `real_results.json`, design audit and independent chain-space validation. `*_design_matched.npz` contain point/lower/upper arrays for independent verification; inspect `np.load(...).files` for additional stored fields. `*_iid_cp_counterfactual.npz` are explicitly counterfactual iid calibration, not coverage for the observed design. |
| `e5_cpu_audit/` | `supp:barcodes` and `supp:calibration` (full-complex computational checks) | Exploratory dense-chain cross-check of historical $\alpha_1$, adjacent $\rho_1$, and windowed $\kappa_1$ at $\theta=0.8$, $\delta=0.05$ for all six old trajectories. `audit.py` regenerates `results.json` and `REPORT.md` in about 24 CPU seconds. `two_cell_deletion.py` verifies the checkpoint-skeleton identity by removing all two-cells and comparing every $\alpha_1$ value and the world636-S1 step-240 equality-condition accounting; `tau_cell_deletion.py` checks two-sided transition invariance and one-sided exceptions. See their reports. There are three independent worlds; these checks are neither new confirmation nor predictive-utility results. |
| `e4_full_invariant_pilot/` | `supp:e4` (post hoc invariant reconstruction) | Exploratory reuse of the already inspected E4 worlds. `pilot.py` reads frozen E4 feature rows, computes actual historical, bridge and windowed invariants and causal-prefix barcodes, then refits two augmentations of the raw-temporal control using only the two pilot worlds. `INVARIANTS.json`, `RESULTS.json`, and `REPORT.md` are reproducible outputs. All $\alpha_k$, $\rho_k$, and two-sided transitions are zero in these trajectories; the selected augmentations do not improve the control. These are descriptive results, not a new confirmation. |

Confidence `simulation_results.json` is **simulation**, and `sample_calibration.json` is hypothetical iid fresh-root/pair planning, not new observations. `validation.json` checks algebra; `independent_validation.json` independently checks saved real-data certificates. The confidence seed is 20261108. Source prose mentions a cyclic-pair permutation bound; at the actual family size its clipped bounds are [0,1], matching the conservative description in the current supplement. Do not generalise that equivalence to a different sample size/family.

Each subdirectory report or README identifies its primary outputs. The E4 result addendum is documented in `e4_v4/results/README.md`.

## Seven-operator designs and assessments (S7--S8)

The [seven-operator guide](seven_operator/README.md) describes the three
assessments and the CPU checks. The 18 registered cases share six task sources.
The repository includes per-case result reports and separate indexes for the
frozen inputs and complete raw archives.

## Visible-state composition (S6)

`composition/README.md` maps the original experiments, archived source files,
returned analyses and portable model/raw archive identities.
`composition/S6_EVIDENCE_AUDIT.json` checks the selected baseline weight
identities, the matched-dose endpoint counts and the complete seeded comparison.
