# E4-v4 result addendum

This directory contains the six-world confirmation results and the archive index.

| File | Role |
|---|---|
| `FEATURES.json` | Byte-preserved frozen confirmation feature export. |
| `REPORT.json` | Byte-preserved frozen confirmation analysis and predictions. |
| `SUPPLEMENT.json` | Formal six-world criterion summary derived from the frozen exports. |
| `PAIRED_FILES.jsonl` | Per-world, per-arm, per-step paired analysis records (204 rows). |
| `DATA_MANIFEST.json` | Portable record of the eight pilot/confirmation archive basenames, sizes and SHA-256 hashes; frozen lock, pilot selection and analyzer hashes; the four result-file hashes; and 272 raw evaluation source digests. |
| `rebuild_from_raw.py` | Portable CPU entry point for archive verification, raw rescoring and fixed-candidate refitting. |
| `RAW_RECONSTRUCTION_AUDIT.json` | Returned check of all 272 raw evaluation files and both frozen feature exports. |

The fixed step-512 external paired effect was +0.0716 (world-bootstrap 95% interval −0.1237 to +0.2832), so its predeclared effect criterion did not pass. The six primitive edge equivalence intervals passed. Neither fixed topology family met the predictive-utility criterion against both full-score controls. These are distinct criteria; the edge result does not rescue the failed external or utility criteria. See `SUPPLEMENT.json` for the exact estimates and `REPORT.json` for the frozen analysis.

The eight large raw run archives, generated inputs, checkpoints and model states are stored separately. `DATA_MANIFEST.json` identifies them by basename, byte count and hash. The formal summary uses the original frozen exports. With the archives available locally, the following separate CPU route reconstructs those exports from the saved predictions:

```sh
python followup/e4_v4/results/rebuild_from_raw.py --archive-root /path/to/archives --output raw_reconstruction_audit.json
```

The program verifies the archive and member bytes and uses the frozen scoring and feature code. All 1,810,432 saved predictions reconstruct all 68 pilot and 204 confirmation feature rows exactly. Refitting the six fixed pilot-selected candidates reproduces their reported predictions to a maximum absolute difference of $6.67\times10^{-16}$ in the recorded local environment. The paired effects and their pass/fail decisions remain unchanged. This portable check agrees with the independent strict-scoring and chain-space audit described in Supplement S5; it uses the frozen scorer itself. Neither route changes a training run, threshold or frozen export.

For byte checks, compare the four result files with `results` in `DATA_MANIFEST.json` using SHA-256. The archive hashes can be checked when the archives are made available separately. The frozen source and protocol are in the parent `e4_v4/` directory; the original-study `reproduce.py` command does not run this later experiment.
