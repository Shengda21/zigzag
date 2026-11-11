# E4-v4 frozen source and result addendum

This directory contains the frozen E4-v4 explicit-operand curriculum source and pilot decision records. The `results/` directory contains the six-world confirmation feature export, frozen report, formal summary, paired-file records and a portable data manifest. It contains no model state, raw per-item prediction file, or generated world input. The study compares coherent and independent *explicit* operands above shared reference practice; it does not isolate internal execution of an unprovided intermediate.

## Provenance and scope

- `CONFIRM_LOCK.json` has SHA-256 `98a95beb672d40cb9c4fdac6420ef1b8e9ee1d5eecb22c34f2d38d333f326c2f`. Every file listed by its `files` field has been checked byte for byte against its frozen hash.
- Frozen code, protocol, configuration, requirements and supporting pilot package files came from the already returned `e4_v4_world961.tar.gz` (SHA-256 `2f4a7a9a700097ae07e6d966049c64bb2ad2f711fd77d0558615037f8fcd5a7b`). The archive is stored outside this repository. The locked `PILOT_REVIEW.md`, `analysis/pilot/SELECTION.json`, and `FEASIBILITY.json` were taken from the returned pilot records because that per-world archive predates their creation; their bytes match the lock.
- `analysis/pilot/FEATURES.json` and `REPORT.json` are returned pilot exports. They are useful to reproduce the pilot-fitted predictors but were not individually listed in the lock; `SOURCE_MANIFEST.json` records their current hashes and their distinct provenance. The pilot report is development evidence, not a confirmation estimate.
- The `analyze.py` here is the **frozen** cloud version (SHA-256 `171127bf254ab9aa21a9c2647370cdc94bb0f1e8da5e562ed2be2d9461439b67`). A later local reporting draft has different bytes and is not substituted for it.

`SOURCE_MANIFEST.json` identifies each included original source file, its size, SHA-256, lock status, and source. `results/DATA_MANIFEST.json` identifies the eight raw archives by basename, byte size and SHA-256, plus 272 raw evaluation source digests without machine-specific paths. The archived `README.md` is a byte-preserved pilot-package file; this guide describes the six-world confirmation. `PACKAGE_MANIFEST.json` is also preserved from the pilot archive and includes expected hashes for the generated pilot inputs that are intentionally absent here.

## Reproduction inputs

To rerun training or independently rescore confirmation predictions, work from a copy with the complete pilot input package, the same Qwen2.5-1.5B-Instruct base weights and tokenizer, and the generated confirmation inputs. `run.py` checks `PACKAGE_MANIFEST.json`, each world's input manifest, the model identity and the confirmation lock before training; it will reject this directory until the referenced pilot input files are supplied. The eight raw world archives are stored separately and are absent from this addendum; `results/DATA_MANIFEST.json` gives their hashes but no public access location. The frozen cloud `analysis/confirm` feature and report exports are preserved byte for byte in `results/`. See `results/README.md` for the exact scope and access limitation.

The experiment fixes pilot worlds 961–962, confirmation worlds 971–976, two paired arms per world, seed index 1, 512 updates and 17 checkpoints. Learner selection uses only the two pilot worlds. For confirmation interpretation, see `PROTOCOL.md`: paired step-512 external exact accuracy, edge equivalence, and predictor comparisons against **both** full-score raw and raw-temporal controls are distinct criteria. Betti and temporal features here are not full zigzag intervals, historical unsupported quotient counts, or windowed cokernel ranks.

See the [results guide](results/README.md) for the CPU replay command and the confirmation outcomes.
