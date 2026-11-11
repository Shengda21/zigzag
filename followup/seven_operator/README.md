# Seven-operator interventions and prescribed connecting rank

The seven-operator study tests whether labelled boundaries identify which
held-out programs a trained interpreter can execute after a rule exchange.
The experiment uses two rule worlds with three initialization and data seeds
per world.

| Assessment | Cases passing | Evaluation bank |
| --- | ---: | --- |
| Primary confirmation | 6/6 | 256 roots per task |
| Fixed-dose confirmation | 6/6 | 256 roots per task |
| Disjoint program patterns | 6/6 | 128 roots per task |

ABCD succeeds in phase A and ABDE succeeds in phase B. Each scores zero
in the other phase. ACEF scores zero in both phases. Its boundary is present
in the union bridge beyond the sum of the two endpoint images. An earlier
checkpoint also supports it. The
[results summary](results/JOINED_RESULTS.json) gives the simultaneous bounds
and [case audit](results/FINAL_LABELLED_EVIDENCE_AUDIT.json) gives the counts.

## Verify the saved results on CPU

From the repository root:

```sh
python followup/seven_operator/verify_results.py
```

The command recomputes simultaneous score bounds and paired contrasts
from the saved success flags and reconstructs union and intersection
barcodes for the 12 measured checkpoint sequences. The six disjoint-pattern
cases evaluate the fixed-dose weights on new programs.

## CPU design checks

From the repository root, with Python 3 and the existing NumPy/SciPy stack:

```
python followup/seven_operator/cpu/shared_graph_rank_design.py
python followup/seven_operator/cpu/verify_cycle_support_counterexample.py
python followup/seven_operator/cpu/verify_labelled_bridge_nonidentifiability.py
```

The first checks twenty exact finite complexes, including rank 64 at 24
vertices, along with pair balance, labelled witnesses and connecting images.
The second compares the bare Pasch trade with the common missing fan:
a canonical boundary can be a sum of endpoint cycles despite being
incomplete at each endpoint. The third keeps the full unlabelled barcodes,
pair-incidence counts and a named boundary's face histories fixed while
changing that boundary's membership in the endpoint image sum. Original
output snapshots accompany the code.

After restoring case archives to a working root, measured diagrams can be
reconstructed without model inference:

```
python followup/seven_operator/cpu/reconstruct_measured_diagrams.py --study-root /root/simods_next --output measured_diagrams.json
```

This uses the copied exact chain implementation and lists every registered
primary/fixed-dose case, including missing complete results. The original
rendering script is also supplied; its historical default figure path
assumes the original workspace layout. The reconstruction wrapper writes
only the explicit output file.

## Frozen inputs, fitted weights, and raw outputs

`runtime/` contains verbatim frozen code, case/registration manifests and
protocols. `results/` holds small returned result snapshots. The [trained weights release](https://github.com/Shengda21/zigzag/releases/tag/trained-weights-v1) contains the fitted LoRA adapters
and controller weights. See the [weight manifest](https://github.com/Shengda21/zigzag/releases/download/trained-weights-v1/weight-manifest.json) for the
asset and path corresponding to each checkpoint. Separate archives hold the
large JSONL banks and raw token outputs.
`INPUT_ARCHIVES.json` and `RAW_ARCHIVES.json` identify their portable
filenames, sizes, digests and workspace-relative source locations. The
source-only tree needs these inputs restored before GPU training/inference.
Each full case archive itself contains its original data and execution code.
The original r5301 technical prefix and the recovered complete r5311 export
are separate artifacts for the same registered case r5301.

Restore archives using their original case directory names under
`/root/simods_next`. This preserves recorded parent-weight paths. For fresh
training, use an empty working root and the primary frozen-input archive
with its documented prompt dependency; for fixed-dose retraining, restore
the completed primary sources and extract only fresh fixed-dose inputs.
The suite `REPRODUCE.md` files give the exact commands and dependency repair.
Never start a new fit inside a directory containing completed outputs.

For analysis from restored outputs, first copy `runtime/` into the working
root and extract the full case archives there. Then rehash the separately
downloaded archives and install their portable verification metadata:

```
python followup/seven_operator/tools/verify_release_archives.py --index followup/seven_operator/RAW_ARCHIVES.json --archive-dir /path/to/archives --study-root /root/simods_next
python /root/simods_next/audit_confirmation_case.py --study /root/simods_next/confirmation_world1_seed1_5301
python /root/simods_next/confirmation_equal_dose_5400/audit_equal_dose_case.py --study /root/simods_next/equal_dose_world1_seed1_5401
python /root/simods_next/cloud_tools/report_confirmation_assessments.py
python /root/simods_next/cloud_tools/report_word_family_evaluation.py
```

Repeat the two raw-audit commands for every directory in their respective
registrations. New-pattern raw-token reconstruction uses each case's
`audit_outputs.py`, and its inheritance check is run by the last report
command. Complete archives contain the original analysis commands and logs.
The first verification command also covers retained development archives;
restore every indexed directory carrying a case audit before requesting
metadata installation, or verify individual files with `--filename`.

For an offline CPU replay of whole-program scoring, supply an indexed
archive and the two tokenizer files from the pinned base revision. The
wrapper checks their digests, restores the archive to an isolated temporary
directory, and runs the archived scorer there. It needs the recorded
controller weights included in the case archive.

```
python followup/seven_operator/tools/replay_archived_whole_programs.py --index followup/seven_operator/RAW_ARCHIVES.json --archive-dir /path/to/archives --archive r5304_full.tar.gz --base-snapshot followup/seven_operator/BASE_MODEL_SNAPSHOT.json --tokenizer-dir /path/to/tokenizer --output whole_program_replay.json
```

`--archive` can be repeated. This replay reconstructs trajectories and paired scores from raw tokens.
It checks logical query counts and new-call counts for non-inherited traces. The separate
case audit above checks training lineage and component qualification.

The word-pattern evaluation inherits the exact seven final weights from
each fixed-dose case. Its 128-root banks exclude all original four/five
program-pattern orbits. It evaluates new patterns at length 8 with unequal
adjacent symbols. It has one separately allocated simultaneous 95% family,
252 scores and 48 paired differences; it adds no training initialization.
`tools/audit_word_family_inheritance.py` cross-checks runtime weight hashes
against the archived fixed-dose assemblies after return.

After restoring the case summaries and archived flags, independently
check simultaneous bounds with `tools/audit_labelled_sampling_bounds.py`
using `--study-root` and `--output`. It directly sums binomial tails
and reconstructs every paired contrast without the frozen interval code.

The three assessments share six task sources (two fixed rule worlds,
three initialization/data seeds each). Separate confidence families are
reported separately. Counts and confidence bounds condition on the fitted
models and frozen generator, rather than a population of training seeds.

## Base model and environment

The numeric backbone is Qwen/Qwen2.5-1.5B-Instruct, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`. Its weight digest is
`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`.
Download the [base weights from Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) and point
`SIMODS_BASE_MODEL` to that snapshot. The case archives also retain the
learned task adapters.
The observed environment is Python 3.12.3, PyTorch 2.9.1+cu128,
Transformers 4.57.6, PEFT 0.21.2, NumPy 2.1.3, SciPy 1.18.1, and Accelerate
1.15.0 on an RTX 6000D with 84 GB of memory. Per-case environment records are authoritative.

The attention controller and primitive/residual execution factorization
are explicit testbed structure. The single-call failures and failed data
repairs remain in the external development archives. Passing this
interpreter's criteria is distinct from repairing those single-call models.
The manuscript's method contribution concerns the matched intervention
design, labelled support and certified bridge diagnostics.
