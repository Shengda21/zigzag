# R4103: reversible intervention on composition-consistent supervision

## Design and frozen decision rule

All arms begin with the same update-128 R3803 control adapter. A fresh,
state-disjoint probe bank and the clean and corrupted variants of one
4,096-example training bank were frozen before querying that adapter on
these probes (manifest SHA-256
`f0326e6fe05a7d5ea7efd5b44de07f3824f13eec3cedc102963a222bbf676054`).
The training bank has 3,072 seven/eight-step programs using exactly three
operators and 1,024 correct unary/binary anchors. The corrupted variant
keeps the first three correct states and permutes the later state strings
among equal-length examples. Prompts, input roots, operation sequences,
training order, target lengths, and the multiset of late-state strings are
matched. No five-operator program is used for training.

The fixed analysis requires robust loss of degree-two support and held-out
five-operator transfer after 128 corrupted updates while unary and binary
prefix scores remain at least .95 and .85. It requires a positive paired
Bonferroni lower bound for the clean-control advantage at each depth. The
same support and transfer criteria must recover after 128 clean updates;
clean continuations from the common start must retain both. The exact
threshold grid, item-cluster bootstrap, and test code are in
`ANALYSIS_SPEC_BEFORE_INTERVENTION_PROBES.md` and `aggregate.py`.

## Results

| Arm and update | Minimum unary | Minimum binary | H2 rank at threshold .8 | Strict five-operator pairs, depth 7 | Depth 8 | Format plus EOS |
|---|---:|---:|---:|---:|---:|---:|
| Common start, 0 | 1.000 | .969 | 4 | 234/256 | 223/256 | .891 |
| Corrupted, 32 | 1.000 | .727 | 0 | 0/256 | 0/256 | 1.000 |
| Corrupted, 64 | 1.000 | .938 | 0 | 0/256 | 0/256 | 1.000 |
| Corrupted, 96 | 1.000 | .961 | 0 | 0/256 | 0/256 | 1.000 |
| Corrupted, 128 | 1.000 | .969 | 0 | 0/256 | 0/256 | 1.000 |
| Clean control, 128 | 1.000 | .984 | 4 | 248/256 | 242/256 | 1.000 |
| Recovery, 32 | 1.000 | .984 | 4 | 238/256 | 215/256 | 1.000 |
| Recovery, 64 | 1.000 | .992 | 4 | 246/256 | 233/256 | 1.000 |
| Recovery, 96 | 1.000 | .984 | 4 | 246/256 | 244/256 | 1.000 |
| Recovery, 128 | 1.000 | .984 | 4 | 246/256 | 245/256 | 1.000 |
| Clean control, 256 | 1.000 | .984 | 4 | 250/256 | 249/256 | 1.000 |

The loss event is present at the required five of seven thresholds with
three consecutive thresholds, and its 1,000-resample item-cluster support
is 1.000. The corrupt-to-recovery bridge supports a new degree-two
checkpoint class on the same grid, also with bootstrap support 1.000.
At depth 7, the paired clean-control minus corrupted difference is
248/256, with a predeclared Bonferroni lower bound of 242/256. At depth 8,
it is 242/256, with lower bound 235/256. All 16 predeclared gates pass.

## Scientific reading

The matched intervention establishes that consistency of later-state
supervision controls both the topological organization of lower-order
behaviours and transfer to unseen five-operator programs in this world.
The restoration after clean supervision shows that this is reversible
within the same adapter lineage. The unchanged unary/binary anchors and
retained endpoint scores isolate the effect from wholesale loss of the
trained individual operators. The constant full-format score after the
first 32 updates excludes output formatting and stopping as an explanation
for the zero high-order strict score in the corrupted endpoint.

This intervention changes training labels, and therefore does not by
itself identify the observed H2 class as a mediator of generalization.
Its causal target is composition-consistent later-state supervision; H2
and held-out five-operator transfer are two independently measured
outcomes. R3803 had already shown that an equally effective matched
curriculum can produce H2 and transfer without the proposed explicit
shell, so a shell-specific optimization claim is unsupported. R4005
produced H2 before meeting the prespecified eighth-step transfer target;
topology alone is not sufficient. These results define the next precise
mechanistic question: which part of composition-consistent supervision
is necessary for transfer, and does its loss predict the support change
across a new rule world?

## Reproduction record

`AGGREGATE.json` contains the complete gate values and checkpoint table.
Each `probe_*/predictions_*.jsonl` retains per-program outputs;
`probe_*/analysis.json` retains the independent per-item scoring. The
small probe archive `probes_all.tar.gz` has SHA-256
`95d7c49c3c64ab459309854a2892e2c975bad4fc2f62125b311458395d60e522`.
The complete cloud archive, including model adapters and run logs, is
downloaded separately.
