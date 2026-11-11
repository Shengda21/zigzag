# R4201: cross-world reversible composition-supervision intervention

The R4103 intervention was repeated, without changing its thresholds or
outcomes, in the nonisomorphic r4004 rule world. The common adapter is
the R4007 update-32 checkpoint selected on development and confirmed on
an untouched bank. The R4201 banks and matched training variants were
frozen before querying that adapter (manifest SHA-256
`fceaabc30fcef33fcba75138b6a2d0d43915a6e335d54e6924645c50e3510295`).
The fresh baseline passed the entry gate: H2 rank four at threshold .8,
minimum unary/binary state-prefix 1.000/.984, and strict held-out
five-operator pair scores 251/256 and 232/256.

All arms use the same 4,096 prompts, roots, programs, example order,
target lengths, and late-state string multiset. The corrupted variant
keeps the first three correct states in each of 3,072 three-operator
long examples but deranges their later states among equal-length
examples; 1,024 unary/binary anchors remain correct in both variants.
No five-operator label is used during intervention or recovery.

| Arm and update | Minimum unary | Minimum binary | H2 rank at threshold .8 | Strict five-operator pairs, depth 7 | Depth 8 | Format plus EOS |
|---|---:|---:|---:|---:|---:|---:|
| Common start, 0 | 1.000 | .984 | 4 | 251/256 | 232/256 | 1.000 |
| Corrupted, 32 | .992 | .727 | 0 | 0/256 | 0/256 | 1.000 |
| Corrupted, 64 | 1.000 | .891 | 0 | 0/256 | 0/256 | 1.000 |
| Corrupted, 96 | .984 | .938 | 0 | 0/256 | 0/256 | 1.000 |
| Corrupted, 128 | 1.000 | .945 | 0 | 0/256 | 0/256 | 1.000 |
| Clean control, 128 | 1.000 | .992 | 4 | 252/256 | 240/256 | 1.000 |
| Recovery, 32 | 1.000 | .969 | 4 | 234/256 | 199/256 | 1.000 |
| Recovery, 64 | .992 | .977 | 4 | 240/256 | 232/256 | 1.000 |
| Recovery, 96 | 1.000 | .977 | 4 | 249/256 | 235/256 | 1.000 |
| Recovery, 128 | 1.000 | .977 | 4 | 246/256 | 230/256 | 1.000 |
| Clean control, 256 | 1.000 | .992 | 4 | 249/256 | 242/256 | 1.000 |

The predeclared five-of-seven, three-consecutive threshold-grid H2 loss
and corrupt-to-recovery support return each have 1.000 item-cluster
bootstrap support. Clean-update-128 minus corrupt-update-128 strict
paired differences are 252/256 at depth seven and 240/256 at depth
eight. Their one-sided Bonferroni lower bounds are 248/256 and 232/256.
All 16 predeclared gates pass. At the corrupt endpoint, format+EOS is
6,782/6,784, so formatting does not account for the absent five-operator
strict transfer. Both clean endpoints retain structural support and
strict transfer.

Together with R4103, the result replicates a reversible response of
the task-defined degree-two support complex and unseen five-operator
behavior to composition-consistent later-state supervision across two
different operator algebras. The intervention identifies the causal
effect of that supervision on two measured outcomes. It does not
identify H2 as an internal mediator. R4006 showed why the independent
strict and semantic outcomes matter: H2 and correct state prefixes can
coexist with a learned stopping error.

`AGGREGATE.json` contains the complete gate table, grid flags, bootstrap
supports, and paired contrasts. The eleven `probe_*` folders retain all
individual generations and independent scores. The compact raw-probe
archive `probes_all.tar.gz` has SHA-256
`64e95b5b667ce56ad1f61b09d30238c3b4cab9ce7c521f7fe64f0d61ab7a6db3`.
The full cloud run, including adapter checkpoints, is archived separately.
