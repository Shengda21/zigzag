# r3803 prospective third-world result

**The predeclared primary joint prediction failed.** The experiment used a
fresh hidden-symbol LoRA trained on at most two distinct operators in the
nonisomorphic r1603 world. The source satisfied its fixed eligibility gate:
minimum unary and binary paired state-prefix rates were 1.000 and 0.891,
and the 25-cell complex at threshold .8 had Betti ranks (1,6,0). It passed
only 15/256 seven-step and 1/256 eight-step held-out five-operator pairs.

The shell and control continuations each received 4,096 length-seven/eight
programs using exactly three real operators, 128 updates, the same
per-operator token totals, and 2,732 identical examples. The test bank was
frozen before the source model was queried. All source, cell-probe,
five-operator test, and continuation states were disjoint. Five-operator
test pairs required every step to matter, and their endpoints were not
reachable in four or fewer operations.

| Update | Shell H2 support | Shell 7 / 8 pairs | Control H2 support | Control 7 / 8 pairs |
|---:|---:|---:|---:|---:|
| 0 | absent | 15 / 1 | absent | 15 / 1 |
| 32 | .835 bootstrap | 206 / 121 | .532 bootstrap | 199 / 137 |
| 64 | 1.000 bootstrap | 231 / 198 | 1.000 bootstrap | 228 / 207 |
| 96 | 1.000 bootstrap | 238 / 208 | 1.000 bootstrap | 240 / 223 |
| 128 | 1.000 bootstrap | 237 / 216 | 1.000 bootstrap | 244 / 226 |

At step 128 both arms have all 25 cells at threshold .8 and Betti ranks
(1,0,4). The shell-minus-control paired strict differences are -7/256 and
-10/256 for seven and eight steps. The prespecified Bonferroni one-sided
bootstrap lower bounds are -0.0586 and -0.0742. The maximum edge-score gap
is .0234, satisfying the edge-matching criterion. The minimum unary and
binary scores pass the retention criteria in both arms. Format+EOS rates
over all 6,784 probes are .859 and .890, below the .98 gate.

The data establish genuine held-out five-operator transfer from long
three-operator supervision and a prospective H2 event in this world. They
do not support the claim that selecting an H2-bearing training face set
causes stronger transfer than the matched non-H2 face set. The event
appears in both arms by update 64, including the control arm that has
higher final transfer. The short-output stopping deficit is also real
and must be addressed as an output-interface issue in any final model.

Frozen bank manifest SHA-256:
`edb1032dc1754224b80b6283a48b33255e8772dec2c3d41064ac632f005dd95f`.
The source protocol's six-way allocation arithmetic correction was
recorded before any r1603 model query. The full cloud archive and all
intermediate checkpoint outputs are being copied to the local project.
