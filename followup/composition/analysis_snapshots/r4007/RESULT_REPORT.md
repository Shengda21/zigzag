# R4007: prospective fourth-world composition and stop calibration

R4005 first failed its eighth-step strict transfer gate, scoring 171/256
despite robust H2. R4006 then trained on longer three-operator traces and
improved the held-out eight-step **state-prefix pair** score to 239/256,
but its strict pair score fell to 133/256 because it appended an extra
state to many otherwise correct eight-step traces. The R4006 failure and
its untouched confirmation bank were retained.

Before querying that bank, R4007 fixed a length-balanced continuation:
from the R4006 update-128 adapter, train on 4,096 fresh examples with
exactly three distinct operators at lengths seven through ten, plus
unary/binary anchors. No five-operator label is used. The earliest
development checkpoint satisfying all fixed criteria was update 32:
249/256 and 236/256 strict pairs, minimum unary/binary 1.000/.992, all-bank
format+EOS 1.000, and H2 grid bootstrap 1.000. Checkpoint selection was
fixed before any confirmation query.

The untouched R4006 bank was then queried with the original repaired
lower-order source, the fixed R4006 adapter, and the selected R4007
adapter. Its bank manifest SHA-256 is
`791d603349008070c8c38d873f9fb70780edb688e4d79449e252b3adc080cb2b`.

| Fixed adapter on the new bank | Minimum unary | Minimum binary | Strict five-operator pairs, depth 7 | Depth 8 | All-bank format+EOS | H2 grid bootstrap |
|---|---:|---:|---:|---:|---:|---:|
| Lower-order source | 1.000 | .984 | 0/256 | 1/256 | 1.000 | absent |
| R4006 before calibration | 1.000 | .992 | 217/256 | 157/256 | 1.000 | 1.000 |
| R4007 calibrated, selected +32 | 1.000 | .984 | 240/256 | 231/256 | 1.000 | 1.000 |

The selected-minus-R4006 paired differences are 23/256 at depth seven
and 74/256 at depth eight. The prespecified 10,000-resample paired
Bonferroni lower bounds are 12/256 and 58/256. The lower-order source has
no H2 at threshold .8, whereas the selected model has rank four. Its
new-checkpoint-supported degree-two event passes the five-of-seven
threshold grid with item-cluster bootstrap 1.000. **All ten frozen
confirmation gates pass.** The selected adapter also retains format+EOS
1.000 and the unary/binary entry gates.

This is evidence of held-out five-operator trace execution in a fourth
nonisomorphic rule world after training without five-operator labels.
The long-trace extension and stop calibration play different empirical
roles: R4006 improves state transitions, while R4007 restores the
instruction-count stopping rule under the strict outcome. The result
does not retroactively turn the R4005 or R4006 fixed-gate failures into
successes. Nor does it establish that H2 causes transfer: R4006 already
had H2 while its strict eighth-step performance was weak. The new-world
confirmation instead tests the conjunction of an independently measured
structural event and high-order behavior, as in the original theory.

`AGGREGATE_CONFIRM.json` contains each gate, paired comparison, and raw
score. The `probe_confirm_*` folders contain every model output and its
independent item-level analysis. The R4007 training manifest SHA-256 is
`11b4ebe0902b1dbe10099b2b1d03ce4243fb880d7ae37019dd0dcbeb6eb122f7`.
