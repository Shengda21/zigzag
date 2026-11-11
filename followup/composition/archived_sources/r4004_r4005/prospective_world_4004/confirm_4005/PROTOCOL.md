# r4005: untouched confirmation bank for repaired r4004 source

The r4004 first source failed its frozen development eligibility. A new
lower-order-only repair bank was frozen before repair updates. This bank is
generated before inspecting any repair-model output. Its roots and every
intermediate state are disjoint from the original source, repair data,
previously queried r4004 development probes, both development continuation
banks, and the earlier r1603 probes.

After all repair checkpoints are scored on the old development probes,
select the earliest of updates 32, 64, 96, 128 satisfying unary minimum
>=.95, binary minimum >=.85, no H2 at theta=.8, and fewer than 205/256
strict five-operator pairs in at least one depth. The selected adapter
must independently pass those same conditions on this untouched bank.
If no checkpoint passes, stop this confirmation attempt as ineligible.

From the eligible selected adapter, train exactly 128 updates of 32 on
the fresh `control_train.jsonl` bank in this directory. It contains
seven/eight-step programs with exactly three distinct operators and no
five-operator labels. Save at 0, 32, 64, 96, 128 and query both frozen
banks without checkpoint selection. The positive prediction at update
128 is a new checkpoint-supported H2 direction on at least five of the
seven theta=.60:.05:.90 thresholds including three consecutive, with
item-cluster bootstrap >=.95; strict five-operator pair scores >=205/256
at both depths; unary minimum >=.95 and binary minimum >=.85. Report
format+EOS on all 6,784 programs, requiring >=.98 for an
output-interface-complete model. This confirmation does not assert that
H2 itself mediates the effect of training.
