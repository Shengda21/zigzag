# R4201: prospective cross-world reversible intervention

Replicate the R4103 matched intervention without changing its outcome
definitions or thresholds in the nonisomorphic r4004 rule world. Start
all arms from the fixed R4007 update-32 adapter selected on development
and independently confirmed on the untouched R4006 bank at 240/256 and
231/256 strict five-operator pairs. Freeze a fresh, state-disjoint 25-cell
probe, 256 paired seven-step and 256 paired eight-step five-operator test
items, and two label variants of one 4,096-example continuation bank
before querying the selected adapter on the new probes.

Exactly 3,072 training examples have seven/eight-step programs using
three distinct operators; 1,024 are correct unary/binary anchors. The
clean variant has all correct states. The corrupted variant retains the
first three correct states but assigns later state strings from a
deranged equal-length donor. Prompts, input roots, programs, training
order, output lengths, and the multiset of late-state strings match.
No five-operator label enters training.

If the initial model passes the fresh-bank entry gate (minimum unary
and binary state-prefix scores at least .95/.85, H2 rank positive at
threshold .8, and strict five-operator pairs at least 205/256 at both
depths), train a clean control for 256 updates. Separately train 128
updates on corrupted labels and 128 clean recovery updates. All phases
use the same optimizer, learning-rate schedule, seed, and source adapter
lineage as R4103. Evaluate baseline, corrupt at 32/64/96/128, clean
control at total update 128, recovery at 32/64/96/128, and clean
control at total update 256.

The primary success rule is identical to R4103: corrupt endpoint H2
loss on at least five of seven thresholds including three consecutive,
with 1,000-resample item-cluster support at least .95; strict transfer
below 205/256 in at least one depth and below the paired clean control
with positive one-sided Bonferroni bootstrap lower bounds at both depths;
unary/binary minima still at least .95/.85. After recovery, a degree-two
new-checkpoint-supported class must reappear on the same grid with
bootstrap at least .95 and both strict scores must return to at least
205/256. Both clean-control endpoints must retain H2 and transfer.
Format+EOS is reported separately on every probe. Every failed gate is
retained. A pass links composition-consistent supervision to the joint
topological and behavioural response across two worlds; it does not
identify H2 as the mediator of the weight update.
