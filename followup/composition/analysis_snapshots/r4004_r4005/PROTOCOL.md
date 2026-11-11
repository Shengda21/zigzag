# r4004: fourth nonisomorphic hidden-rule confirmation world

This world is fixed before training. It uses five hidden symbols and a
different operator-order multiset (2,2,3,8,8) from r1601, r1602, and
r1603. The model is initialized afresh from Qwen2.5-1.5B-Instruct and
receives the same 16,384 lower-order training-program schedule as r3803,
with all answers recomputed from these new rules. No rule descriptions
appear in prompts and no source example contains more than two distinct
operators. Train 512 updates of 32 with the same rank-16 LoRA settings.

Before querying this source, freeze a new 25-cell bank, 256 paired
seven-step and 256 paired eight-step five-operator test items, and a
4,096-example three-operator continuation bank. All data partitions must
have disjoint roots and intermediate states. The cell score is exact
state-prefix correctness for the input program length; transfer uses
exact full trace and true EOS. The fixed theta grid, face closure,
union-bridge H2 direction, and 1000 item-cluster bootstrap are the same
as r3803.

Predeclare source eligibility: minimum unary >= .95, minimum binary
>= .85, no H2 at theta=.8, and fewer than 205/256 strict five-operator
pairs in at least one depth. The positive confirmation prediction after
128 continuation updates is a new H2 support direction on at least five
of seven thresholds including three consecutive, bootstrap >= .95;
strict five-operator pair counts >=205/256 at both depths; minimum unary
>=.95 and binary >=.85. Report format+EOS separately and require >=.98
for an output-interface-complete model. No source checkpoint selection
or post hoc task-bank changes are allowed. This experiment independently
tests generality of the structural readout; it does not substitute for a
causal intervention on the support direction.
