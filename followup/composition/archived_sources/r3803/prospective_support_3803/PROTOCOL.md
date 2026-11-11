# r3803: prospective topological curriculum test in a third hidden-rule world

This is the first confirmation attempt after the exploratory r3701 prefix
rescore. Use the previously fixed, nonisomorphic r1603 rules, but train a new
LoRA from the Qwen2.5-1.5B-Instruct base with hidden operator symbols only.
No rule text appears in prompts. Its 16,384 source examples contain at most
two distinct real operators and are inherited, with prompt text stripped,
from the frozen r1603 source bank. Train 512 updates of 32 examples with
rank-16 LoRA and the original source schedule. This source is eligible only
if the new probe gives minimum unary prefix score >= .95, minimum binary
prefix score >= .85, and no H2 at theta=.8. Report eligibility even if it
fails; no source checkpoint selection is allowed.

The ambient complex is fixed at five vertices, ten unordered edges, and ten
unordered triangular faces with F2 boundaries. Vertex probes use one-step
programs; edge items pair the two orders of a two-step program; face items
pair two seven/eight-step programs using the same three operators and a
shared root. Each cell has 128 disjoint items. The *primary* program score
is exact correctness of the first L output states, where L is the input
program length; extra output is ignored for this state-computation score.
The held-out five-operator transfer score separately requires the entire
trace and true EOS. Report both metrics on all probe programs. Apply the
paper's face-minimum closure and fixed theta grid .60:.05:.90. Freeze all
probe words, roots, and statistical criteria before querying the source.

Compare two 128-update curricula from that same eligible source. Each arm
sees 4,096 fresh seven/eight-step programs with exactly three real
operators, identical length counts, identical batch/order/optimizer and
the same per-operator triple incidence (A,B,C,D,E = 3,3,4,4,4 over six
equally represented triple types). The **shell** triple set is
{ACD, ACE, ADE, BCD, BCE, BDE}; its six faces include an H2 cycle when
their boundaries are present. The **control** set is
{ABC, ACD, ADE, BCE, BDE, CDE}; it has no H2 cycle with the same face count
and operator incidence. These sets were determined solely by exact F2
combinatorics before r1603 hidden-model outputs. Both arms train full trace
tokens. This contrast tests whether geometric closure matters beyond data
volume and operator exposure. Pairwise operator exposures are not exactly
matched, so post-training edge scores must remain within .05 in both arms
for a mechanistic interpretation.

Save and evaluate at updates 0, 32, 64, 96, and 128, without choosing a
checkpoint. Use 256 paired, step-essential seven-step and 256 paired
eight-step five-operator programs on new roots and states, disjoint from all
training and cell probes. The primary joint prediction is: in the shell arm
at step128 a new checkpoint-supported H2 direction appears on at least five
of seven thresholds, including three consecutive, with >=.95 item-cluster
bootstrap support; the control lacks that direction; shell strict transfer
is >=205/256 at both depths and exceeds control with positive paired
Bonferroni 95% lower bounds at both depths; both arms retain unary >=.95,
binary >=.85, format+EOS >=.98. Report all failures. The timing of the
topological event and transfer is reported at all five checkpoints, not
selected after seeing outcomes. This is one world; a fourth fresh world is
required before a multiworld claim.
