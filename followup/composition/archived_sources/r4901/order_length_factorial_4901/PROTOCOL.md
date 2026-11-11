# R4901: operator-variety by instruction-length evaluation

No training or checkpoint selection. Query the same fixed lower-order and
successful adapters used by R4401 in each of r1603 and r4004. Cross lengths
5 and 8 with exactly 2,3,4,5 distinct operators. Each condition has 256
paired programs, with a common root ID across the four variety conditions
at each length. The two members use the same subset and different order;
no immediate operator repetition is allowed. Subsets are cycled evenly.

Natural valid programs are sampled without requiring irreducible terminal
functions or distinct intermediate states, because long two-operator
programs can cycle. Report the number of effective state-changing steps
and distinct states, so this change from the earlier hard bank is explicit.
All roots and generated states are disjoint from prior training/probe
partitions. Prompt and output interfaces are identical across conditions.

Report full-trace strict EOS, requested-state-prefix accuracy, final-state
accuracy and first incorrect step, separately for all eight conditions,
both adapters, and both worlds. Pair success requires both members.
Within-length variety contrasts use common root IDs; 10,000 paired
bootstrap resamples report intervals. An absence-of-effect claim requires
simultaneous equivalence intervals for trained k=3,4,5 versus k=2 to fall
inside +/- .05 at each fixed length, with Bonferroni correction across the
six contrasts per world. Otherwise report the observed gaps without
claiming independence from operator variety. Record actual training length
distributions of the source and continuation banks.
