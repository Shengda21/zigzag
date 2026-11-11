# r4006 development extension for the failed eighth step

The untouched r4005 confirmation failed its joint target: 219/256 strict
seven-step pairs but only 171/256 eight-step pairs, despite robust H2.
This is a new development intervention, not a reclassification of r4005.
Start from its fixed update-128 control adapter and train 128 further
updates on 4,096 fresh, state-disjoint programs. Of these, 3,072 use
exactly three distinct real operators and length nine or ten (1,536
each), distributed across all ten unordered triples; 1,024 are correct
unary/binary anchors. No five-operator labels are used. Longer low-variety
programs are intended to strengthen repeated state-transition use at
the previously weak eighth step.

Save and score updates 32,64,96,128 on the previously queried r4005 bank
for development. Select the earliest checkpoint with strict five-
operator pairs >=205/256 at both depths, unary >=.95, binary >=.85,
format+EOS >=.98, and robust H2 grid support >=.95. If none passes,
report this extension as failed. A selected adapter must then pass all
these gates on a new, untouched, state-disjoint r4006 confirmation bank
frozen before querying the selected adapter. The earlier r4005 result
remains a failure regardless of this experiment's outcome.
