# R4701: single-face separability gate (development)

The fixed source is r1603 R3803 control update 128. Ten independent
continuations start from exactly this adapter, one for each unordered
operator triple. This is a gate for selective forgetting, not a
confirmation of capability emergence or a test of a topology curriculum.

Each continuation has 4,096 examples and 128 updates: 1,008 seven/eight
instruction traces using the target triple, 224 clean traces for each of
the other nine triples (2,016 total), and 1,072 correct unary/binary
anchors. Only the target triple's states after the first three are
permuted among equal-length target examples. The target dose is chosen
before model queries, using the earlier R4601 observation that 1,008
corrupted long examples can cause a substantial effect. All arms use the
same optimizer and update schedule. No four/five-operator label is trained.

A new 25-cell bank has 128 items per cell, state-prefix scoring, face
closure, and the existing seven thresholds. Each checkpoint at updates
32,64,96,128 is queried on this diagnostic bank. For a target, select the
earliest checkpoint with raw target-face score <.60, every other raw face
score >=.80, minimum unary >=.95 and binary >=.85, and diagnostic syntax
plus true EOS >=.98. Selection never uses four-operator transfer labels.
No passing checkpoint means failure to realize selective manipulation at
this fixed dose and schedule; it is not proof of universal inseparability.

A separate bank has 128 paired length-eight programs per four-operator
subset. Pair IDs share an input root across all five subsets. Each word
uses its four operators, has no immediate repetition, has every step
essential to its final state, and has a final state unreachable in three
or fewer operations. Reference traces and all states are disjoint from
training and diagnostic partitions. Strict success requires both members'
entire state traces and the true EOS. Query the selected checkpoint, or
the fixed update128 if no manipulation checkpoint passes. The latter is
descriptive and cannot pass the joint gate.

The two four-operator subsets containing the target face have incomplete
boundaries, and the other three have complete boundaries if manipulation
passes. The primary per-arm contrast is mean strict pair accuracy of the
three complete-boundary subsets minus that of the two incomplete-boundary
subsets. Resample the 128 common root IDs, preserving all five subsets,
10,000 times. Report the gap, its 95% percentile interval, and a one-sided
Bonferroni lower bound across ten target faces (0.5th percentile).

The gate passes only if at least eight of ten targets realize the
manipulation, have contrast >=.30, and have positive Bonferroni lower
bound. The 10x10 matrix of source-minus-intervened face scores is reported
in full. Report off-target effects separately for faces sharing two or
one operators with the target. No training seed is inferred from item
bootstrap. A passing gate leads to a separately frozen exchange study in
two worlds with at least three training seeds each; failure blocks that
study until a new, explicitly versioned manipulation is justified.
