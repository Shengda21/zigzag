# R4301: counterfactual continuation through a visible intermediate state

This read-only intervention probes how two fixed successful adapters use
their generated state traces: R3803 r1603 control update 128 and R4007
r4004 stop-calibrated update 32. Neither model is selected or trained on
this experiment. The two rule worlds have different operator-order
multisets and conditional operations.

Before any R4301 model query, freeze for each world 128 seven-step and
128 eight-step programs using all five operators, with unseen roots,
unseen intermediate states, no immediate repeated operation, no final
state reachable in four or fewer operations, and no single instruction
deletion preserving the final state. At an instruction boundary after
step three or four, construct two assistant prefixes with the same
prompt and the same previous printed states. One prefix includes the
correct current state; the other replaces only its eight digits with a
random valid state at Hamming distance at least five. The replacement's
next-state successor must differ from the original next state and be a
genuine transition. Append the same ` | ` delimiter to both prefixes.

Generate the remaining trace greedily from each prefix with the same
token cap and EOS rule. Score the first newly generated state against
the relevant rule-world successor: original successor for the unedited
prefix, counterfactual successor for the edited prefix. Also record
whether the edited output follows the original successor, the full
remaining trace prefix, strict exact continuation with EOS, syntax,
and stopping length. If the model merely recomputes from the unchanged
input and program, its edited continuation should prefer the original
successor. If it uses the visible intermediate state, it should prefer
the edited successor. All item-level outcomes and paired confidence
intervals are reported separately for both worlds and both depths.

The primary cross-world trace-use gate requires, in each world, at least
90% unedited first-successor accuracy, at least 75% edited first-successor
accuracy, and a positive 2.5th-percentile paired bootstrap lower bound
for edited-successor accuracy minus original-successor accuracy under
the edited prefix. The 10,000 resamples use program IDs as paired units.
The exact full remaining trace is secondary and does not redefine the
first-successor outcome. A failure blocks the claim that the displayed
trace acts as an execution state, even if held-out five-operator exact
matches are high.
