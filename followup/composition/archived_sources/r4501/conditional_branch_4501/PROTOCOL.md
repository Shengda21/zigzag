# R4501: branch-flipping counterfactual at a conditional operator

R4301 showed that two fixed adapters follow a randomly substituted
visible state. This stricter test isolates the conditional rule in each
world. Before any query, freeze 256 fresh five-operator programs per
world, half seven and half eight instructions. In each length, 64 items
have an active conditional swap D at the intervention boundary and 64
have an inactive one. The boundary is after instruction three or four.
The true current state's first two control digits must differ, and the
two coordinates potentially swapped by D must differ.

For each item, compare a true partial assistant trace with a second
partial trace in which only the two current control digits are exchanged.
This flips the predicate `first digit > second digit` while keeping the
conditional operator and its two candidate swap values unchanged.
The input root, all program symbols, earlier printed states, output
syntax, and decoding are identical. The alternative is a valid,
previously unseen state; the true and edited next states, and the
remaining reference suffixes, are computed independently from the
rule world. All roots, program words, and generated states are disjoint
from prior training and probe banks where the fixed models were used.

The fixed adapters are R3803 control update 128 for r1603 and R4007
stop-calibrated update 32 for r4004. Neither is tuned on this experiment.
The primary gate requires at least 90% correct unedited immediate
successors and 75% correct edited immediate successors per world, a
positive paired 2.5th-percentile bootstrap lower bound for edited versus
original successor under the edited prefix, and at least 70% edited
successors separately in each branch direction. Full remaining-trace
strict success, syntax, and EOS are secondary. Failure of any fixed gate
precludes a branch-specific trace-use claim.
