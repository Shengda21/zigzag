# R4401: single-call composition versus externally controlled one-step execution

Use the 256 frozen R4301 five-operator programs in each of the two rule
worlds; they were not used to select any adapter. For each world, fix a
lower-order source adapter trained on at most two distinct operators and
a high-capability adapter trained on three-operator traces. Query both
adapters once with the full program and the original output format.
Query the lower-order adapter additionally through an external controller:
at each instruction, provide its current predicted state and only the
next operator in a fresh one-step prompt, parse the first predicted
eight-digit state, and feed it to the next call. If the first state is
invalid, the controller stops and the entire program fails. Do not
replace an erroneous predicted state with the oracle state. This is a
strong baseline because the controller supplies the instruction pointer
and intermediate state, at seven or eight model calls rather than one.

Report for both worlds and depths: single-call full-trace strict accuracy
for lower-order and high-capability adapters; one-step controller full
state-trace accuracy using its first valid state; the stricter controller
rate requiring each step's output to contain exactly one state and EOS;
one-step primitive accuracy conditional on a correct current state; and
total prompt, generated tokens, calls, and wall time for each route.
The primary comparison is descriptive and uses the same fixed programs,
not a claim that the routes have equal inference cost. The lower-order
source's single-call failure and controller success, if observed, show
what the three-operator continuation internalized: long-program control
and trace emission without external state updates. All results, including
failures, are retained.
