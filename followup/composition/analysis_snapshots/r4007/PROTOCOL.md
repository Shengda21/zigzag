# R4007: length-balanced state-trace stop calibration

R4006 failed its strict eighth-step gate on the r4005 development bank.
An independently scored error decomposition made before constructing this
intervention found 495/512 correct eight-step state prefixes at its final
checkpoint, but 119 of these appended one extra state. Seven-step prefixes
were correct in 504/512 programs, with 34 extra states. EOS and state
syntax were otherwise correct. Thus this experiment targets the learned
relation between instruction count and trace length while retaining the
compositional state transitions.

Continue the fixed r4006 update-128 adapter for 128 updates on 4,096 fresh,
state-disjoint examples, none using five distinct operators: 1,024 each of
lengths seven and eight, 512 each of lengths nine and ten, all with exactly
three distinct operators, plus 512 unary and 512 binary anchors. Use the
same prompt, exact trace targets, optimizer, and learning-rate schedule.
Evaluate steps 32, 64, 96, and 128 on the previously queried r4005 bank.
Select the earliest checkpoint having at least 205/256 strict pairs at
both depths, minimum unary and binary state-prefix scores at least .95
and .85, all-bank format+EOS at least .98, and degree-two new-checkpoint
support on the established threshold grid with bootstrap at least .95.
If none passes, the intervention fails and the fresh bank is not queried.

If a checkpoint passes development, evaluate that checkpoint and the fixed
r4006 source on the untouched r4006 confirmation bank, which was frozen
before any r4006 model query. The selected checkpoint must pass all the
same absolute gates on this bank. Its strict-pair advantage over the
r4006 source must have a positive 2.5th-percentile paired bootstrap lower
bound at each depth (10,000 pair resamples). This final comparison tests
whether calibration transfers to new programs and states, rather than
only to the development bank. The r4005 and r4006 failures remain recorded.
