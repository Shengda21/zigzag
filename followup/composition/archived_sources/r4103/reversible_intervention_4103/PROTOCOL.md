# r4103: reversible composition-supervision intervention

Start all arms from the same r3803 control adapter at update 128, which
already transfers to held-out five-operator programs and supports H2.
Freeze a fresh state-disjoint 25-cell probe, 256 paired seven-step and
256 paired eight-step five-operator test items, and two label variants of
one 4,096-example continuation bank before querying this adapter on the
new probes. Exactly 3,072 training examples contain seven/eight-step
programs with three distinct operators; 1,024 are balanced unary/binary
anchors. The clean variant has all correct states. The corrupted variant
retains the first three correct states of each long program but assigns
its later states from a deranged same-length donor. All prompts, roots,
programs, example order, output lengths, and the multiset of later-state
strings are identical between arms. Anchors remain correct in both.

From the common adapter, train a clean control for 256 updates on the
clean bank. Separately train 128 updates on the corrupted bank, then
128 recovery updates on the clean bank. Use the same optimizer and learning
rate schedule in each 128-update phase and the same random seed. Evaluate
the initial model, corrupt phase at updates 32,64,96,128, recovery phase
at 32,64,96,128, and clean control at total updates 128 and 256. No
five-operator labels occur in any training phase.

Primary intervention prediction: the initial model has a robust H2
support complex and >=205/256 strict five-operator pairs at both depths;
at corrupt update 128 H2 is lost on the fixed threshold grid, five-operator
transfer falls below 205/256 in at least one depth and below the matched
clean control with positive paired Bonferroni lower bounds, while minimum
unary and binary state-prefix scores remain >=.95 and >=.85. After clean
recovery, an H2 support direction reappears on at least five of seven
thresholds including three consecutive with >=.95 item-cluster bootstrap
support, and both strict five-operator pair scores return to >=205/256.
The clean control must retain H2 and both transfer gates. Format+EOS is
reported on every probe; it is an independent output-interface outcome.
All failures are reported. This intervention tests whether composition-
consistent later-state supervision jointly controls the structural
readout and independent high-order transfer; it cannot by itself prove
that the H2 class mediates the update.
