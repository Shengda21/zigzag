# R4601: equal-dose, topology-selective late-state intervention

This is a development experiment in the fixed r1603 rule world. Both arms
start from the same R3803 control update-128 adapter. Their 4,096 prompts,
roots, operator words, training order, and answer lengths are identical.
Each bank has 2,520 seven/eight-step programs using exactly three distinct
operators and 1,576 correct unary/binary anchors. Exactly 1,008 long
examples per arm retain their first three true states and replace all
later states by independently sampled, length-matched eight-digit strings.
These replacement suffixes are
the same multiset in both arms. The two corrupted triple sets have equal
operator incidence and, by balanced word generation, exactly equal
corrupted operator-token counts. No five-operator example is trained.

The cycle-preserving arm keeps the six triples
`ACD ACE ADE BCD BCE BDE` clean. The acyclic arm keeps
`ABC ACD ADE BCE BDE CDE` clean. Their six clean face sets have,
respectively, one and zero second-homology dimensions when all their
boundaries are present. The corrupted complements are determined before
any model query. This manipulation tests whether the *arrangement* of
composition-consistent supervision matters at equal corruption dose.
It does not equate the identities of the clean triples.

The fixed, state-disjoint R4103 topology and hard-transfer banks are
reused for this development experiment. Query the common source and both
arms at updates 32, 64, 96, and 128 without checkpoint selection. The
cell score is paired exact state-prefix correctness, and the transfer
score is paired strict full-trace plus true EOS on 256 seven-step and 256
eight-step five-operator pairs. Use the face-minimum closure, F2 homology,
and threshold grid 0.60:0.05:0.90. The primary endpoint is update 128.

The joint hypothesis requires the cycle-preserving arm to have at least
one more H2 dimension at theta .8 than the acyclic arm; the positive
degree-two new-checkpoint-supported rank must hold on at least five
thresholds, including three consecutive, with 1,000 item-cluster
bootstrap support >= .95. The cycle arm must attain >=205/256 strict
pairs at each depth and have a positive one-sided Bonferroni 95% paired
bootstrap lower bound over the acyclic arm at *both* depths. Each arm must
retain minimum unary >=.95, minimum binary >=.85, and all-bank format+EOS
>=.98. Every clause is required. Report the result as a failure if any
clause fails, and retain all checkpoints and per-cell scores.

Even a positive development result would require replication in an
independent rule world and a fresh hard bank before a general mechanism
claim. It would identify the intervention on face-specific supervision,
not an internal parameter-space H2 mediator.
