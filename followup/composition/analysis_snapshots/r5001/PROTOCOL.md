# R5001: minimum sufficient training variety, paired independent training seeds

Question: is three-operator trace practice necessary for the observed
one-call execution of unseen five-operator combinations, or is matched
long two-operator practice sufficient?

Use the fixed r1603 and r4004 operator worlds, Qwen2.5-1.5B-Instruct and
rank16 LoRA. For each world train three independent LoRA initializations
from the same base for 512 lower-order updates on the existing 16,384
source records. All three sources, including any poor source, are retained.
From each source start two matched 128-update continuations. The two arms
share example order, paired roots, lengths, optimizer, total examples and
per-operator token counts. Each has 4,000 long traces (2,000 at length7,
2,000 at length8) and 96 identical unary/binary anchors. Long programs use
exactly two or three distinct operators respectively. There are no
four/five-operator labels in training. Tokenizer lengths are checked before
GPU training. Source and continuation orders vary with the training seed;
the paired arms share the seed and source weights.

The two sources and endpoints are evaluated on fresh model-blind 25-cell
banks (128 items per cell) and 256 paired essential five-operator programs
at each of depths7/8. All probe states are disjoint from source and
continuation training states. Cell scores use requested-state prefixes;
the independent primary outcome requires both programs' full traces and
the true EOS. Evaluate only the fixed source512 and endpoint128; no
checkpoint selection is used.

The constructive sufficiency criterion for the two-operator arm requires
in every one of six world/seed runs: >=205/256 strict pairs at both depths,
minimum unary >=.95, minimum binary >=.85, and overall syntax plus EOS
>=.98. Report all outcomes, seed scatter, H2 ranks and paired three-minus-
two contrasts. Paired item-bootstrap intervals use the same pair IDs
within each world/seed, with 10,000 resamples and Bonferroni one-sided lower
bounds across twelve world/seed/depth contrasts. These intervals condition
on a trained model and do not replace the three training seeds. A negative
contrast does not itself prove population equivalence; noninferiority is
reported only as a separately stated margin comparison. Failure of source
or endpoint criteria is retained without dropping a seed.

This is an independent training-seed test of the minimum sufficient order
in two previously used rule worlds. It is not a new-rule-world confirmation
or a level(iii) capability-emergence test. The task probes execution on
unseen operator sets at lengths already present in continuation training.
