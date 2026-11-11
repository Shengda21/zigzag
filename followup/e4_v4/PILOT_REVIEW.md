# E4-v4 pilot review before confirmation

Decision: GO

Recorded 2026-11-08 after all four arms in worlds 961 and 962 finished 512 updates, all 17 evaluations, and wrote `COMPLETE.json`. This GO authorizes only the unchanged, reserved six-world confirmation specified in `PROTOCOL.md`; it is not a finding of positive treatment effect or topology utility. No confirmation data or result had been generated at this review.

## Fixed feasibility gate

`analysis/pilot/FEASIBILITY.json` reports `passed=true`, and an independent audit recomputed the gate. Both worlds have zero actual-Qwen per-record token mismatches; paired arms each use 5,134,336 input tokens and 950,272 answer tokens. Across all four final arms, every edge strict-format rate is 1.0 and the minimum individual edge exact rate is 0.99609375. Mixed-control exact is at least 0.99609375; reference-control exact is 1.0. External strict-format rates are 0.998046875, 0.998046875, 1.0, and 1.0. All four external exact values are inside (0.01,0.99), satisfying the prespecified nondegeneracy rule.

| World | Independent external exact | Coherent external exact | Coherent minus independent |
|---|---:|---:|---:|
| 961 | 0.41015625 | 0.05859375 | -0.3515625 |
| 962 | 0.04296875 | 0.31640625 | +0.2734375 |

The two-world paired mean is -0.0390625, with descriptive world-bootstrap interval [-0.3515625, 0.2734375]. The directions conflict, and the protocol did not require a favorable pilot effect. This is why independent-world confirmation is necessary. No treatment claim follows from this two-world pilot.

## Measurement sensitivity and limits

The external targets vary across checkpoints and arms. Each arm/world has four to six distinct Betti feature vectors and five to six distinct cell masks; temporal feature vectors also vary. All thresholded complexes reach their final masks by step 256 in world 961 and step 224 in world 962; final Betti vectors are identical across the four arms. Confirmation must report that late-stage topology saturation and cannot imply that final Betti values explain the external differences.

The cloud package's pilot REPORT gives in-sample MSE 0.0182780 for the temporal family, 0.0198337 for raw, and 0.0196666 for raw_temporal. Its leave-one-pilot-world-out selection error for temporal is 0.0212329, versus 0.0208276 for raw. The apparent in-sample temporal advantage is development evidence only. The packaged REPORT reports paired gains only versus raw_temporal; a pre-endpoint local reporting correction computes the protocol-required comparison versus both raw and raw_temporal without changing features, selection, checkpoint, or learner. The frozen cloud package remains the source of the gate and selected models; complete both-control reporting will be supplied as a supplemental computation from the fixed predictions, with its code provenance disclosed.

The study estimates the effect of additional coherent explicit-operand examples above shared reference rehearsal. A positive confirmation effect would not isolate internally executed composition. The topology feature families are deterministic compressions of raw scores; any confirmed predictive gain would be a finite-sample inductive-bias result. Worlds are codebook replicates under one fixed optimizer seed, and six-world bootstrap intervals are descriptive.

Independent review: `INDEPENDENT_PILOT_GO_AUDIT.md`. No version, world assignment, training row, threshold, endpoint, checkpoint, candidate learner, or selection rule is changed in making this decision.
