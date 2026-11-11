# Exploratory CPU pilot: manuscript invariants on frozen E4 score trajectories

**The six E4 confirmation worlds have already been viewed. Every result here is descriptive reuse, not new confirmation. Learner selection with only two pilot worlds is highly unstable. Nothing in this directory supersedes the frozen E4 analysis.**

## Feasibility and definitions

Yes: FEATURES.json retains all ten diagnostic exact cell scores, the four threshold masks, current raw temporal vectors and targets. The generator specifies the fixed 14-cell complex (four structural vertices, six edges, four triangular faces, no 3-cell). Thus its score-based complexes, induced homology maps and full zigzag intervals can be computed without raw model generations. This reconstructs the mathematical invariants conditional on exported scores; it does not independently validate those scores or permit item-level resampling.

The computation uses the manuscript's exact F2 homology and union zigzag implementation. It computes historical artifact alpha in degrees 0–2 as beta(history) minus rank of all checkpoint images into history; bridge rho as the cokernel of both endpoint images; windowed left/right/both kappa at theta 0.6/0.7/0.8/0.9 and delta 0/0.025/0.05. The two-sided kappa is tau. The window computation maps strict-threshold bridge homology into permissive-threshold bridge homology and quotients by permissive-threshold endpoint images, as in the current manuscript. No future checkpoint enters any feature.

Family A appends these 132 invariant coordinates to the frozen 148-coordinate raw temporal baseline. Family B appends 96 summaries of full union barcodes on each observed prefix: bar count, live count, sum/max live node lengths, ended count, sum/max ended node lengths, and bridge-only singleton count, in each degree and threshold. All exact prefix intervals are retained in INVARIANTS.json. These summaries are one fixed exploratory encoding; this is not an exhaustive test of every possible barcode learner.

## Validation

The target is external strict exact accuracy at the current checkpoint, as in e4_s5.tex. The original five learner candidates and clipping are reused; leave-one-pilot-world-out MSE on worlds 961/962 selects each augmented family. Both select standardized ridge alpha 100, the same learner frozen for raw temporal. Fits use pilot worlds only. Reused worlds 971–976 supply descriptive evaluation. Loss is averaged over both arms and 17 checkpoints per world before paired bootstrap (10,000 draws, seed 42). Checkpoints are not independent replicates.

The frozen raw temporal MSE was reproduced to numerical tolerance. All exported masks agree with reconstructed masks. A split-triangle check produces rho_1=alpha_1=1; every computed tau obeys window monotonicity. The vendor barcode routine internally checks its full interval decomposition against node dimensions.

## Findings

Across all eight worlds, two arms, 17 checkpoints, four thresholds and degrees 0–2, **alpha, rho and tau are identically zero**, including the nonzero windows. These quantities therefore offer no varying signal in this E4 regime. One-sided kappa is nonzero on 7 degree-one and 70 degree-two checkpoint/threshold entries (counting an entry if any tested one-sided window is positive). This mostly captures boundary acquisition and the final two-dimensional tetrahedral shell class. The absence of a 3-cell is part of the declared E4 ambient model.

| Family | Pilot leave-world-out selected MSE | Reused-world MSE | Gain vs raw temporal | Descriptive 95% world interval | Positive worlds |
|---|---:|---:|---:|---|---:|
| Frozen raw temporal | 0.0216075 | 0.0217452 | — | — | — |
| Raw temporal + exact invariants | 0.0217245 | 0.0217582 | -0.00001297 | [-0.0003603, 0.0004297] | 2/6 |
| Raw temporal + prefix barcode summaries | 0.0239968 | 0.0236940 | -0.00194882 | [-0.0064814, 0.0029472] | 2/6 |

Paired MSE gains (positive favors augmentation):

| World | Exact invariants | Prefix barcodes |
|---|---:|---:|
| 971 | 0.00005948 | -0.00433533 |
| 972 | -0.00056321 | -0.00186737 |
| 973 | -0.00004745 | -0.00791824 |
| 974 | 0.00100518 | 0.00746308 |
| 975 | -0.00008260 | -0.00893101 |
| 976 | -0.00044920 | 0.00389597 |

Neither augmentation improves pooled MSE over the frozen raw temporal baseline, and both paired intervals cross zero. Their selected pilot validation errors are also worse. There is no positive predictive-utility evidence here and no justification from this pilot for buying a direct independent GPU replication of the same E4 design.

A scientifically useful new GPU study would first require a separately designed non-nested regime where alpha/rho/tau vary across independent worlds, an external target with meaningful variation, adequate independent development worlds, and a prospectively fixed encoding and learner. The existing conflicting-rule forgetting design demonstrates a possible mathematical regime, but its association with an independent behavioral endpoint remains unestablished. This is a design condition, not evidence that a redesigned experiment will succeed.

## Files and scope

- pilot.py: reproducible computation; reads frozen pilot and confirmation exports, writes only here.
- INVARIANTS.json: exact invariant coordinates, full causal prefix intervals and summaries.
- RESULTS.json: all five pilot candidate losses, selected learners, reused-world predictions, and paired losses/intervals.

No frozen source, manuscript, original confirmation analysis, archived raw data, cloud state or model training was modified.
