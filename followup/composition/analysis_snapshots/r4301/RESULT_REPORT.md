# R4301: models follow a counterfactually edited intermediate state

Two fixed successful adapters were queried on 256 new, state-disjoint
five-operator programs per world. At step three or four, the visible
current state in a partial assistant trace was replaced by an unrelated
valid eight-digit state, while the input, complete instruction sequence,
all earlier printed states, separators, and decoding rule stayed the
same. The replacement was at Hamming distance at least five, and its
next rule-world successor differed from the original successor. The
banks and primary criteria were frozen before any R4301 query (manifest
SHA-256 `63aba923384f69ffb6ee1a281f2177d7f627b3f7e39bae217b3ed797c854a971`).

| World and fixed adapter | Unedited first successor | Edited first successor | Edited output follows original successor | Edited full remaining trace and EOS | Edited preference paired lower bound |
|---|---:|---:|---:|---:|---:|
| r1603, R3803 control +128 | 255/256 | 248/256 | 0/256 | 233/256 | .945 |
| r4004, R4007 calibrated +32 | 255/256 | 253/256 | 0/256 | 247/256 | .973 |

The edited first-successor counts are 124/128 and 124/128 at depths
seven and eight in r1603, and 126/128 and 127/128 in r4004. The
corresponding strict remaining-trace counts are 118/128 and 115/128,
then 124/128 and 123/128. All 1,024 continuations across both worlds
have valid state syntax and EOS. Every predeclared cross-world gate
passes: the unedited and edited successor rates exceed .90 and .75,
and the paired bootstrap lower bound for following the edited rather
than original successor is positive in each world.

The intervention changes only the intermediate value available to the
decoder, not the input or operation sequence. Its next output almost
always follows the replacement, and the whole suffix usually continues
from it. This is direct behavioural evidence that the displayed trace
is used as an execution state for subsequent operations, rather than
being only a decorative explanation of an answer recomputed from the
unchanged prompt. It does not identify an internal activation or
parameter-space mechanism. Together with R4103 and R4201, the result
links coherent intermediate-state supervision to a usable compositional
state interface and to high-order held-out transfer.

`AGGREGATE.json` includes all world/depth counts and confidence bounds;
each `probe_*/analysis.jsonl` retains item-level outcomes. The complete
remote run archive is stored locally as `full_archive.tar.gz` with SHA-256
`f4b1805d837b22905846534434d2b737d6e792f101b7e4a9711c635457808564`.
