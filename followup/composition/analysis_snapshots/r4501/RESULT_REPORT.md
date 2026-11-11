# R4501: the visible state controls the conditional branch

The fixed r1603 and r4004 adapters were given 256 new five-operator
programs per world, evenly divided by seven/eight instructions and by
whether the next conditional swap D was active. Immediately before D,
the counterfactual prefix exchanged only the two visible control digits.
This flips `first digit > second digit` without changing the two values
that D may swap, the root, any program symbol, or the earlier trace.
Banks and criteria were frozen before the queries (manifest SHA-256
`366cec679e08ad9d7fe70a38ae663c4ccbb0bfe68ea51bb1fbf5f286e0d941c6`).

| World | Unedited next state | Edited next state on new branch | Edited output follows old next state | Edited full suffix, strict EOS |
|---|---:|---:|---:|---:|
| r1603 | 256/256 | 254/256 | 0/256 | 243/256 |
| r4004 | 256/256 | 256/256 | 0/256 | 250/256 |

For the edited active-to-inactive and inactive-to-active branches,
immediate-state success is respectively 128/128 and 126/128 in r1603,
and 128/128 in both directions in r4004. The paired bootstrap lower
bounds for following the edited rather than old next state are .980
and 1.000 by world. All ten predeclared world-level gates pass. Every
continuation has valid state syntax and EOS.

Unlike a broad random-state replacement, this intervention changes the
truth value of a known conditional rule while leaving its two candidate
swap values intact. The model applies the branch implied by the edited
visible state in both directions and usually continues the full unseen
five-operator suffix correctly. Along with R4301, this directly supports
the behavioural claim that the trace carries reusable execution state;
it does not identify the internal neural implementation of that state.

`AGGREGATE.json` and the two `probe_*/analysis.jsonl` files retain the
world, depth, and branch-direction outcomes. The complete raw cloud run
is in `full_archive.tar.gz` (SHA-256
`838bc6118f4efc978d88dcb5b75407c5b84682f3353ec0cc25ca125c5cca5020`).
