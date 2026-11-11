# R4006 development extension: strict gate failed because of trace overrun

Starting from the R4005 update-128 adapter, R4006 trained on 3,072
previously unseen nine/ten-step programs with exactly three operators and
1,024 unary/binary anchors. Its four checkpoints were selected against the
previously queried R4005 bank under the frozen protocol. No checkpoint met
the strict eighth-step gate, so the untouched R4006 confirmation bank was
not queried. Its manifest SHA-256 is
`791d603349008070c8c38d873f9fb70780edb688e4d79449e252b3adc080cb2b`.

| Checkpoint | Strict five-operator pairs, depth 7 | Strict depth 8 | Correct-state-prefix pairs, depth 7 | Correct-state-prefix depth 8 | H2 grid bootstrap |
|---|---:|---:|---:|---:|---:|
| R4005 source | 219/256 | 171/256 | 222/256 | 171/256 | 1.000 |
| R4006 +32 | 178/256 | 128/256 | 224/256 | 211/256 | 1.000 |
| R4006 +64 | 164/256 | 91/256 | 241/256 | 232/256 | 1.000 |
| R4006 +96 | 199/256 | 89/256 | 245/256 | 236/256 | 1.000 |
| R4006 +128 | 216/256 | 133/256 | 248/256 | 239/256 | 1.000 |

At update 128, 504/512 seven-step and 495/512 eight-step programs have
every requested state correct. Of these, 34 and 119 respectively append
exactly one extra state. All 1,024 hard-bank programs use valid state
syntax and a valid EOS. Thus the longer low-variety curriculum improved
held-out five-operator state transitions, especially at depth eight, but
trained the model to continue traces after the requested instruction
count. This explains the apparent reversal under strict scoring. It does
not change the failed predeclared R4006 result. The diagnostic motivates
the separately frozen R4007 stop-calibration intervention, which uses no
five-operator labels and must pass strict scoring on a new bank.

Per-program outputs and independent scores are retained in the four
`probe_extension_*` folders. `ERROR_DIAGNOSTIC.json` records first error
positions and the extra-state decomposition.
