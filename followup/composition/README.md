# Visible-state composition: Supplement S6

This directory contains code and results for the visible-state composition studies. `ARCHIVES.json` records the archive sizes,
digests, member roots, exact saved adapters, and paths relative to the
experiment follow-up workspace. Large files remain in those archives.
`archived_sources/` contains verbatim lightweight source members; the
`analysis_snapshots/` directory holds separately identified returned
summaries. See `SOURCE_MANIFEST.json` for the origin of each file.

| Evidence | Original study identifiers |
| --- | --- |
| r1603 matched shell/control and source | r3803 |
| r4004 source, entry repair, fixed failure and stopping calibration | r4004/r4005, r4006, r4007 |
| Reversible supervision and dose-matched clean branches | r4103, r4201 |
| Visible state and conditional branch interventions; controller baseline | r4301, r4501; r4401 |
| Failed equal-dose topology comparison and single-face separation | r4601, r4701 |
| Operator variety by length; six-source minimum-order study | r4901, r5001 |

Restore each archive using its original member root under the indicated
working-root subdirectory (the r4006/r4007 archives belong below
`prospective_world_4004`). The complete archives contain the data and run
records needed by their original analysis commands. The protocols record the stopping and program-length amendments. The [trained weights release](https://github.com/Shengda21/zigzag/releases/tag/trained-weights-v1) contains the saved adapters.
Use the extraction roots in `ARCHIVES.json` to place the weights. The complete
raw archives are stored separately.
