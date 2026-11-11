# R4401: full-program control versus an external one-step controller

The same 256 frozen R4301 five-operator programs in each world were
queried with two fixed adapters. The lower-order source had seen at most
two distinct operators in training; the later adapter had seen three-
operator traces but no five-operator program. Each adapter was asked for
the entire trace in one call. The lower-order source was also placed in
an external controller, which supplied the current predicted state and
only the next operator in seven or eight separate model calls. No oracle
state was substituted after an error. The comparison was frozen before
these queries (manifest SHA-256
`3d2fd0d53b520bb1bdb8b18010392136a573ecfbdfec9797e25b83ad5bfd6679`).

| World | Lower-order one call, strict full trace | Lower-order external controller, full trace | Three-operator-trained one call, strict full trace |
|---|---:|---:|---:|
| r1603 | 36/256 | 256/256 | 238/256 |
| r4004 | 7/256 | 256/256 | 251/256 |

At depth seven/eight separately, the lower-order one-call scores are
26/128 and 10/128 in r1603, and 6/128 and 1/128 in r4004. The later
one-call scores are 123/128 and 115/128, then 126/128 and 125/128.
All 3,840 one-step controller transitions match the rule-world state;
all have one exact state and EOS. The paired later-minus-lower one-call
strict gains have 2.5th-percentile lower bounds of .680 and .750 in
r1603, and .891 and .938 in r4004 for depths seven and eight.

Across each world's 256 programs, the controller uses 1,920 model calls
and 140,160 prompt tokens. The later one-call route uses 256 calls and
20,352 prompt tokens. Its generated-token totals are 32,418 in r1603
and 32,384 in r4004; the controller generates 30,720 in each. Thus
the external controller supplies program position and intermediate
state at roughly 7.5 times the calls and 3.2 times the combined token
volume. Batched wall times are recorded in `AGGREGATE.json`; they are
hardware- and batch-dependent and are not used as a latency claim.

This strong baseline establishes that the primitive operators are
available to the lower-order model: external composition is perfect.
Its poor one-call full-program score isolates the missing ability as
autonomous multistep trace control under the original interface. The
three-operator continuation supplies that ability without five-operator
labels. R4301 further shows that the later model follows an edited
visible intermediate state, giving a direct behavioural check of the
execution interpretation. The external controller remains an effective
alternative with a different inference budget; these results do not
claim that one-call output is more accurate than it.

`AGGREGATE.json` retains all world/depth scores and inference costs;
`probe_*/single_*.jsonl` and `probe_*/controller.jsonl` retain every
generation and closed-loop state. The complete remote run is in
`full_archive.tar.gz` (SHA-256
`71da6d839b7a2afae1ac7a776b08250493be248019ec5b94877507b7ca30a4d1`).
