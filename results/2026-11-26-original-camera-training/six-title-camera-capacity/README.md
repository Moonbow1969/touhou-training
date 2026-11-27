# Matched six-game camera model capacity test

Two policies learned movement, focus, shooting and Bomb on all 86 original replay stages from six titles. The convolutional policy has 165,691 deployed parameters. Adding residual self-attention over its 7 × 6 spatial grid raises the count to 249,235. The residual starts at zero. The recorded digest matches for all 192,000 ordered training samples, horizontal flips and brightness multipliers in both fits.

| Original development game | Correct directions, CNN → attention | Windows | Correct turns, CNN → attention |
|---|---:|---:|---:|
| TH10 | 4,288 → 4,286 | 6,600 | 145 → 156 |
| TH11 | 1,282 → 1,312 | 2,446 | 34 → 37 |
| TH12 | 5,114 → 5,046 | 9,806 | 223 → 241 |
| TH13 | 7,973 → 7,935 | 11,835 | 299 → 285 |
| TH17 | 11,727 → 11,800 | 24,643 | 376 → 376 |

On the repeated group of three TH11 players, all-direction accuracy changed from 4,357 to 4,368 of 8,425. Correct starts changed from 38 to 34 of 717 and turns from 124 to 118 of 926. This single matched fit pair shows mixed effects across games. The separately registered TH16 test compares both six-title fits with the five-title actor that never trained on TH16. Its four new players are excluded from training.

[Identical input digests and actor identities](matched-inputs.json) · [CNN weights](cnn/policy.ts) · [Attention weights](attention/policy.ts) · [Development stages](development/held.json) · [Repeated TH11 players](repeated-th11/held.json) · [TH16 prospective comparison](prospective-th16-comparison.json) · [Frozen TH16 roster identifiers](prospective-th16-roster.json)
