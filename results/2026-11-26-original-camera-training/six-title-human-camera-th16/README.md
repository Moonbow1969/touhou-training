# Human camera SFT on six games, including TH16 Hard and Lunatic

One shared camera policy trained on 86 original human replay stages from TH10, TH11, TH12, TH13, TH16 and TH17. The dataset contains 289,336 distinct four-tick windows. The fit sampled 192,000 presentations in 3,000 updates, balancing across all twelve game and difficulty groups. The optimizer saw 15,176 TH16 Hard and 16,160 TH16 Lunatic presentations.

| TH16 human source | Character and season | Distinct windows | Bomb onsets |
|---|---|---:|---:|
| Hard, smilelkj | Reimu Summer | 2,163 | 2 |
| Lunatic, とーほー | Marisa Autumn | 2,951 | 2 |

The four shared outputs are movement, focus, shooting and Bomb. The original TH16 evidence also preserves 31 seasonal input onsets. In 29 cases the native release state activated and the season meter fell. The two observed high input bits coincided throughout 21,775 source ticks. This actor has no season-release output. Both TH16 players contributed training data. Assessing TH16 gameplay or performance on new TH16 players requires separate tests.

Under the same total sampling budget, all five existing games lost some top-choice direction matches on their previous development stages after TH16 was added.

| Development game | Correct directions, five → six training titles | Windows |
|---|---:|---:|
| TH10 | 4,358 → 4,287 | 6,600 |
| TH11 | 1,327 → 1,282 | 2,446 |
| TH12 | 5,197 → 5,115 | 9,806 |
| TH13 | 7,999 → 7,973 | 11,835 |
| TH17 | 11,874 → 11,726 | 24,643 |

On the three repeatedly measured TH11 players, correct direction predictions changed from 4,450 to 4,357 of 8,425. Sampling fewer examples per existing game and adding new content changed together. [All original development stages](development/held.json) · [Three repeated TH11 players](repeated-th11/held.json) · [Same-data model capacity comparison](../six-title-camera-capacity/protocol.json)

[Supervised weights](policy/policy.ts) · [Training loss](policy/loss.jsonl) · [Actual per-rank draws](policy/report.json) · [Six-title training registration](protocol.json) · [Both native human source records](added-th16.json) · [Original cameras and checksum](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th16-retail-hard-lunatic-first-human-cameras-receipt.json)
