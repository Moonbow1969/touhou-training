# Human camera SFT on six games, including TH16 Hard and Lunatic

One shared camera policy trained on 86 original human replay stages from TH10, TH11, TH12, TH13, TH16 and TH17. The dataset contains 289,336 distinct four-tick windows. The fit sampled 192,000 presentations in 3,000 updates, balancing across all twelve game and difficulty groups. The optimizer saw 15,176 TH16 Hard and 16,160 TH16 Lunatic presentations.

| TH16 human source | Character and season | Distinct windows | Bomb onsets |
|---|---|---:|---:|
| Hard, smilelkj | Reimu Summer | 2,163 | 2 |
| Lunatic, とーほー | Marisa Autumn | 2,951 | 2 |

The four shared outputs are movement, focus, shooting and Bomb. The original TH16 evidence also preserves 31 seasonal input onsets. In 29 cases the native release state activated and the season meter fell. The two observed high input bits coincided throughout 21,775 source ticks. This actor has no season-release output. Both TH16 players contributed training data. Assessing TH16 gameplay or performance on new TH16 players requires separate tests.

[Supervised weights](policy/policy.ts) · [Training loss](policy/loss.jsonl) · [Actual per-rank draws](policy/report.json) · [Six-title training registration](protocol.json) · [Both native human source records](added-th16.json) · [Original cameras and checksum](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th16-retail-hard-lunatic-first-human-cameras-receipt.json)
