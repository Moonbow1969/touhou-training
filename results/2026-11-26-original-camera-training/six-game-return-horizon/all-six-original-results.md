# Six-game PPO on 192 new Hard and Lunatic original tests

The 96 paired starting allocations produced 0/96 confirmed Stage 2 arrivals for the λ=0.985 actor and 0/96 for the full-return actor. The respective counts of unresolved stages are 0 and 0. All six titles supplied eight pairs at each difficulty. The two fits used the same 192 separate training credits and shared an initial policy, critic and out-of-fold baseline.

| Game and difficulty | Stage 2, λ=.985 → full return | Unknown stage | Mean ending spare lives | Resource | Mean ending resource |
|---|---:|---:|---:|---|---:|
| TH08 Hard | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | last playable Bombs | 0.625 → 1.125 |
| TH08 Lunatic | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | last playable Bombs | 1.625 → 1.500 |
| TH10 Hard | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | final Power | 0.406 → 0.594 |
| TH10 Lunatic | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | final Power | 0.112 → 0.406 |
| TH11 Hard | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | final Power | 0.000 → 0.000 |
| TH11 Lunatic | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | final Power | 0.000 → 0.000 |
| TH12 Hard | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | terminal Bombs | 1.125 → 0.625 |
| TH12 Lunatic | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | terminal Bombs | 0.375 → 0.875 |
| TH13 Hard | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | terminal Bombs | 1.000 → 1.125 |
| TH13 Lunatic | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | terminal Bombs | 0.625 → 0.750 |
| TH17 Hard | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | last playable Bombs | 0.625 → 0.875 |
| TH17 Lunatic | 0/8 → 0/8 | 0 → 0 | 0.00 → 0.00 | last playable Bombs | 1.250 → 1.000 |

TH10 and TH11 spend Power to cast. TH11 resets it after a fatal loss. TH08 and TH17 report Bomb stock from the last playable observation. Game RNG can differ within a pair. One TH08 Hard recorder lost host memory mid-game; the registered replacement rule preserved both original attempts and used a new paired seed. The TH12 physical denominator retains confirmed terminations with incomplete input coverage; the conditional comparison requires both sides to pass the control audit.

[All twelve cells and source hashes](all-six-original-results.json) · [TH08 endpoints and color recordings](evaluation/TH08.md) · [Other five works, action traces and films](evaluation/README.md) · [TH12 coverage sensitivity](evaluation/th12-physical-and-qualified.json) · [Fitted policies](README.md)
