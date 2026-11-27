# Does keeping samples per game constant preserve imitation accuracy?

All four fits ran on the same RTX 5090 with the same encoder, training seed, replay roster within each title set and sampling recipe. The five-title treatment adds two TH11 stages. The longer run increases training presentations from 192,000 to 240,000. Both the four-title short fit and the five-title long fit expect 48,000 draws per training game.

| Original game | Four titles, 3,000 | Five titles, 3,000 | Four titles, 3,750 | Five titles, 3,750 |
|---|---:|---:|---:|---:|
| TH10 | 4,324/6,600 | 4,247/6,600 | 4,323/6,600 | 4,184/6,600 |
| TH11 | 1,230/2,446 | 1,271/2,446 | 1,170/2,446 | 1,212/2,446 |
| TH12 | 5,290/9,806 | 5,056/9,806 | 5,108/9,806 | 5,132/9,806 |
| TH13 | 8,049/11,835 | 7,917/11,835 | 8,044/11,835 | 7,843/11,835 |
| TH17 | 11,979/24,643 | 11,738/24,643 | 11,904/24,643 | 11,742/24,643 |

Each cell counts correctly predicted human movement directions. The [results by game and player](scores.json) include turn, start and stop counts and movement log loss. These 19 development stages informed this comparison. The study measures imitation on known games. Separate campaigns measure survival.

[Registration](protocol.json) · [Development curves](movement.png)

[four-3000 actor](four-3000/policy.ts) · [Measured game/rank draws and held actions](four-3000/report.json) · [Training loss](four-3000/loss.jsonl)

[five-3000 actor](five-3000/policy.ts) · [Measured game/rank draws and held actions](five-3000/report.json) · [Training loss](five-3000/loss.jsonl)

[four-3750 actor](four-3750/policy.ts) · [Measured game/rank draws and held actions](four-3750/report.json) · [Training loss](four-3750/loss.jsonl)

[five-3750 actor](five-3750/policy.ts) · [Measured game/rank draws and held actions](five-3750/report.json) · [Training loss](five-3750/loss.jsonl)
