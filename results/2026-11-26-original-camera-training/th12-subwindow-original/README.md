# Two learned actions per camera observation in TH12

Both supervised policies observed the original game once every four native ticks. The first held its selected action through the window. The second predicted one action for the first two ticks and another for the next two using the same camera observation. Both played 16 ordinary full-stock credits.

| Difficulty | Pairs | Stage 2, four ticks → two plus two | Mean last Stage 1 tick | Mean first life-loss tick | Mean Bombs at end |
|---|---:|---:|---:|---:|---:|
| Hard | 8 | 0/8 → 0/8 | 1,750.5 → 2,080.6 | 471.0 → 519.5 | 2.000 → 1.875 |
| Lunatic | 8 | 0/8 → 0/8 | 1,516.3 → 1,761.6 | 311.9 → 384.1 | 1.875 → 1.875 |

The second command differed from the first in 4,903 of 6,877 windows for the two-action model. In these changed half-windows the original input reader recorded 9,776 matching ticks, one mismatch and five missing reads. All 32 games ended with zero spare lives. For final Stage 1 progress, the two-sided randomized-order p-value was .222 on each difficulty. The paired-order analysis was specified after observing the first pair. Every original endpoint and timing exception remains in the records.

[Original allocation and actor hashes](th12-pre-outcome-registration.json) · [All paired endpoint records](held-pairs.json) · [Actual tick-level actuation](whole-credit-actuation-audit.json) · [Order-randomization analysis](paired-order-results.json)
