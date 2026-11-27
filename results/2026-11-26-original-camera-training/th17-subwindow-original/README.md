# Testing the TH12 two-command intervention on TH17

Two private official TH17 trial instances ran side by side on Accelerator. Each policy received a screen every four original ticks. The four-tick model held one action, and the two-command model predicted a separate action for each half of the same window. Each difficulty supplied eight ordinary full-stock pairs.

| TH17 difficulty | Stage 2, four ticks → two plus two | Mean first life-loss tick | Mean last Stage 1 tick | Mean last playable Bombs | Exact order-test p for progress |
|---|---:|---:|---:|---:|---:|
| Hard | 0/8 → 0/8 | 713.0 → 663.4 | 2,204.8 → 2,375.1 | 3.000 → 2.750 | .611 |
| Lunatic | 0/8 → 0/8 | 428.8 → 638.5 | 1,640.5 → 2,377.3 | 3.000 → 2.875 | .111 |

All 32 credits ended with zero spares. The original game consumed 6,975 of 6,992 changed midpoint commands at the measured tick, with one mismatch and 16 unresolved observations. The two instances together completed 95.47 full credits per hour including resets, with 14.25 observed decision frames and 56.94 native ticks per wall second. A 17-credit transport sample averaged 8.34 MB per game. All step-level inference used the local Unix socket.

TH12 also showed higher mean terminal Stage 1 ticks on both difficulties with this intervention. Neither work produced a Stage 2 arrival in either arm. Game-specific progress improvements remain uncertain with eight pairs in each cell. [TH12 comparison](../th12-subwindow-original/README.md) · [TH17 pairs and exact order tests](original-paired-results.json) · [Measured throughput](actual-native-throughput.json) · [LAN bytes](original-lan-payload.json) · [All 32 original records](th17-held-native-records.release.json) · [All 32 movies](th17-whole-credit-movies.release.json)
