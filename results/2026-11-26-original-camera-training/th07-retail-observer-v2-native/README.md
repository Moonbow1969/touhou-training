# TH07 retail Normal replication on the qualified v2 observer

**The registered stop rule was met.** On the five strictly accepted original-game pairs, the Focus/Bomb-preserving human SFT averaged 5,376 Stage 1 ticks against the parent's 6,344. Its deaths per unit of Stage 1 exposure were 1.180 times the parent's. The stop threshold was 1.0. Actual mean Bomb use remained within its registered range. The candidate received no promotion or further training in this study.

| Registered endpoint | Parent | Fixed SFT 71281 |
|---|---:|---:|
| Strictly accepted pairs | 5 | 5 |
| Stage 1 total native ticks | 31,722 | 26,879 |
| Mean Stage 1 native ticks per credit | 6,344.4 | 5,375.8 |
| Stage 1 deaths / 10,000 ticks | 4.729 | 5.581 |
| Stage 2 arrivals | 0/5 | 0/5 |
| Actual Bombs, total / mean per credit | 38 / 7.6 | 33 / 6.6 |
| Actual Bombs / 10,000 Stage 1 ticks | 11.98 | 12.28 |
| Mean terminal spare lives / Bomb stock | 0 / 0.6 | 0 / 1.2 |
| Deaths with Bomb stock | 4/15 | 6/15 |
| Fraction of observed native ticks in Focus | 50.58% | 52.78% |
| Mean official final score | 666,992 | 506,210 |

The candidate's mean cast count was 0.868 times the parent’s. The intended minimum was six qualified pairs and at least 30,000 Stage 1 ticks per arm. The fixed technical replacement limit stopped enrollment at five pairs and 26,879 candidate ticks. Every one of the 14,636 complete four-tick windows in those five pairs matched the official RPY and the native records. Independent resampling of complete credits gave a descriptive 95% interval of 0.915–1.518 for the death rate ratio and −2,434 to +512 ticks for the mean progress difference. These data support the predefined decision to stop the variant; the small cohort leaves the size and sign of its population effect uncertain.

### Full roster

| Seed | Stage 1 ticks, parent / SFT | Actual casts, parent / SFT | Primary disposition |
|---|---:|---:|---|
| 2173751 | 8,315 / 4,822 | 9 / 6 | Parent observer 125.46 ms, one skipped decision and a missing independent native key witness |
| 2173752 | 8,213 / 5,043 | 9 / 9 | Included |
| 2173753 | 4,974 / 6,362 | 7 / 6 | Included |
| 2173754 | 5,137 / 4,100 | 7 / 4 | Candidate observer 217.27 ms |
| 2173755 | 5,065 / no Game Start | 8 / unavailable | Candidate container exited before the menu handshake |
| 2173756 | 4,804 / 4,830 | 6 / 6 | Included |
| 2173771, reserve | 6,695 / 6,818 | 7 / 8 | Included |
| 2173772, reserve | 7,036 / 3,826 | 9 / 4 | Included |

Fifteen natural original games finished from default-stock Normal Reimu A Game Start through Game Over, with zero continues and zero Stage 2 arrivals. Of the sixteen assigned attempts, one candidate Wine container exited during setup. Its slot never issued a policy action. Thirteen of the fifteen completed games passed all individual control checks. Candidate and parent used the same controller seed, physical CPU set and stock per pair. The original game's RNG was independent. The previous cohort and all engineering games were kept separate.

The two game containers had distinct, unlimited-bandwidth CPU sets and averaged 2.05–2.08 cores apiece, excluding the host actor. Over the fifteen completed games, p99.9 of the native read time was at most 0.10 ms. Two isolated maximum reads exceeded 100 ms. These occurred near the other slot's instance transitions. The TH08 VM's two monitored worker threads reported a last scheduling CPU in a TH07 game's set on 28 of 11,896 quarter-second thread samples. These observations describe placement and timing; they do not identify the cause of either stall.

See the [registration](protocol.json), [primary results](result.json), [per-credit hardware and input telemetry](cohort-telemetry.json), [original endpoints including the menu failure](all-original-endpoints.json), and [qualified observer evidence](../th07-retail-observer-v2/README.md). Per-pair release receipts index the raw original camera, key journals, official RPYs and scorecards for every played game. The movie shows the earliest pair that passed both arms. [Watch the matched full-credit recording](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-observer-v2-held-2173752.mp4).
