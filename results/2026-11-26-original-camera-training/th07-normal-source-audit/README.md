# TH07 Normal replay imitation

The official Touhou 7 v0.11 trial replayed two Normal Reimu A recordings on one Momentum host. The public category listed 327 recordings. Two matched the trial input format and both were accepted by the original Replay menu. The remaining 325 candidates carried the retail build bytes. [Replay catalogue](https://maribelhearn.com/gensokyo?game=PCB&diff=Normal&shot=ReimuA) · [Official trial](https://www16.big.or.jp/~zun/html/th07_update.html)

| Source | Stage | 4-tick training/evaluation pairs | Current keys equal replay word | Previous keys equal preceding word |
|---|---:|---:|---:|---:|
| Boredperson, TRAIN | 1 | 2,672 | 2,712/2,712 | 100.00% |
| zylo, HELD | 1 | 2,617 | 2,649/2,649 | 100.00% |

At native stage tick t the current 16-bit controller word agreed with replay[t]. The archive records a screen read between equal native clock reads, the downsampled X11 image and the absolute change since t-4. Direction, Focus and Shoot predict t+1. Bomb predicts a rising edge in t+1 through t+4. The renderer can present on its own refresh cycle. The clock bracket validates the sampled game state, without individually measuring display-buffer latency.

The learner fine-tuned the shared six-title actor for two uniform passes of the training source using Adam at 0.00001. All four action heads shared the objective. Full optimizer history and weight hashes are in policy/report.json.

| Held human action metric | Shared source | TH07 SFT |
|---|---:|---:|
| Direction cross-entropy (nats) | 1.2037 | 1.1526 |
| Direction top-1 | 61.71% | 62.82% |
| Focus cross-entropy (nats) | 0.3247 | 0.2000 |
| Focus top-1 | 84.60% | 92.13% |
| Shoot cross-entropy (nats) | 0.0882 | 0.0642 |
| Shoot top-1 | 98.85% | 98.78% |
| Bomb cross-entropy (nats) | 0.0150 | 0.0137 |
| Bomb top-1 | 100.00% | 100.00% |
| Sum of four cross-entropies | 1.6316 | 1.4305 |
| Direction on expert changes | 17.2% | 17.0% |
| Direction on stationary-to-moving onsets | 1.8% | 0.6% |

There were 511 held expert changes and 169 held movement onsets. Predicting the preceding direction on every sample gives 80.47%. The held clip contains 0 decision windows with a Bomb press. Per-author action metrics support only this two-source Stage 1 pilot. The policy has not been tested in live play in this experiment.

Source credits and original files are linked in pilot-protocol.json. The two replay entries carry different uploader names. Identity beyond that catalogue attribution was not verified. The camera archive includes derived gameplay observations and telemetry only. The original game executable, data and third-party replay files are available from their respective sources.
