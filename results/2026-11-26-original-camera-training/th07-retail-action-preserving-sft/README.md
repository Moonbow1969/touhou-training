# TH07 retail human SFT with parent Focus and Bomb targets

The registered 71281 checkpoint met the offline criteria on both held human authors and both original-game camera traces. It gained 3.66 percentage points in movement accuracy and 2.08 points on decisions where the human changed direction. Its median player localization error was 6.33 native pixels. This experiment measures behavior on recorded images. A new ordinary Game Start evaluation is needed to determine survival and Stage 2 arrival.

| Training seed | Held movement change | Held direction-switch change | Bomb probability ratios, cases A / B | Focus probability differences, cases A / B |
|---|---:|---:|---:|---:|
| 71281, registered | +3.66 pp | +2.08 pp | 1.011 / 0.973 | −0.001 / +0.013 |
| 71282 | +3.64 pp | +2.21 pp | 1.035 / 1.005 | −0.004 / +0.007 |
| 71283 | +3.73 pp | +2.34 pp | 0.999 / 0.953 | +0.010 / +0.024 |

Every seed started from the same four-head visual model. Each saw all 16,000 training windows five times in 2,500 updates on an RTX 5090. Identical per-epoch SHA-256 digests verified that its ordered minibatches matched the earlier four-head human SFT. Movement and shooting used human action targets, and the auxiliary loss used the player's original position. Focus and Bomb used the parent's full two-class distributions on the training frames, with a fixed KL weight of eight on each head. Assessment used 8,000 windows from two other authors.

Case A contained 1,303 positive-Bomb-stock observations from an ordinary parent-policy credit, and B contained 988 from the earlier SFT credit. Replaying both models on the same original images controls for the encountered state. In A, the registered child's Bomb probability averaged 1.488% versus 1.471% for the parent. In B it averaged 1.589% versus 1.633%. The same held humans produced only one positive Bomb label among 8,000 windows, which limits assessment of when to cast from those labels alone.

The protocol and acceptance file bind the training bank, reference model, signed original recordings, fixed candidate, and thresholds. The result directory includes all three final models, per-stage and per-author action metrics, minibatch hashes, and individual source-camera audits. The complete training log is in `train.log`.
