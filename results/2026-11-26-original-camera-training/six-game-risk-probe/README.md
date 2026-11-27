# Can the policy’s camera features predict an upcoming loss?

The original game supplies the outcome label. A frozen five-game behavior encoder receives the current and preceding images. A learned residual estimates the probability of a stock loss four through sixteen policy decisions later, in addition to a training-only elapsed-decision baseline for each game and rank. Both arms share the same clock. We removed frames within three decisions of the next loss or sixteen after the preceding loss. Of 192 full credits from the six-game update, 144 trained the probe and 48 tested it. Eligible decision counts were 65,611 and 22,667.

| Game | Difficulty | Brier loss change, camera + clock minus clock | Average precision, clock → camera + clock |
|---|---|---:|---:|
| TH08 | Hard | +0.0028 | 0.140 → 0.242 |
| TH08 | Lunatic | -0.0098 | 0.146 → 0.361 |
| TH10 | Hard | +0.0184 | 0.288 → 0.297 |
| TH10 | Lunatic | +0.0330 | 0.609 → 0.514 |
| TH11 | Hard | -0.0368 | 0.452 → 0.644 |
| TH11 | Lunatic | -0.0450 | 0.327 → 0.636 |
| TH12 | Hard | -0.0013 | 0.172 → 0.284 |
| TH12 | Lunatic | +0.0001 | 0.151 → 0.245 |
| TH13 | Hard | -0.0050 | 0.225 → 0.278 |
| TH13 | Lunatic | -0.0072 | 0.156 → 0.339 |
| TH17 | Hard | +0.0018 | 0.168 → 0.140 |
| TH17 | Lunatic | -0.0006 | 0.188 → 0.178 |

Across the 48 held credits, mean average precision rose from 0.252 to 0.347. The primary Brier loss difference was −0.0041, with a whole-credit 95% bootstrap interval of [−0.0087, +0.0005]. The largest Brier improvements occurred on TH11. Calibration worsened on both TH10 difficulties. The classifier had no role in choosing actions.

[Every held credit and training metrics](report.json) · [Split, endpoint definition and comparison](protocol.json) · [Probe weights](risk-head.pt)
