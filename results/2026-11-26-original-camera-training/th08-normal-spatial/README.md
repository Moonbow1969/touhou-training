# TH08 Normal player localization and policy supervision

The earlier policy-history and action-duration comparisons motivated a visual representation check. Original build 0.02 records provide synchronized native player coordinates. A small probe trained on 7,680 windows from five players and evaluated on 3,072 other windows from two held players. A linear readout of the 128-dimensional policy embedding had median position error **59.0 px** in the original playfield, compared with **10.2 px** for a learned 1×1 convolution on the frozen spatial features. The latter still had **237.8 px** P90 error. The pretrained localizer's median was 15.3 px. This scores linear decodability of player position only. Bullet coordinates and collision hazards were not labeled in this study.

We then trained two policies with the same learned spatial-attention route, frozen image trunk and initialization. Only the treatment optimized an additional position heatmap loss. Both used the same sample indices, 60,638 eligible original human windows, 800 updates and 38,400 presentations per arm. The resulting actors accept only the standard camera pair. These two previously used held pilots are a development set.

| Held camera frames | Action imitation | Action imitation plus location |
|---|---:|---:|
| Direction accuracy, 19,274 windows | 49.23% | 49.18% |
| Direction accuracy at changes, 4,221 windows | 23.74% | 23.69% |
| Direction NLL | 1.4692 | 1.4659 |
| Median localization error across individual stages | 13.9–16.1 px | 10.1–10.9 px |
| P90 localization error across individual stages | 227.4–328.4 px | 193.6–326.7 px |

The spatial route improved localization under matched supervision, with effectively unchanged direction accuracy. The policy checkpoints have no original-game results. The next visual diagnostic requires verified projectile labels or an isolated rendered-pixel intervention. A player localization win is insufficient justification for a larger training budget.

[Frozen feature probe](../th08-normal-player-probe/report.json) · [Probe weights](../th08-normal-player-probe/probes.pt) · [Matched policy protocol](protocol.json) · [Per-stage results and update log](report.json) · [Imitation actor](imitation.ts) · [Location-supervised actor](imitation-plus-location.ts)

## Recovery distribution check on the original

Across the eight subsequently verified Stage 1 credits, the camera embedding moved farther from held human play near **13 of 16** nonterminal life losses with the grayscale and difference pair. With the difference channel zeroed, that held for **6 of 16**. Median change in human reference percentile was +5.8 with both channels and −8.1 with grayscale alone. Human replay input had a four-native-tick interval, while original policy decisions averaged about 100 ms. Since the shift is sensitive to that channel and capture regimes differ, this check leaves the cause of the deaths unresolved. [Per-credit measurements](recovery-probe-eight-credits.json)
