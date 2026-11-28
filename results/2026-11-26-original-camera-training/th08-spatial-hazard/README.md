# TH08 four-frame spatial hit forecast

This model receives four original grayscale images at native tick offsets −12, −8, −4 and 0, plus their signed differences. The stride-four encoder keeps a 28×24 spatial map. A learned player locator predicts the center of a 7×7 feature crop. The forecast uses that crop and global pooled features. Original player coordinates supervised the locator during training. The camera model takes no coordinates at inference.

The same 45 original credits, held-credit folds and physical fatal-hit labels from the frozen-encoder experiment were used. Requiring three exact and vulnerable interframe gaps left 17,067 observations for evaluation. The privileged logistic baseline was refit on the same observations as the spatial model. Five independent models ran 540 updates each on an RTX 4070.

| Held fold | Baseline AP | Spatial AP | Spatial early alerts | Held false positives | Median player error |
|---|---:|---:|---:|---:|---:|
| 1 | 0.086 | 0.068 | 3/27 | 5.91% | 41.3 px |
| 2 | 0.084 | 0.150 | 10/30 | 3.53% | 31.0 px |
| 3 | 0.155 | 0.129 | 9/27 | 3.66% | 32.9 px |
| 4 | 0.113 | 0.132 | 17/27 | 7.87% | 29.3 px |
| 5 | 0.148 | 0.153 | 2/24 | 3.62% | 39.4 px |

The spatial model alerted on 41/135 fatal hits at least 20 original ticks ahead. No fold passed the combined gate. Held localization error at the 90th percentile ranged from 107 to 152 native pixels. The full result motivates a matched resolution and localization comparison on newly collected credits. The result does not identify the cause of the weak forecasts. No policy or reward changed in this experiment.

[Forecast figure](held-folds.png) · [Frozen protocol](protocol.json) · [Held metrics and optimizer logs](report.json) · [Physical hit truth](../th08-camera-hazard/native-hit-vs-hud-audit.json)
