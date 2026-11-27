# How much expert input changes inside a four-tick decision?

These 47 synchronized Hard and Lunatic replay stages are the actual training sources of the fitted four-title camera SFT. There are 148,415 eligible windows and 593,660 observed native ticks. The deployed four-tick movement convention selects the first direction and holds it. The table compares that rule with two successive two-tick commands. The final column gives the minimum disagreement possible if an oracle sees all four expert ticks and selects the most frequent direction in hindsight.

| Title and difficulty | Human windows | Two-tick direction mismatch | Four-tick direction mismatch | Four-tick oracle minimum |
|---|---:|---:|---:|---:|
| TH10 Hard | 12170 | 4.14% | 11.91% | 8.07% |
| TH10 Lunatic | 6087 | 3.90% | 11.34% | 7.72% |
| TH12 Hard | 9980 | 4.09% | 12.18% | 8.27% |
| TH12 Lunatic | 11848 | 4.42% | 12.61% | 8.60% |
| TH13 Hard | 10903 | 3.39% | 9.50% | 6.49% |
| TH13 Lunatic | 13964 | 4.34% | 11.97% | 8.12% |
| TH17 Hard | 43173 | 4.70% | 13.17% | 8.98% |
| TH17 Lunatic | 40290 | 4.96% | 13.81% | 9.37% |

This is a measurement of recorded human input. Executing shorter commands after one camera observation requires predicting them. A registered matched-capacity visual SFT comparison tests prediction of two successive commands from the same observation. Native survival needs a separate original-game test. [Counts by stage, including focus, shot and Bomb](report.json) · [Prediction comparison](../../../four-title-subwindow-sft/protocol.json)
