# TH08 Normal Ghost architecture screen

Claude Opus 5.5 reviewed the original-game outcomes and chose a four-arm, fixed-budget architecture comparison. Each arm used the same replay data and 500 optimizer updates at each of three seeds. The 58,696 eligible camera windows represent five Normal pilots. Draws per run numbered 12,000, of which 10,835–10,939 were distinct. The six previously used held stages from two other pilots provided 15,558 camera windows. Twelve candidate fits completed. Their gradient loops used 117 seconds total on the RTX 4070, excluding validation and serialization.

| Architecture | Parameters | Busy-frame gain in percentage points, three seeds | Median player-location error range |
|---|---:|---|---:|
| 7×6 pooled baseline | 233,804 | Reference | 60.2–63.3 px |
| 14×12 spatial keypoints | 234,566 | +0.08, +0.41, +0.75 | 56.1–60.4 px |
| Wider 7×6, matched estimated compute | 520,608 | +0.13, 0.00, −0.15 | 61.1–74.7 px |
| 7×6, approximately four times the parameters | 922,936 | +0.10, 0.00, +0.08 | 65.7–71.4 px |

Every candidate missed the registered 1.5-point busy-frame gate, and the spatial readout missed the 20-pixel mechanism criterion. The widened pooled arm used 3.4% fewer estimated multiply-accumulates than the spatial arm. These results apply to the short from-scratch budget and the held pilots. A larger checkpoint was not selected for original deployment. The study cannot establish a model-size ceiling.

[Claude review and criteria](strategy-review.md) · [Run protocol](protocol.json) · [Paired seed metrics](comparison.json) · [Comparison plot](capacity-comparison.png) · [Twelve policy checkpoints and optimizer logs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-normal-capacity-twelve-policies.zip)
