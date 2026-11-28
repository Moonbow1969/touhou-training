# TH08 Normal warm-start spatial readout

Claude Opus 5.5 requested a full-data test after the short from-scratch capacity screen. Both architectures start from the same frozen trained original camera trunk. For each of three seeds both readouts see every one of 58,696 causal training windows five times. Each of the six fits completes 293,480 presentations and 2,295 updates. Held data contain 15,558 windows across six stages from two previously used development pilots.

| Training seed | Spatial gain on busy frames | Stages with busy-frame gain | Overall direction gain | Pooled → spatial median position error |
|---:|---:|---:|---:|---:|
| 280801 | +2.73 pp | 6/6 | +2.56 pp | 46.96 → 40.08 px |
| 280802 | +0.59 pp | 4/6 | +0.82 pp | 51.53 → 43.85 px |
| 280803 | +3.86 pp | 5/6 | +2.23 pp | 51.38 → 41.45 px |

Spatial readout improves overall direction prediction but gets fewer correct directions when the real player switches direction at all three seeds. Spatial errors stay above 40 pixels in this probe, beyond the predeclared 20-pixel promotion threshold and on the stop side of the 40-pixel checkpoint-selection rule. One seed misses the busy-frame threshold as well. These results establish a small offline advantage on certain frames and leave original-game performance open. The spatial checkpoint did not enter native evaluation.

[Review and gates](strategy-review.md) · [Frozen run protocol](protocol.json) · [Paired results](comparison.json) · [Six policies and learner logs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-normal-warm-readout-six-policies.zip)
