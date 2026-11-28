# TH08 Normal camera forecast before a fatal hit

We trained small supervised readouts to test whether the deployed TH08 Normal Ghost encoder retained enough information for an early risk signal. The input cohort contains 45 original-game credits, 40,074 policy decisions and 135 fatal hits. The six paired policy evaluations remained sealed. The deployed encoder had already been updated with PPO on these training credits. The whole-credit split assesses the new readout, with fresh originals required before any live use.

The physical observer found all 135 hits one native tick at a time. Each episode matched a later stock decrement or Game Over. The HUD-derived endpoint lagged its physical hit by 28–36 native ticks. Five reversible hit episodes were excluded as endpoints. Both camera images and intervening clock samples had to be in the normal player state, with no missing intermediate ticks, to qualify. The primary forecast window was 8–32 ticks before the physical hit. A 16–40-tick window served as a sensitivity check.

The two-layer readout sees only the frozen 128-dimensional grayscale camera representation. A logistic reference sees elapsed time, causal native x/y, current stocks and time since the previous hit. In each of five rotations, three whole-credit folds train the readouts, one calibrates the probabilities and sets the alarm cutoff at 5% false positives, and one scores the results. The groups preserve the proportions of the two original collection policies.

| Held-fold metric | Position, resources and clock | Frozen camera features |
|---|---:|---:|
| Mean average precision, 8–32 ticks | 0.1136 | 0.1212 |
| Fatal hits alerted at least 20 ticks ahead | 19/135 | 44/135 |
| Mean average precision, 16–40 ticks | 0.1140 | 0.1107 |
| Fatal hits alerted at least 28 ticks ahead | 20/135 | 37/135 |

The primary window contained 17,706 eligible decisions and 780 positives. The camera readout's observed held false-positive rate ranged from 3.13% to 7.43% across folds. All five folds missed the joint gate of at least 1.5× the reference average precision, at least 50% early hit coverage and calibration error at most 0.02. This probe produced no change to the live controller or PPO reward.

[Frozen protocol](protocol.json) · [Per-fold results and optimizer logs](report.json) · [Held-fold figure](held-folds.png) · [Original PPO evaluation](../th08-progress-lambda1-round2-evaluation)
