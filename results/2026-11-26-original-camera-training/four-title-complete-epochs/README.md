# Four-title full-epoch camera policy training on Momentum

Twenty complete traversals made 5,142,900 sample presentations. From the first to the last epoch the four TH17 development players gained 71 correct starts and 89 correct turns. Their mean all-direction negative log likelihood worsened from 1.522 to 1.570. At epoch 20, all four development games also had worse mean turn likelihood than at epoch 1. The registered original-game test compares these two fixed endpoints. [Per-game curves](development/movement-by-epoch.png) · [Counts and player-weighted likelihoods](development/by-title-and-epoch.json) · [Original test design](../full-epochs-original/camera-protocol.json)

Each epoch visits all 257,145 eligible four-tick human windows once in random order. The 74 original replay stages span TH10, TH12, TH13 and TH17 on Hard and Lunatic. Every work and difficulty carries equal total loss weight over an epoch. Each minibatch updates the camera encoder and movement, focus, shot and Bomb heads. The same initialization and optimizer run supply the snapshots after 1, 3, 10 and 20 passes.

Evaluation uses separate credited players. Ten earlier development stages and eight repeated TH17 stages report action imitation. The six new TH10/12/13 players have a separate prospective registration. The prior 68-stage study used a different sampling schedule, batch size and set of source stages.

| Epoch | Unique windows per epoch | Total window presentations | Weighted joint loss | Held TH17 directions | Correct starts | Correct turns |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 257,145 | 257,145 | 1.9286 | 11,930/24,643 | 133/2,476 | 392/3,171 |
| 3 | 257,145 | 771,435 | 1.7534 | 12,045/24,643 | 125/2,476 | 421/3,171 |
| 10 | 257,145 | 2,571,450 | 1.5877 | 12,198/24,643 | 152/2,476 | 450/3,171 |
| 20 | 257,145 | 5,142,900 | 1.4920 | 12,101/24,643 | 204/2,476 | 481/3,171 |

[Registration and exact stage hashes](protocol.json) · [Original replay capture and exclusions](../human-train-six-complete-credits/later-train-complete-roster-frozen.json) · [Prospective six-player allocation](../unseen-later-stage-author-challenge/prospective-six-players-frozen.json)

[Epoch 1 weights](epoch-01/policy.ts) · [Human metrics and training log](epoch-01/report.json)

[Epoch 3 weights](epoch-03/policy.ts) · [Human metrics and training log](epoch-03/report.json)

[Epoch 10 weights](epoch-10/policy.ts) · [Human metrics and training log](epoch-10/report.json)

[Epoch 20 weights](epoch-20/policy.ts) · [Human metrics and training log](epoch-20/report.json)

[Exact training data and initialization](training-data.md)

Five of six prospectively allocated new players supplied synchronized later-stage screens on TH10, TH12 and TH13. One TH12 Lunatic stage fell below the frozen optical alignment threshold on both capture attempts. In 19,551 qualified decisions, movement hits rose from 9,086 at epoch 1 to 9,175 at epoch 20. Start hits rose from 127 to 180 of 1,694 and turns from 324 to 410 of 2,621. Separate full-stock original testing recorded 0/64 Stage 2 for each epoch. [Per-title metrics and all four epochs](prospective-six/by-title-and-epoch.json) · [Learning curves](prospective-six/movement-by-epoch.png) · [128 native tests](../full-epochs-original/evaluation/README.md) · [Frozen human capture roster](../unseen-later-stage-author-challenge/held-original-synchronized-frozen.json)
