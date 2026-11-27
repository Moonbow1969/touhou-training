# Full-action five-title PPO from the reviewed 84-stage human SFT

The human visual source trained on 84 stages and 284,222 available windows across these five games. All four action heads and the visual encoder initialized the shared on-policy actor. Training and held evaluation use separate ordinary full-stock starts on Hard and Lunatic. The TH12 training selection uses strict native input evidence. Held evaluation preserves both physical and coverage-qualified populations.

| Policy | Original PPO training games | Decisions | Accepted epochs | Final maximum title/rank KL |
|---|---:|---:|---:|---:|
| reviewed-ppo | 80 | 35,897 | 4 | 0.00491 |

[Study registration](camera-protocol.json)

[reviewed-ppo weights](policy/policy.ts) · [Optimizer trace](policy/report.json)

[TH10 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/reviewed-five-human-online-th10-exact-optimizer-input.zip)

[TH11 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/reviewed-five-human-online-th11-exact-optimizer-input.zip)

[TH12 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/reviewed-five-human-online-th12-exact-optimizer-input.zip)

[TH13 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/reviewed-five-human-online-th13-exact-optimizer-input.zip)

[TH17 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/reviewed-five-human-online-th17-exact-optimizer-input.zip)

Each title supplied eight Hard and eight Lunatic starts to this update. The actor trained on the local NVIDIA RTX 4070. It controls all nine movement directions, focus, shooting and Bomb. The original TH12 collector reserved 24 full games; 22 had complete input evidence and the frozen rank-balanced rule selected 16. [Accepted epochs and learning rates](learner-console.txt) · [Exact training roster](fit-protocol.json) · [Native movement evidence on three games](three-title-original-action-coverage.json) · [Second-generation protocol](../chained-five-game-ppo/camera-protocol.json)

All 16 TH11 training games yielded an undiscounted return of -3. TH17 had no positive reward rows on these 16 starts, and longer Hard games tended to spend more resources. Discounting at the fitted gamma of 0.999 distinguishes the timing of losses. Within each of the ten title/rank cells, discounted return from the opening correlated positively with playable decision count (Spearman range 0.31 to 0.95, eight starts per cell). This diagnostic uses training data and measures no held performance. [Observed rewards and discounted returns for every start](actual-training-rewards.json)
