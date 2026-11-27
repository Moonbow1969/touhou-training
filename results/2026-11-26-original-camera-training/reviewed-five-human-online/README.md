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

The Windows coordinator now imports numerical and image libraries only while scoring completed games. Four live waiters previously held 761–763 MiB of private committed memory each. Replacements reported 10.11–11.70 MiB each and recovered the same native-game rosters from disk. Inference and rollout continued on their original workers.

Five completed study coordinators had retained 4.35 GiB of private commit despite having no child process. Five retail collection supervisors also moved their numerical imports behind their readiness checks, releasing a further 5.05 GiB. Available system commit rose from 1.37 to over 14 GiB on the coordinator while physical runs continued. [Measured private bytes for the five completed coordinators](retired-completed-coordinators.json)

During this shared-cluster batch, TH17 returned 16 complete games in 1,101.85 seconds from its first to last real action (52.28 games/hour including gaps). The four private instances served overlapping studies. Within this batch, three instances played simultaneously for 0.71 s and four never did. [Per-lane activity and time intervals](th17-train-actual-rollout-throughput.json)
