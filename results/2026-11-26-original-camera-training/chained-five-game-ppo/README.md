# Second consecutive full-action five-title PPO update from human SFT

The second five-title on-policy update restored the actor and critic fitted on the first batch. The 84-stage supervised actor is their shared ancestor. All five games supplied fresh Hard and Lunatic full-stock starts. The registered comparison uses generation one against generation two on separate complete original games.

| Policy | Original PPO training games | Decisions | Accepted epochs | Final maximum title/rank KL |
|---|---:|---:|---:|---:|
| reviewed-ppo2 | 80 | 37,287 | 4 | 0.00481 |

[Study registration](camera-protocol.json)

[reviewed-ppo2 weights](policy/policy.ts) · [Optimizer trace](policy/report.json)

[TH10 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/chained-five-game-ppo-th10-exact-optimizer-input.zip)

[TH11 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/chained-five-game-ppo-th11-exact-optimizer-input.zip)

[TH12 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/chained-five-game-ppo-th12-exact-optimizer-input.zip)

[TH13 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/chained-five-game-ppo-th13-exact-optimizer-input.zip)

[TH17 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/chained-five-game-ppo-th17-exact-optimizer-input.zip)
