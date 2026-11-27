# Six-title actor credit assignment from fresh full-stock games

Two six-game PPO policies started from the same six-game actor and critic. One uses actor GAE λ = 0.985 and one uses λ = 1. Both use the same out-of-fold clock baseline, value targets, 192 freshly played games, learning rate and training seeds. Each game contributed 16 Hard and 16 Lunatic full stocks. The full-return estimator was independently checked against direct discounted reward-to-go. TH08 collection was split between Gamer and Entropy. TH10 stocks and their final control boundaries were checked directly before both fits.

| Policy | Original PPO training games | Decisions | Accepted epochs | Final maximum title/rank KL |
|---|---:|---:|---:|---:|
| six-gae985 | 192 | 106,263 | 4 | 0.00611 |
| six-return | 192 | 106,263 | 4 | 0.00551 |

[Study registration](camera-protocol.json)

[six-gae985 weights](six-gae985/policy/policy.ts) · [Optimizer trace](six-gae985/policy/report.json)

[six-return weights](six-return/policy/policy.ts) · [Optimizer trace](six-return/policy/report.json)

[TH08 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/six-game-return-horizon-th08-exact-optimizer-input.zip)

[TH10 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/six-game-return-horizon-th10-exact-optimizer-input.zip)

[TH11 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/six-game-return-horizon-th11-exact-optimizer-input.zip)

[TH12 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/six-game-return-horizon-th12-exact-optimizer-input.zip)

[TH13 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/six-game-return-horizon-th13-exact-optimizer-input.zip)

[TH17 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/six-game-return-horizon-th17-exact-optimizer-input.zip)

[All 192 independent original-game evaluation results](all-six-original-results.md)
