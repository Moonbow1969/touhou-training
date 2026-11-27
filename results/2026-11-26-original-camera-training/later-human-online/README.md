# Original full-action PPO from the 68-stage human visual SFT

The deployed human camera SFT had trained on 68 original replay stages and 231,511 available decisions. It played eight fresh ordinary Hard and Lunatic credits on each of TH10 and TH11. The two-title PPO child restored all supervised policy weights and started with a zero value head. Both source and child control movement, focus, shooting and Bombs. The held test uses eight pairs per difficulty and title, 64 new complete original games.

| Policy | Original PPO training games | Decisions | Accepted epochs | Final maximum title/rank KL |
|---|---:|---:|---:|---:|
| later-ppo | 16 | 3,496 | 4 | 0.00326 |

[Study registration](camera-protocol.json)

[later-ppo weights](policy/policy.ts) · [Optimizer trace](policy/report.json)

[TH10 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/later-human-online-th10-exact-optimizer-input.zip)

[TH11 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/later-human-online-th11-exact-optimizer-input.zip)
