# Original full-action PPO from the 68-stage human visual SFT

The deployed human camera SFT had trained on 68 original replay stages and 231,511 available decisions. It played eight fresh ordinary Hard and Lunatic credits on each of TH10 and TH11. The two-title PPO child restored all supervised policy weights and started with a zero value head. Both source and child control movement, focus, shooting and Bombs. The held test uses eight pairs per difficulty and title, 64 new complete original games.

| Policy | Original PPO training games | Decisions | Accepted epochs | Final maximum title/rank KL |
|---|---:|---:|---:|---:|
| later-ppo | 16 | 3,496 | 4 | 0.00326 |

The 64 independent full-stock tests ended at Game Over. Pure SFT reached Stage 2 in 0/32 and its PPO child in 0/32. All four paired wall-duration intervals include zero. TH10 Hard mean final Power fell from 0.181 to 0.050 and Lunatic rose from 0.081 to 0.331. TH11 resets Power after the final loss. [All four game and difficulty cells, input counts and films](evaluation/README.md) · [Matched duration intervals](evaluation/paired-by-difficulty.json)

[Study registration](camera-protocol.json)

[later-ppo weights](policy/policy.ts) · [Optimizer trace](policy/report.json)

[TH10 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/later-human-online-th10-exact-optimizer-input.zip)

[TH11 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/later-human-online-th11-exact-optimizer-input.zip)
