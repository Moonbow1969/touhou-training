# Two-title PPO from the 68-stage human direction feedback model

The 68-stage human camera SFT learned a direction head conditioned on its preceding command. The full original TH10 and TH17 policy restored this actor with a new zero critic. Each work contributed 16 Hard and 16 Lunatic ordinary training starts. The independent evaluation compares the human parent and its PPO child in eight matched pairs per work and difficulty.

| Policy | Original PPO training games | Decisions | Accepted epochs | Final maximum title/rank KL |
|---|---:|---:|---:|---:|
| teacher-ppo | 64 | 28,717 | 4 | 0.00410 |

[Study registration](camera-protocol.json)

[teacher-ppo weights](policy/policy.ts) · [Optimizer trace](policy/report.json)

[TH10 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-step-feedback-online-th10-exact-optimizer-input.zip)

[TH17 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-step-feedback-online-th17-exact-optimizer-input.zip)
