# Two-title PPO from the 68-stage human direction feedback model

In eight paired TH10 Hard campaigns the parent reached mean Stage 1 tick 1,726 and PPO reached 2,110.5. The eight PPO-minus-parent differences were +638, +1,272, +837, +435, −156, −80, +53 and +77 ticks. The mean difference was +384.5 ticks (+22.3%), with paired bootstrap interval +77.75 to +727.75. Six of eight pairs favored PPO. A two-sided exact paired sign test gave p=.289. TH10 Lunatic had five wins and three losses, mean +91 ticks, interval −271.75 to +429.75 and sign p=.727. Both ranks and actors had 0/8 Stage 2. The experiment alternated actor order on the same machine. Native game RNG was uncontrolled. [Per-seed differences and order sensitivity](native10/paired-sign-sensitivity.json) · [TH10 all resources, both difficulties](native10/ranked-native-comparison.json)

The 68-stage human camera SFT learned a direction head conditioned on its preceding command. The full original TH10 and TH17 policy restored this actor with a new zero critic. Each work contributed 16 Hard and 16 Lunatic ordinary training starts. The independent evaluation compares the human parent and its PPO child in eight matched pairs per work and difficulty.

| Policy | Original PPO training games | Decisions | Accepted epochs | Final maximum title/rank KL |
|---|---:|---:|---:|---:|
| teacher-ppo | 64 | 28,717 | 4 | 0.00410 |

[Study registration](camera-protocol.json)

[teacher-ppo weights](policy/policy.ts) · [Optimizer trace](policy/report.json)

[TH10 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-step-feedback-online-th10-exact-optimizer-input.zip)

[TH17 source optimizer inputs](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/four-step-feedback-online-th17-exact-optimizer-input.zip)
