# Steam TH10 1.00a Normal from Game Start

All games start on Normal with Reimu A and two spare lives. The policies independently sample movement, focus, shooting and Bomb from original camera inputs every four advancing game ticks. In-game randomness is uncontrolled. Each terminal preserves the original Game Over image. Raw telemetry is read-only. Menu attempts without a verified opening have separate technical files; they contributed zero policy decisions.

| Policy and controller seed | Stages entered | Last native tick | Decisions | Native input agreement |
|---|---|---:|---:|---:|
| six-ppo-101001-3 | 1 | 3,136 | 784 | 100.00% |
| normal-ppo-101001 | 1 | 2,270 | 560 | 100.00% |
| normal-ppo-101002 | 1 | 3,889 | 973 | 100.00% |
| six-ppo-101002-retry3 | 1 | 3,325 | 832 | 100.00% |

[Verified recordings, native traces, training logs and code](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th10-steam-retail-normal-4-credits-with-traces-and-recordings.zip) · [First six-title PPO whole Game Over credit](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th10-steam-retail-normal-six-title-ppo-first-full-credit.mp4) · [Control and endpoint numbers](summary.json)
