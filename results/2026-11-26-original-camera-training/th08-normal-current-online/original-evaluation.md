# TH08 Normal original-game SFT and PPO comparison

Both policies played the official 0.02 trial on Entropy using Ghost Team, two spare lives and three Bombs at entry. Each credit ran to its first original Continue prompt. The evaluator released the keys, declined the continue and recorded the original ranking screen. All four action heads controlled the game, including focus, shot and Bomb.

Four independent whole credits provided 2,364 decisions for PPO. Their controller seeds were 80960101–80960104. The two held controller seeds below had the arm order reversed on the same physical lane. The original game RNG was uncontrolled.

| Controller seed | Policy | Final stage | Spare lives | Bombs | Active seconds | Decisions | Median input delay, ms |
|---:|---|---:|---:|---:|---:|---:|---:|
| 80960251 | Online PPO | 1 | 0 | 3 | 55.54 | 528 | 16.59 |
| 80960251 | Human SFT | 1 | 0 | 3 | 56.12 | 528 | 17.25 |
| 80960252 | Online PPO | 1 | 0 | 3 | 68.64 | 664 | 15.29 |
| 80960252 | Human SFT | 1 | 0 | 3 | 64.82 | 633 | 14.47 |

Human SFT reached Stage 2 in 0/2 games, with 0 unknown endpoints. Mean active duration was 60.47 s. The median of the two per-game median action intervals was 102.47 ms. Re-running the frozen actor on stored observations reproduced sampled log probabilities to within 3.81e-06.

Online PPO reached Stage 2 in 0/2 games, with 0 unknown endpoints. Mean active duration was 62.09 s. The median of the two per-game median action intervals was 102.51 ms. Re-running the frozen actor on stored observations reproduced sampled log probabilities to within 3.81e-06.

Two paired controller seeds give a pilot estimate. Each score row retains the original game frames, key logs and SHA-256 receipts. The report includes evidence for independently reviewed stage readings.

Complete training game recordings, actions and terminal evidence are available in the [four-credit training archive](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-normal-current-online-four-on-policy-normal-originals.zip).

The [four-credit evaluation archive](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th08-normal-current-online-two-pairs-sft-vs-ppo-original.zip) contains both held pairs and the per-file checksums. The [per-credit results](native-paired-results.json) include the decision timing, resource counts and physical inputs.
