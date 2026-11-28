# TH06 Normal in the original 0.13 trial

The first zero-shot human SFT credit reached Rumia in Stage 1. The original Normal ranking recorded **631,260 points, Stage 1**. It ended at the first Continue prompt with zero spare lives and zero Bombs after 2,985 decisions. The complete recording runs for 298.85 seconds of original observation time. [Movie, trajectory and original result screens](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th06-original-trial013-normal-sft-first-receipt.json)

The actor received the 96 × 112 playfield and its difference from the preceding observation. Human replay supervision covered six other titles. It sampled nine-way movement, focus, shoot and Bomb. Its 1,487 movement decisions included all eight non-neutral directions. An independent control credit verified left, right, up, down and reduced focus speed from the sprite, shot appearance from playfield images and a Bomb use from the stock changing from three to two. The X server echoed the keys at every scored decision.

The original executable comes from ZUN's three-stage trial. The official Steam Classic release has a separate startup-only record. This experiment ran in one isolated Wine/Xvfb lane on Entropy. The game and the CPU actor exchanged images and key states locally. The first whole credit averaged 9.99 decisions per second. The median observation-to-key-echo interval was 14.72 ms, and the median full decision interval was 99.97 ms. [Author's game and trial download](https://www16.big.or.jp/~zun/html/th06.html) · [Prior Steam Classic preflight](https://github.com/Moonbow1969/touhou-training/tree/aeee7a30/results/2026-11-26-original-camera-training/th06-classic-normal-preflight)

The PPO study reserves four separate full-credit training seeds and four fresh paired seeds. It excludes the engineering credit from optimization. The original HUD supplies spare-life changes, effective Bomb spend and the final loss of the active life. A Bomb press with no resource change contributes zero. The game RNG is uncontrolled. Outcomes are counted at the original Continue screen, and the subsequent Normal ranking gives the stage and score. [Frozen training and evaluation protocol](ppo-protocol.json)

## First title-specific online PPO round

The optimizer used 11,755 on-policy decisions from four completed SFT credits. The 328 gradient updates produced a mean joint action KL of 0.00301 on the frozen training images. Across those images, 11,709 rewards were zero, 12 were life losses and 34 were effective Bomb costs. The four undiscounted returns ranged from −3.27 to −3.24. The four action heads and visual encoder all received gradients. [Training data, checkpoint and optimizer log](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th06-original-trial013-normal-first-online-ppo-four-train-receipt.json)

The four paired controller seeds each ran from the ordinary Normal Reimu A start through Game Over. Every original results screen recorded zero continues. The original highlighted ranking gives the maximum stage, and the separate original scorecard gives its progress percentage, score, misses and Bomb uses. [Full trajectories, source images and readout code](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th06-original-trial013-normal-first-online-ppo-four-held-pairs-receipt.json)

| Arm | Stage 2 reached | Mean original progress | Mean final spare lives | Mean final Bombs | Mean Bombs used |
|---|---:|---:|---:|---:|---:|
| SFT | 0/4 | 11.77% | 0.00 | 0.75 | 8.25 |
| PPO | 0/4 | 11.64% | 0.00 | 0.25 | 8.75 |

The paired change in source-defined progress was -0.12 percentage points for PPO. The descriptive 95% paired bootstrap interval spans -0.74 to +0.37 points. The four seeds control policy sampling. The original game RNG remained uncontrolled. [Paired outcomes](held-normal-results.csv) · [Native results figure](held-paired-native-outcomes.png) · [Original scorecards](held-native-ranking-evidence.jpg)

The first registered pair has a 4 minute 35 second side-by-side recording. The two clocks align at the first playable frame. Original full-color frames were retained every sixth decision, and the evidence pack retains the grayscale camera at every decision. This trial session displayed a refresh-rate warning when asked to export a native game replay. [Complete paired recording](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th06-trial013-normal-first-held-sft-ppo-pair.mp4)
