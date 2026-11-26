# TH10 Normal story, full-credit evaluation

Each game starts with Reimu A on stage 1 in the original TH10 0.02a trial. The player gets two spare lives. All 36 primary credits ended when the game took the final active life. Each checkpoint used the same 12 controller seeds and the same host within each seed. The original game's RNG varied between launches. For the first six seeds SFT ran before PPO1. For the other six PPO1 ran first. PPO2 ran after both.

| Frozen policy | Stage 2 arrivals | Native Game Overs | Mean wall time to last life loss | Mean settled Power | Bomb-on decisions | Confirmed spirit strikes |
|---|---:|---:|---:|---:|---:|---:|
| Replay SFT | 0/12 | 12 | 37.61 s | 0.246 | 9 | 1 |
| PPO update 1 | 0/12 | 12 | 42.99 s | 0.092 | 9 | 1 |
| PPO update 2 | 0/12 | 12 | 36.39 s | 0.133 | 6 | 0 |

The game recorded −1 terminal spares for every run. Stage 2 entry stocks are missing for every policy because all runs ended in stage 1. Across the 12 paired controller seeds, PPO1 changed mean survival by +5.38 s relative to SFT (95% paired bootstrap interval −2.25 to +12.56 s). PPO2 changed it by −1.22 s (−6.29 to +4.59 s). Game speed and system load affect wall time. The progression result is unchanged across both updates. [Per-seed metrics and uncertainty](sft-ppo1-ppo2-full-credits.json) · [Paired figure](sft-ppo1-ppo2-full-credits.png)

TH10 charges 1.00 Power per successful spirit strike. A sampled Bomb with no available Power, a held key during an existing strike, or a death that deducts Power can all look like a Bomb event in an action-only log. We count an activation when a native X input coincides with a one-unit Power spend on a stable life. Separate original HUD reads confirm the calibration and distinguish Power before the last loss from settled Game Over Power. Of 36 life losses per policy, the player held at least 1.00 Power immediately before 11 SFT deaths, 17 PPO1 deaths and 15 PPO2 deaths. All 43 lacked a Bomb request in the preceding half-second. These observations measure action selection. They do not establish that a Bomb could have saved every life. [Calibration images](native-bomb-calibration.jpg) · [Settled HUD readings](settled-sft-power.jpg) · [Conditional Bomb probabilities](bomb-propensity.json)

## Inputs and trace archives

Each rollout actor uses only the 96 × 112 grayscale playfield and its own action history. It samples movement from nine directions including neutral, and independently samples Focus, Shoot and Bomb. Actions remain active for a requested 120 ms between observations. A separate read-only native observer measures game state and key consumption. Every native death and stage transition enters the log. The game runs continuously through respawns. Geometric avoidance, aim correction and automatic Bomb are disabled.

Each ZIP contains one folder per controller seed with the four sampled actions, their behavior log probabilities, value estimates, full input features, terminal flag and separated reward components in `behavior.npz`. Timestamped inventory, coordinates and native key state are in `native.csv.gz`. The JSON receipt records the hashes of the executable, policy, behavior and terminal screenshot. Each recorded native life change contributes its signed delta, a verified stage advance gives +2, survival gives +0.001 per second of valid play, and a confirmed Power-paid strike costs 0.03. Full-resolution game assets are supplied by the local game installation.

- [Original SFT training rollouts, 4 credits](train-sft-native-traces.zip)
- [First PPO training rollouts, 8 new credits](train-ppo1-native-traces.zip)
- [Second PPO training rollouts, 8 new credits](train-ppo2-native-traces.zip)
- [Frozen SFT evaluation, 12 credits](held-sft-native-traces.zip)
- [Frozen PPO1 evaluation, 12 credits](held-ppo1-native-traces.zip)
- [Frozen PPO2 evaluation, 12 credits](held-ppo2-native-traces.zip)

## Independent visual SFT pilot

The visual actor adds a frozen image-based player locator and uses a separately trained policy. Four additional held controller seeds compare this visual SFT with replay SFT on matched hosts, with the ordering reversed halfway through the pilot. Both arms reached stage 2 in 0/4 credits. SFT averaged 46.08 s and visual SFT averaged 39.31 s. Each lasted longer on two paired seeds. Both ended with zero mean settled Power. The pilot measures the combined checkpoint and visual representation. [Per-seed metrics](learned-locator-pilot.json) · [Paired figure](learned-locator-pilot.png) · [Frozen SFT traces](pilot-sft-native-traces.zip) · [Frozen visual SFT traces](pilot-visual-native-traces.zip)

## Projectile visibility

Eight fresh PPO2 training credits supplied 2,455 valid paired camera and native projectile observations. The latest native read preceded its matching capture by a median of 20.8 ms and a maximum of 43.9 ms. A convolutional detector with 4,993 parameters learned from five whole credits, selected its epoch and detection threshold on a sixth, and was evaluated on the remaining two. Across 552 test frames with 3,095 visible projectile instances, it matched 1,191 within 12 native game pixels. Precision was 0.340, recall was 0.385 and F1 was 0.361. A fixed spatial prior learned from training positions scored 0.014 F1. The individual test-credit F1 values were 0.362 and 0.360. [Label and alignment manifest](train-ppo2-paired-manifest.json) · [Validation history and test metrics](bullet-visibility.json) · [Model weights](bullet-visibility.pt)

The same frozen detector recalled 129 of 386 bullets within 64 native pixels of the player, with 0.374 precision and 0.334 recall in that region. Each example uses the first valid frame with ten or more native bullets on its held credit. [Camera, detector response and matched spatial annotations](bullet-visibility.png)

## Representative complete credits

The video aligns the three frozen policies on the first registered primary seed, 711001, and plays at measured wall speed. Each column uses that policy's actual saved camera stream and native inventory, then shows the original color Game Over screenshot. The game camera runs at approximately eight observations per second. [Watch the three complete credits](TH10-Normal-711001-3-policies.mp4) · [Video timing and source hashes](TH10-Normal-711001-3-policies.json)

## Rollout throughput

The 12-credit fresh PPO2 evaluation cohort ran on three independent lanes on each of two hosts. From the first claim to the final sealed archive, the batch took 164.05 seconds and sampled 4,159 policy decisions, or 25.35 decisions per wall second. Median per-credit setup time was 23.62 seconds, active play was 39.64 seconds and final capture and packaging took 2.55 seconds. The compressed capture payload across all 12 credits was 41.25 MiB. The per-credit timing file records each original host's timestamps and its clock-synchronization assumption. [Timing and transport measurements](throughput-fresh-ppo2.json)
