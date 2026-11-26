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

An independent Accelerator lane kept Xvfb, Wine and audio alive between credits. Each reset started a new original-game process in a fresh writable directory. The first cold launch took 22.115 seconds to verified play. The two warm launches took 11.330 and 11.279 seconds. All three entered Normal stage 1 as Reimu A at (0, 400) with two spares, zero Power, zero score and released keys. The runs lasted 259, 431 and 432 policy decisions before their native Game Overs. This qualification uses three diagnostic seeds on one machine. [Fresh-state receipts and earlier rejected attempts](warm-reset.json) · [Original entry and Game Over screens](warm-reset.jpg)

## Additional frozen comparisons

One further on-policy PPO update used new whole-credit training data from three games. Its TH10 comparison used another 12 held controller seeds per policy. Stage 2 remained 0/12 for PPO2 and 0/12 for PPO3. Mean survival was 41.97 seconds for PPO2 and 36.49 seconds for PPO3. The paired difference was −5.48 seconds (95% resampled interval −12.66 to +1.25). The game recorded one successful Bomb with PPO2 and zero with PPO3. [Per-credit stocks and effects](follow-ppo2-ppo3.json) · [Paired figure](follow-ppo2-ppo3.png) · [PPO2 logs](follow-held-ppo2-native-traces.zip) · [PPO3 logs](follow-held-ppo3-native-traces.zip)

A separate six-seed comparison added a future-life-loss auxiliary objective to PPO2 using the same training trajectories. Both checkpoints reached stage 2 in 0/6 credits. Mean survival changed from 42.29 to 46.05 seconds (paired difference +3.76, interval −0.16 to +8.50). Both used one successful Bomb, and all 12 Game Overs had zero settled Power. The policies chose the same four-part action on 2,056 of 2,090 steps within matched credit lengths. Four matched pairs agreed on every sampled action within their common length. [Protocol](aux-eval-protocol.json) · [Outcomes](aux-ppo2-vs-ppo2-aux.json) · [Figure](aux-ppo2-vs-ppo2-aux.png) · [Per-seed action agreement](aux-realized-actions.json)

## Privileged teacher sensitivity

The frozen four-head teacher used by the portable SFT had learned from original human replay states in TH13, TH14 and TH17. On 1,197 TH10 PPO2 training observations, we swapped each nonempty projectile set with a different set from the same credit, spare count and exact bullet count, holding player position and motion history fixed. This changed the most probable movement on 12 observations (1.00%). Movement probabilities shifted by 0.050 mean total variation. The most probable Bomb choice was unchanged throughout. Repeating the same swaps with motion history disabled changed movement on 156 observations (13.03%); this measures dependence under a missing-history intervention. The archive supplies position and velocity for up to 96 native bullets but lacks per-bullet radius, a native stage-frame counter and boss state. The audit includes imputation sensitivity and per-credit coverage. [Full channel mapping and controlled results](privileged-teacher-sensitivity.json)

On the teacher's original held human replays, masking past movement reduced accuracy from 65.97% to 39.51% across 94,432 decisions. For the 28,966 four-tick direction changes, top-choice accuracy rose from 15.74% to 26.30% while log loss worsened from 2.123 to 2.202. The masked teacher favored neutral. It identified 83.0% of stops compared with 27.4% with history, but only 5.4% of moving-to-moving turns compared with 13.7% with history. For 7,848 stationary-to-movement onsets, accuracy fell from 7.57% to 4.78%, while log loss improved from 2.656 to 2.393. These are correlated decisions from four entire held replays across three games. [Per-title held counts and log losses](held-human-teacher-history.json)

Two additional SFT fits used identical seed, architecture, optimizer budget, replay shards and training draws. One masked motion history in 30% of training examples. Both received full history at evaluation. The train-time dropout changed overall movement accuracy from 65.98% to 65.38%, stop accuracy from 27.71% to 31.51%, and movement onset accuracy from 7.52% to 6.21%. The complete training curves and both frozen weights are available. [Matched training results](human-history-dropout.json) · [Comparison figure](human-history-comparison.png) · [Control checkpoint](human-history-control.pt) · [Dropout checkpoint](human-history-dropout30.pt)
