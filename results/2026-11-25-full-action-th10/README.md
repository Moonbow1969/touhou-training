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
- [Frozen SFT evaluation, 12 credits](held-sft-native-traces.zip)
- [Frozen PPO1 evaluation, 12 credits](held-ppo1-native-traces.zip)
- [Frozen PPO2 evaluation, 12 credits](held-ppo2-native-traces.zip)

## Independent visual SFT pilot

The visual actor adds a frozen image-based player locator and uses a separately trained policy. Four additional held controller seeds compare this visual SFT with replay SFT on matched hosts, with the ordering reversed halfway through the pilot. Both arms reached stage 2 in 0/4 credits. SFT averaged 46.08 s and visual SFT averaged 39.31 s. Each lasted longer on two paired seeds. Both ended with zero mean settled Power. The pilot measures the combined checkpoint and visual representation. [Per-seed metrics](learned-locator-pilot.json) · [Paired figure](learned-locator-pilot.png) · [Frozen SFT traces](pilot-sft-native-traces.zip) · [Frozen visual SFT traces](pilot-visual-native-traces.zip)
