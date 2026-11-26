# Original Normal full-credit benchmark

Each entry is one credit from the Stage 1 start, with the original two reserve lives. The four-head policy controls direction, focus, shooting and Bombs throughout. Each pair uses the same controller seed on the same host. The original game RNG remains uncontrolled. Endpoint images and per-step action traces are indexed by unique credit.

## Primary matched cohorts

| Title / reference → candidate | Stage 2 | Mean time to credit end (s) | Paired Δs [95% interval] | Final reserve lives | Mean final resource | Confirmed Bomb casts (total) |
|---|---:|---:|---:|---:|---|---:|
| TH10 · SFT → PPO1 | 0/12 → 0/12 | 37.61 / 42.99 | +5.38 [-2.24, +12.55] | 0.0 / 0.0 | 0.2458 / 0.0917 Power | 1 / 1 |
| TH12 · SFT → PPO1 | 0/12 → 0/12 | 63.68 / 68.44 | +4.76 [-8.09, +17.28] | 0.0 / 0.0 | 1.750 / 1.667 Bombs | 15 / 14 |
| TH13 · SFT → PPO1 | 0/12 → 0/12 | 211.15 / 220.10 | +8.95 [-55.65, +72.76] | 0.0 / 0.0 | 1.250 / 1.417 Bombs | 29 / 31 |
| TH10 · SFT → PPO2 | 0/12 → 0/12 | 37.61 / 36.39 | -1.22 [-6.35, +4.22] | 0.0 / 0.0 | 0.2458 / 0.1333 Power | 1 / 0 |
| TH12 · SFT → PPO2 | 0/12 → 0/12 | 48.38 / 47.33 | -1.05 [-7.60, +5.79] | 0.0 / 0.0 | 1.667 / 1.667 Bombs | 11 / 9 |
| TH13 · SFT → PPO2 | 0/12 → 0/12 | 211.15 / 277.36 | +66.21 [+21.09, +114.08] | 0.0 / 0.0 | 1.250 / 1.083 Bombs | 29 / 38 |

120 verified credits, 85,765 decisions. The paired bootstrap resamples controller seeds within each host, preserving the assigned host mix. TH12 SFT/PPO1 ran on Steam Deck. TH12 SFT/PPO2 ran independently on Entropy. TH10 and TH13 reuse their SFT observations across those two comparisons.

[Paired figure](evaluation/paired-full-credits.png) · [Every original credit](evaluation/credits.json) · [Verification totals](evaluation/verified.json)

## Independent third-update comparison

| Title / reference → candidate | Stage 2 | Mean time to credit end (s) | Paired Δs [95% interval] | Final reserve lives | Mean final resource | Confirmed Bomb casts (total) |
|---|---:|---:|---:|---:|---|---:|
| TH10 · PPO2 → PPO3 | 0/12 → 0/12 | 41.97 / 36.49 | -5.48 [-12.38, +1.04] | 0.0 / 0.0 | 0.0500 / 0.0125 Power | 1 / 0 |
| TH12 · PPO2 → PPO3 | 0/12 → 0/12 | 58.80 / 52.64 | -6.16 [-15.03, +0.22] | 0.0 / 0.0 | 1.667 / 1.917 Bombs | 12 / 9 |
| TH13 · PPO2 → PPO3 | 0/12 → 0/12 | 190.32 / 170.17 | -20.15 [-68.56, +28.28] | 0.0 / 0.0 | 1.083 / 1.583 Bombs | 23 / 22 |

72 verified credits, 47,663 decisions. The paired bootstrap resamples controller seeds within each host, preserving the assigned host mix. Do not combine these samples with the earlier cohort to estimate a within-seed change.

[Paired figure](followup/paired-full-credits.png) · [Every original credit](followup/credits.json) · [Verification totals](followup/verified.json)

## Exploratory visual-loss auxiliary comparison

| Title / reference → candidate | Stage 2 | Mean time to credit end (s) | Paired Δs [95% interval] | Final reserve lives | Mean final resource | Confirmed Bomb casts (total) |
|---|---:|---:|---:|---:|---|---:|
| TH10 · PPO2 → PPO2-AUX | 0/6 → 0/6 | 42.29 / 46.05 | +3.76 [-0.17, +8.50] | 0.0 / 0.0 | 0.0000 / 0.0000 Power | 1 / 1 |
| TH12 · PPO2 → PPO2-AUX | 0/6 → 0/6 | 51.25 / 59.17 | +7.92 [-4.47, +19.04] | 0.0 / 0.0 | 1.833 / 1.833 Bombs | 3 / 3 |

24 verified credits, 9,692 decisions. The paired bootstrap resamples controller seeds within each host, preserving the assigned host mix. Do not combine these samples with the earlier cohort to estimate a within-seed change.

[Paired figure](aux-evaluation/paired-full-credits.png) · [Every original credit](aux-evaluation/credits.json) · [Verification totals](aux-evaluation/verified.json)

All observed credits ended in Stage 1. Resource balances on entry to Stage 2 are therefore undefined. An arm with 0/12 has a two-sided Wilson 95% interval of 0–24.25%. For 0/6 it is 0–39.03%. TH10 spends Power to cast Bombs. Final TH10 Power is read on the settled terminal HUD, after the game's last-life transition. TH12 and TH13 report discrete Bomb stock.

The sampled game versions were the official TH10, TH12 and TH13 trials. These experiments do not establish full retail-story completion. TH13 optional manual Trance is outside the shared action vocabulary. The human SFT uses eight-tick action labels. TH13 ordinarily observes four native stage ticks per action. TH12 uses 120 ms wall holds; measured renderer speeds differ by host.
