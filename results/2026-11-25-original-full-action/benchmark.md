# Original Normal full-credit benchmark

Each entry is one credit from the Stage 1 start, with the original two reserve lives. The four-head policy controls direction, focus, shooting and Bombs throughout. Each pair uses the same controller seed on the same host. The original game RNG remains uncontrolled. Endpoint images and per-step action traces are indexed by unique credit.

## Primary matched cohorts

| Title / reference → candidate | Stage 2 | Mean time to credit end (s) | Paired Δs [95% interval] | Final reserve lives | Mean final resource | Confirmed Bomb casts (total) |
|---|---:|---:|---:|---:|---|---:|
| TH10 · SFT → PPO1 | 0/12 → 0/12 | 37.61 / 42.99 | +5.38 [-2.41, +12.68] | 0.0 / 0.0 | 0.246 / 0.092 Power | 1 / 1 |
| TH12 · SFT → PPO1 | 0/12 → 0/12 | 63.68 / 68.44 | +4.76 [-8.45, +17.27] | 0.0 / 0.0 | 1.750 / 1.667 Bombs | 15 / 14 |
| TH13 · SFT → PPO1 | 0/12 → 0/12 | 211.15 / 220.10 | +8.95 [-55.37, +72.48] | 0.0 / 0.0 | 1.250 / 1.417 Bombs | 29 / 31 |
| TH10 · SFT → PPO2 | 0/12 → 0/12 | 37.61 / 36.39 | -1.22 [-6.43, +4.68] | 0.0 / 0.0 | 0.246 / 0.133 Power | 1 / 0 |
| TH12 · SFT → PPO2 | 0/12 → 0/12 | 48.38 / 47.33 | -1.05 [-7.64, +5.77] | 0.0 / 0.0 | 1.667 / 1.667 Bombs | 11 / 9 |
| TH13 · SFT → PPO2 | 0/12 → 0/12 | 211.15 / 277.36 | +66.21 [+20.18, +114.34] | 0.0 / 0.0 | 1.250 / 1.083 Bombs | 29 / 38 |

120 verified credits, 85,765 decisions. The paired bootstrap interval resamples controller seeds. TH12 SFT/PPO1 ran on Steam Deck. TH12 SFT/PPO2 ran independently on Entropy. TH10 and TH13 reuse their SFT observations across those two comparisons.

[Paired figure](evaluation/paired-full-credits.png) · [Every original credit](evaluation/credits.json) · [Verification totals](evaluation/verified.json)

All observed credits ended in Stage 1. Resource balances on entry to Stage 2 are therefore undefined. An arm with 0/12 has a two-sided Wilson 95% interval of 0–24.25%. For 0/6 it is 0–39.03%. TH10 spends Power to cast Bombs. Final TH10 Power is read on the settled terminal HUD, after the game's last-life transition. TH12 and TH13 report discrete Bomb stock.

The sampled game versions were the official TH10, TH12 and TH13 trials. These experiments do not establish full retail-story completion. TH13 optional manual Trance is outside the shared action vocabulary. The human SFT uses eight-tick action labels. TH13 ordinarily observes four native stage ticks per action. TH12 uses 120 ms wall holds; measured renderer speeds differ by host.
