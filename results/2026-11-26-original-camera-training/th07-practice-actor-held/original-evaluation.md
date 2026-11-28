# TH07 Normal Letty practice, matched original evaluation

The frozen parent and the practice-trained four-head PPO each started 20 fresh official TH07 Normal Reimu A Letty nonspell 1 stocks, with 2 spare lives, 3 Bombs, 128 Power and no continues. Each registered controller seed played both arms on the same physical host and private instance configuration. Each lane had five parent-first and five candidate-first pairs. Original game RNG was independent.

| Physical measurement | Frozen parent | Practice PPO |
| --- | ---: | ---: |
| Original starts | 20 | 20 |
| Confirmed deaths / playable native ticks | 55 / 95,727 | 53 / 92,553 |
| Deaths per 1,000 playable ticks | 0.575 | 0.573 |
| Median ticks until first miss | 1,390 | 1,860 |
| Stocks still playable at the 6,000-tick limit | 5 | 6 |
| Native Game Over | 15 | 14 |
| Confirmed practice clear | 0 | 0 |
| Mean remaining Bombs at endpoint | 0.8 | 0.6 |
| Native Bomb uses across all stocks | 138 | 130 |

The death-rate ratio across all 20 matched original pairs was 0.997 with a 95% paired bootstrap interval of 0.788–1.247. Twenty-four of forty stocks passed all prespecified control checks. Comparing the eight pairs for which both arms passed gave 0.995 (0.653–1.369). Censoring exposure at the first nonexact control window gave 0.967 (0.722–1.266). Exposure includes invulnerability time. Both policies had a positive Bomb balance immediately before their first death on 9/20 stocks. This experiment tests one Stage 1 practice start. Ordinary Game Start progression requires a separate test.

[All 40 original stock records](physical-stocks.csv) · [Independent scoring and intervals](independent-held-results.json) · [Action counts](action-distribution.json) · [Independent survival and rate plots](independent-held-figure.png).

[Watch the oldest registered pair side by side](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-first-matched-letty-stock.mp4). It uses the actual 96 × 112 policy images sampled every four native ticks, displays stocks and controls, then shows each real terminal screen.

[Original sources 1–5](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-normal-letty-held-sources-01.zip) · [6–10](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-normal-letty-held-sources-02.zip) · [11–15](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-normal-letty-held-sources-03.zip) · [16–20](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-normal-letty-held-sources-04.zip) · [Verified remote digests and byte counts](published-sources.json).
