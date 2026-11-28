# TH07 retail Normal, parent vs Focus/Bomb-preserving human SFT

Fourteen ordinary Reimu A Game Starts reached the original Game Over with zero continues. Both policies reached Stage 2 in 0 of 7 attempts. The candidate spent 55 Bombs across its seven credits, the same total as the parent. Two pairs passed every preregistered technical check; the prespecified minimum was six. The candidate remains unpromoted.

| Primary measure | Parent | SFT, seed 71281 |
|---|---:|---:|
| Admitted full credits | 2 | 2 |
| Total Stage 1 native ticks | 11,869 | 14,910 |
| Mean Stage 1 native ticks per credit | 5,934.5 | 7,455.0 |
| Stage 1 deaths per 10,000 native ticks | 5.055 | 4.024 |
| Actual Bombs per credit | 8.5 | 8.0 |
| Actual Bombs per 10,000 Stage 1 ticks | 14.32 | 10.73 |
| Focus held during observed playable ticks | 47.83% | 49.64% |
| Mean spare lives / Bombs at Game Over | 0 / 0 | 0 / 0 |

The candidate-to-parent death rate ratio was **0.796**. The registered promotion cutoff was 0.75, with at least 30,000 Stage 1 ticks per arm. The Bomb-use ratio of 0.941 met the registered range of 0.7–1.4. One prior-to-death Bomb stock was positive out of six deaths in each admitted arm. Each pair shared its physical CPU slot and policy-sampling seed. The original game drew independent randomness. These sample sizes cannot establish a survival benefit.

| Controller seed | Stage 1 ticks, parent / SFT | Actual Bombs, parent / SFT | Primary pair disposition |
|---|---:|---:|---|
| 2172751 | 5,070 / 5,289 | 8 / 9 | Parent had one redundant, internally inconsistent observer read |
| 2172752 | 6,787 / 6,381 | 8 / 7 | Included |
| 2172753 | 6,622 / 8,083 | 9 / 9 | SFT had one redundant, internally inconsistent observer read |
| 2172754 | 5,082 / 8,529 | 9 / 9 | Included |
| 2172755 | 6,821 / 8,355 | 9 / 9 | SFT observer exceeded 100 ms, missed two scheduled decisions and lost independent native evidence for one applied window |
| 2172756 | 4,926 / 4,820 | 8 / 7 | Parent observer exceeded 100 ms |
| 2172771, reserve | 4,186 / 5,010 | 4 / 5 | SFT score at the last native sample was 625,510; the official score card and saved replay both gave 625,830 |

The third candidate technical failure stopped the study under its registered per-arm limit. No exposure extension ran. Every attempted credit contributed its original score card, official saved replay, input and policy-camera records to a signed release. Rows in the table retain the original-game endpoint even where the primary measurement failed.

An independent sensitivity analysis checked every attempt with a narrow rule for redundant reads straddling two ticks. Both adjacent ticks needed intact physical input evidence, stable stock and no affected decision. The official RPY still had to prove every retained control window. Two previously excluded pairs qualified under this rule. Across those four pairs, mean Stage 1 ticks were 5,890.25 and 7,070.50, and the death rate ratio was **0.833**. Both arms spent 34 Bombs and reached Stage 2 zero times. A regression test removed all physical input evidence for one other four-tick window while leaving the original RPY intact; the independent verifier rejected that window.

The first pair with both arms technically accepted is available at measured wall speed. It shows the actual policy camera, original opening display, resources, four action heads and terminal score cards. [Matched full-credit recording](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-action-preserving-2172752.mp4).

The directory contains the [registration](protocol.json), [primary scores](result.json), [independent sensitivity scores](sensitivity-summary.json), [all original endpoints](all-original-endpoints.json), and [negative control](duplicate-fixture.json). Seven per-pair release receipts record 171,710,422 bytes of source evidence with whole-archive and per-file SHA-256 digests. The signed comparison uses the current campaign exclusively.
