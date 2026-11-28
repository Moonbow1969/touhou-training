# TH08 dual-resolution original training data

32 ordinary Normal Ghost Team credits on the official TH08 0.02 trial produced 30,833 decisions.
The 96×112 four-head PPO selected every action. A separate CPU captured the same playfield at 192×224. Both lanes ran in separate processes on the same physical host.

| Outcome | Count |
|---|---:|
| Qualified full Stage 1 credits | 27 |
| Qualified decisions | 26,354 |
| Qualified native hit onsets followed by stock loss or Game Over | 81 |
| Reached Stage 2 | 0 |
| Confirmed original Game Over in Stage 1 | 32 |
| Unknown terminal stage | 0 |

The final report retains every registered physical outcome. Each `rows` record in the [manifest](qualified-train.json) selects its normalized 96×112 training trajectory via `file`. In the matching `highres` NPZ, `gray[original_observation_indices]` selects the original 224×192 luminance observations corresponding to its retained actions. The remaining high-resolution frames document transitions and the actual terminal. The observed raw policy action stream, log probabilities, nonzero critic, stock readings and native ticks are also preserved. All gameplay inputs came from the frozen learned policy, without continues.

The original source ZIP under `archive` contains the game-consumed word archive, clock.tsv, gates.tsv, timing.jsonl, native HUD observations, and full-sized menu and terminal photographs. To reconstruct a local dataset, download all four release artifacts for each original and place them at the respective `file`, `highres`, `archive`, and `physical_behavior` paths from the manifest or per-credit selection. Verify each SHA256. The [release index](published-sources.json) records every downloadable file and its size.

[Original registration](registration.json) · [Guard originals](first-two-intake.json) · [Native outcome summary](original-collection-summary.json)

The two workers collected 51.72 whole credits per hour including startup and transfer. Of 32 physical Game Overs, all ended in Stage 1, with zero remaining spares and a mean of 1.375 Bomb units. Registered on-policy learning uses 27 of those credits. The five withheld sources and their actuator or terminal-attribution incidents remain in the published data. The 128 original assets total 2,191,605,452 bytes. Sampled peak study storage across collection and staging was 2,301,404,855 bytes against a 2,500,000,000-byte budget. [Throughput](original-throughput.json) · [Storage](storage-report.json)

The paired crops contain genuine fine detail. Out of 331,516,416 aligned policy pixels, 83.39% had nonconstant high-resolution subpixels and 8.82% had a within-cell contrast of at least 32 gray levels. Four policy pixels differed by 2 levels after shrinking the high-resolution images, and all others agreed within 1. The low-resolution camera averages a weighted integer sum of the original RGB pixels directly. The high-resolution camera rounds an area-resized RGB image before converting it to gray, then the alignment check takes a second rounded average. [Independent pixel audit](dual-camera-spatial-detail.json)
