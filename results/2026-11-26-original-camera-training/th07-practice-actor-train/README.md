# TH07 Normal practice training

Twenty preassigned training originals ran in retail TH07 1.00b using official thprac 2.3.1.1, Normal Reimu A and initial Power 128. The frozen four-head policy saw the 96 × 112 original camera and its previous-frame difference. Each stock began at a verified practice section and ended at native Game Over or a playable cutoff after 6,000 native ticks. Every original used zero continues.

| Native practice start | Originals | Eligible for PPO | Median Stage 1 ticks | Game Over | Playable at 6,000 ticks |
| --- | ---: | ---: | ---: | ---: | ---: |
| Letty nonspell 1, 2 spares / 3 Bombs | 10 | 5 | 3,920 | 10 | 0 |
| Letty nonspell 1, 0 spares / 1 Bomb | 5 | 4 | 1,021 | 5 | 0 |
| Cirno midboss nonspell 1, 2 spares / 3 Bombs | 5 | 4 | 5,959 | 3 | 2 |

The game consumed all 18,303 sampled requests. Of 18,296 comparable windows, 18,287 had exact four-tick control. Seven credits failed the prespecified control gate, leaving 13 credits and 11,108 decisions eligible for PPO. Five windows held the preceding command for eight ticks. Four others received the preceding input on the first tick and the current input on the remaining three. The original raw sources include all technical exclusions. Seed 21749201 entered native Game Over exactly at the 6,000-tick limit. The original retry flag, exhausted stock and three misses establish the terminal.

No run observed practice stage clear. These starts cannot measure ordinary Game Start Stage 2 arrival. An additional Accelerator setup stopped at the title menu with zero actor decisions and zero gameplay. Its images and signed disposition are preserved separately. The real gameplay for that preassigned seed subsequently ran on Gamer. The production actor used isolated X displays and disjoint CPU sets on the same host after initial qualification. The controller seed controls action sampling. The original game RNG is unsynchronized.

[Per-original stock, exposure and control counts](qa.csv) · [Timing exceptions](native-timing-exceptions.json) · [Physical overlap and observer latency](physical-throughput.json) · [Physical versus PPO counts](original-summary.json).

The full training-stock movie displays 1,500 captured policy observations with original stock, sampled controls, and the real terminal screen. Playback is 15 observations per second. [Watch the full Letty practice stock](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-letty-normal-full-6000-native-ticks.mp4).

[Source part 1](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-normal-practice-train-source-01.zip) · [Part 2](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-normal-practice-train-source-02.zip) · [Part 3](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-normal-practice-train-source-03.zip) · [Part 4](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-normal-practice-train-source-04.zip) · [Verified byte counts and hashes](published-sources.json).
