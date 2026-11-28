# Perfect Cherry Blossom 1.00b human camera bank

The original Japanese retail game replayed six Normal Reimu A pilots through the complete authored input streams of Stage 1 and Stage 2. The twelve captures contain 39,736 original camera samples. Each sampled native current key, preceding key and input cursor matched its exact replay record. 283 archived color frames reproduce their policy images byte for byte.

| Role | Distinct pilots | Complete stages | Valid four-tick windows | Fixed candidate windows |
|---|---:|---:|---:|---:|
| train | 4 | 8 | 24,868 | 16,000 |
| held | 2 | 4 | 14,149 | 8,000 |

The player split preceded inspection of the held images. All twelve source archives contain the measured raw input, policy cameras, original player position, sparse full-color frames and original menu/endpoint photographs. Each archive includes the per-member digests. The release receipts and index bind the source replay and archive hashes. Reproduction requires a privately supplied matching retail installation.

| Player | Split | Stage 1 | Stage 2 |
|---|---|---|---|
| Green Spell | train | [2,731 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-01-s1-camera-input-xy.zip) | [3,903 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-01-s2-camera-input-xy.zip) |
| lipers | train | [2,687 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-02-s1-camera-input-xy.zip) | [3,626 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-02-s2-camera-input-xy.zip) |
| STL | train | [2,520 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-03-s1-camera-input-xy.zip) | [3,628 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-03-s2-camera-input-xy.zip) |
| シャオ | held | [2,524 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-04-s1-camera-input-xy.zip) | [3,775 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-04-s2-camera-input-xy.zip) |
| ZEX | held | [3,417 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-06-s1-camera-input-xy.zip) | [4,568 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-06-s2-camera-input-xy.zip) |
| SEO | train | [2,598 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-09-s1-camera-input-xy.zip) | [3,759 frames](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-100b-human-09-s2-camera-input-xy.zip) |

Training code can stream one capped stratum at a time with `scripts/th07_retail_bank.py`. Pass the SHA256 of the index, the directory with the signed files, and either the native capture root or a directory of released ZIPs. Its optimizer-facing entry point yields the four training authors. An explicit held entry point opens the two evaluation authors.

The independent clean-profile Practice menu shows only Stage 1 selectable for this shot and rank. The authored replay menu supplied the original Stage 2 starting snapshots. These measurements establish original playback and input alignment. Fresh ordinary-game policy runs require separate evidence for control and terminal progression.

Published archives total 463,684,998 bytes. Frozen acquisition protocol SHA256 `2477b7ce207c3a5bfc9541a973ae6a4c67762fae285c7252aa55d4f43ac34315`. Complete index SHA256 `3178f7e13d71e636d01a9a3352bbf50a04bbb39d92231826977a16cc3fd754c3`.
