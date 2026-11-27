# TH08 zero-shot original-game evaluation

Both policies trained on TH10, TH11, TH12, TH13 and TH17. The supervised policy and its full-action PPO child have a verified training history excluding TH08. Sixteen new controller seeds on the original TH08 trial each contributed one ordinary full-stock game per actor with no Continue. Raw game randomness was uncontrolled.

| Difficulty | Stage 2, SFT → PPO | Mean time to Game Over, seconds | Mean time to first lost life, seconds | Mean last playable Bombs | Mean terminal spare lives |
|---|---:|---:|---:|---:|---:|
| Hard | 0/8 → 0/8 | 69.26 → 73.28 | 25.09 → 30.13 | 2.125 → 1.500 | 0 → 0 |
| Lunatic | 0/8 → 0/8 | 51.05 → 57.93 | 17.18 → 18.87 | 2.125 → 1.875 | 0 → 0 |

All 32 endpoints were identified. The 95% paired stratified bootstrap intervals for the mean time difference were −8.57 to +18.35 seconds on Hard and −0.49 to +17.17 on Lunatic. Both arms used nine directions and the three learned buttons. The physical logs cover 19,634 decisions.

[Per-seed outcomes and interval estimates](original-paired-results.json) · [Checkpoint lineage](verified-checkpoint-lineage.json) · [Trial registration](protocol.json)

[All 16 Hard recordings and input traces](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/reviewed-five-unseen-th08-th08-original-hard-held.zip) · [All 16 Lunatic recordings and input traces](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/reviewed-five-unseen-th08-th08-original-lunatic-held.zip)
