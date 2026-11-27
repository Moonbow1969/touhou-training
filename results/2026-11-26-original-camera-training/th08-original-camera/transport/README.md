# Same-machine transport budget for TH08 original frames

The Windows learner and Ubuntu WSL2 camera process exchanged the same 24 held-out-of-training engineering frames from the first two full credits. The order of four treatment conditions was randomized within each of three repeats. Every measured transfer passed a byte-for-byte digest check. The table reports the median of three run summaries. The sender had a 0.8 CPU cgroup quota. These numbers measure the local Windows–WSL pipe and include the request, send and decode. Camera capture, inference, game hold, remote LAN time and disk writes are outside this interval.

| Payload | Codec | Mean bytes per observation | Median round-trip ms | 95th percentile ms |
|---|---|---:|---:|---:|
| whole rgb | raw | 921,600 | 83.83 | 93.64 |
| whole rgb | zstd | 384,010 | 6.23 | 16.30 |
| policy camera and lossless observers | raw | 145,544 | 1.33 | 147.26 |
| policy camera and lossless observers | zstd | 50,308 | 1.77 | 2.07 |

Compact transport contains the complete policy image, the lossless crop for resource and difficulty recognition, and the continue-text crop. It requires separate local video recording to retain a full-color movie. For the current four- and five-game rollout workers, model inference already runs next to each original game. The rollout files are transferred after the episode. [All repetitions, CPU encoding time and tails](report.json)
