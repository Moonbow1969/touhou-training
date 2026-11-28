# TH08 Normal Ghost direction and duration

Claude Opus 5.5 recommended an inexpensive comparison of direction every decision against direction with a committed duration of one to four decisions. Both arms receive four causal camera states and their preceding direction, freeze the same encoder, and train the same-sized readout using 58,653 Normal windows, identical batches, and 1,200 updates per arm. During training the preceding direction is replaced with a random direction 20% of the time.

| Human label on held replay frames | Predict every decision | Commit direction for 1–4 decisions |
|---|---:|---:|
| Human changes direction (3,299 windows) | 26.83% | 26.49% |
| Human keeps direction (12,248 windows) | 53.90% | 52.42% |

Each policy uses its own previous prediction as it moves through the same fixed human camera stream. The treatment improves change-direction accuracy in two of six stages, ties in one, and declines in three. It falls below both deployment thresholds, a five-percentage-point gain on changes and 80% accuracy on held directions. These are offline results from two previously used held pilots. The direction-and-duration readout remains an offline checkpoint and has no original-game results.

Training used 153,600 presentations per arm. The decoder weight SHA-256 hashes are `2ba88f4d6a7ede058876112b3f74b6290afbc438620f779a0f19f867a2004a85` (every decision) and `89ad83e7e695615fb15c942f7545b9a897327bd08237057c0c315a26ee86d6a9` (duration).

[Runner](../../../scripts/train_th08_normal_action_chunks.py) · [Protocol](protocol.json) · [Per-stage scores](report.json) · [Learner log](loss.jsonl) · [Every-decision readout](one-step.pt) · [Duration readout](chunk.pt)
