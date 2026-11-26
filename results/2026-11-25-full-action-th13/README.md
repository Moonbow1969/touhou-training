# TH13 original full-credit benchmark

TH13 ran on Endeavour with six isolated Wine/Xvfb gameplay lanes. Each original Trial 0.01a Normal Reimu credit started with one active life, two spares and two Bombs. The shared movement, focus, shooting and Bomb heads controlled every decision after the opening. TH13's manual Trance key is outside the shared action vocabulary. Controller seeds matched across arms. The original game's RNG varied between processes.

## Frozen checkpoint comparison

Each row comprises all 12 held seeds 731001–731012. Every credit ended on the original ranking screen with zero active lives and zero Continues. Since nobody entered Stage 2, the entry inventory is undefined.

| Checkpoint | Stage 2 arrivals | Median last observed Stage 1 tick | Mean Bombs at final loss | Effective Bombs per credit | Losses with Bomb stock |
| --- | ---: | ---: | ---: | ---: | ---: |
| Human SFT | 0/12 | 5,172.5 | 1.25 | 2.42 | 28/36 |
| PPO1 | 0/12 | 5,379 | 1.42 | 2.58 | 26/36 |
| PPO2 | 0/12 | 6,734.5 | 1.08 | 3.17 | 22/36 |

The per-arm Wilson 95% interval for 0/12 is 0 to 24.25%. Relative to SFT, PPO2 reached a later native ranking tick on 9/12 matched controller seeds. Its mean paired difference was +1,244 ticks with a 95% paired bootstrap interval from +319 to +2,185. The two-sided exact sign test gave p=0.146. The native clock measures Stage 1 progress independently of the wall-time rate of the six software-rendered games. Of 52,344 combined requested decisions, 52,333 matched the original immediate input word. All nine movement classes occurred under each checkpoint. [Paired results](primary-paired-native-progress.json) · [Seed-level results](full-credits.tsv) · [Figure](original-held-full-credits.png)

The uninterrupted 4:18 original recording starts with all three lives and ends on the ranking screen at 366,130 points with one unused Bomb. The delivered stream contains 3,096 frames at 12 fps. [PPO2 seed 731012](ppo2-731012-full-credit-12fps.mp4) · [Decoding and endpoint receipt](ppo2-731012-video.json)

## Training-only diagnostics

Eight additional PPO2 credits contributed 12,368 one-to-one pre-action images and 11,405 stable original player positions. Recomputing all 84 visual features from the saved images matched the deployed actor to within 3×10⁻⁸. In the conservative interior, 98.2% of sampled up commands and 98.2% of sampled down commands moved Reimu over one native pixel in the requested direction. Across all qualified decisions, 1,556 of 3,226 down requests occurred while she was already at the lower boundary. [Original movement measurements](ppo2-training-physical-motion.json) · [Complete-data hashes](complete-ppo2-training-pixels-manifest.json) · [64-frame coordinate fixture](original-player-coordinates-fixture.npz)

Across the 20 training credits, all 60 initial lives were lost, no extra lives were earned and nobody changed stage. The realized undiscounted stock component was therefore −3 in every training episode. The time and Bomb terms supplied all differences in total episodic return. [Training reward components](training-reward-decomposition.json)

## Reproduction

The per-arm trace ZIPs store one folder per seed containing the scoring receipt, original event journal, decision arrays, behavior log-probabilities and a separate final native stock snapshot. These can reproduce the terminal outcomes and train-only PPO batches. The original game is installed separately. [Worker build and capture](../../tools/rollout-worker-probe/th13-full/README.md) · [Input calibration](input-calibration.json) · [Six-lane throughput](rollout-performance.json)
