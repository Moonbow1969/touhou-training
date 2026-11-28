# TH07 0.11 Normal four-head SFT with an auxiliary player locator

The auxiliary loss raised held movement accuracy from 60.858% to 62.489% across three paired seeds on zylo's Stage 1. All three paired differences were positive. These are offline results for one previously scored player. Native full-credit evaluation is a separate experiment.

| Held metric | Four-head action loss | Action plus native XY loss |
|---|---:|---:|
| Movement accuracy | 60.858% | 62.489% |
| Movement accuracy on 511 expert direction changes | 18.917% | 19.439% |
| Stable movement accuracy on 2,106 decisions | 71.035% | 72.934% |
| Movement NLL | 1.13209 | 1.12980 |
| Joint NLL over four action heads | 1.37364 | 1.38159 |
| Focus accuracy | 92.625% | 91.950% |
| Shoot accuracy | 98.714% | 98.739% |
| Mean Bomb probability | 0.572% | 0.579% |
| Mean per-seed median native player error | 9.72 px | 5.74 px |
| Mean per-seed 90th-percentile native player error | 23.87 px | 17.43 px |

All 2,617 held decisions had a negative Bomb label. The sample cannot measure Bomb recall. Movement gains were +1.414, +1.834 and +1.643 percentage points for training seeds 71281, 71282 and 71283. The same-author contiguous end block scored 76.610% versus 76.360% movement. The three seeds share one training replay and one held replay.

The frozen parent comes from `chained-six-game-ppo/policy/policy.ts`, SHA-256 `a6e1dec1625af50e214f141a4e405795fbdae905d78c4b887e423de0bca6a77e`. Boredperson contributed 2,129 optimizer rows and 533 chronological sanity rows. The last training camera tick was 8836 and the first sanity tick was 8848. The four-tick camera histories and future-label windows are disjoint across this boundary. The spatial loss uses the same-frame native player XY collected before the policy action. All deployed controls consume only the two-channel camera tensor. Both arms reused the inherited spatial and action parameters. Paired seeds shared byte-identical epoch permutations.

Every run used eight complete passes, 536 Adam updates and a constant learning rate of 0.00003. Training used an equivalent nonoverlapping 4×4 pooling expression for deterministic CUDA backward. The greatest measured logit difference against the deployed TorchScript across the six final checkpoints' common training probe was 0.00000120. Held evaluation used the deployed TorchScript logits directly. The first CUDA compatibility probe exited before its first successful update. Its log is retained.

`protocol.json` records the design. `fit/result.json` contains each seed, both arms, complete held and sanity metrics, gates and observed GPU memory. `fit/<seed>-<arm>/fit.json` binds the deployed model to its minibatch order, epoch losses and SHA-256. `train.log` captures all 48 epochs. The signed camera dataset resides separately under `th07-normal-native-xy`. It contains the source-side audits, selected native rows, full camera streams and the registered Boredperson/zylo split. Run `scripts/fit_th07_locator_ablation.py --protocol protocol.json --data /path/to/th07-normal-native-xy --parent /path/to/parent/policy.ts --out /new/empty/run`.
