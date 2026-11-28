# TH07 two-phase PPO with measured native durations

The preceding strict-window PPO finished its twenty-pair comparison with 13/20 original practice clears versus 11/20 for the frozen parent. The paired gain was 10 percentage points with a 95% bootstrap interval of −20 to +40. The 20-point gate triggered this separate fit from the same original training allocation. [Previous independent comparison](../../th07-letty-nonspell2-held) · [Release decision](release-decision.json)

| Optimizer input | Strict-window fit | Native-duration fit |
|---|---:|---:|
| Admitted original training trajectories | 18 | 29 |
| Verified goal completions in admitted sources | 11 | 18 |
| Value transitions | 10,783 | 19,221 |
| Actor decisions | 10,783 | 19,218 |

The native input archive established all 76,892 original key slots in the admitted traces. The three transitions with a mismatching prior key contributed value targets. All other decisions had the sampled mask in every original tick. The new fit used 14,585 transitions beginning at Nonspell 2 and 4,636 beginning at the later final spell. One physical original lacked complete source issuance after a postprocessing failure and stayed excluded. [Timing and source proof](../../th07-retail-curriculum-ppo/runtime) · [Manifest](manifest.json)

The RTX 4070 completed 402 accepted Adam steps in two epochs. Joint action KL was 0.00147 and 0.00167. The selected camera-plus-native value baseline had out-of-fold RMSE 0.640 across the new training credits. The four learned actor heads still receive the camera alone. The frozen checkpoint hash is `b24720608a5dc88a6aa6b56fbbae7080388e1784b7fe9e474d8f90eeb8920ce0`. Its twenty paired held seeds are disjoint from the previous evaluation and all training originals. [Checkpoint](policy.ts) · [Optimizer output](learner.log) · [Fit report](report.json) · [Source and value metrics](protocol.json)
