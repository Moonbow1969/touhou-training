# One camera policy trained on five original Touhou games

The same PPO checkpoint trained on Hard and Lunatic in TH10, TH11, TH12, TH13 and TH17. A frozen human-replay SFT actor collected the 48 original training credits. The learner replayed each observation and checked the behavior log probabilities, restored the SFT actor, initialized the value head to zero and completed four epochs. Each game and difficulty cell received equal weight. The policy controls nine-way movement, focus, shooting and casting from game images. [Checkpoint and optimizer log](policy/report.json) · [TorchScript actor](policy/policy.ts) · [Registered design](camera-protocol.json)

| Game | Hard credits | Hard decisions | Lunatic credits | Lunatic decisions |
| --- | ---: | ---: | ---: | ---: |
| TH10 | 4 | 907 | 4 | 684 |
| TH11 | 8 | 2,083 | 8 | 1,682 |
| TH12 | 4 | 1,220 | 4 | 1,090 |
| TH13 | 4 | 3,618 | 4 | 2,722 |
| TH17 | 4 | 2,280 | 4 | 1,937 |
| Total | 24 | 10,108 | 24 | 8,115 |

The joint fit reuses all 32 physical training credits from the four-game experiment and all 16 from the TH11 specialist. The union contains 18,223 distinct policy decisions. The original archives contain the trajectories. Each row in the joint manifest points back to its source and checksum. [Joint manifest](training/credits.json) · [Four-game raw release](../four-game-merged-online/release-archive.json) · [TH11 raw release](../th11-original-ppo/train-release.json)

The registered original-game evaluation has eight paired controller seeds per game and difficulty, 80 pairs in total. For TH10, TH11, TH13 and TH17 the frozen SFT control comes from the same previously registered build, seed and host; it ran before this candidate. Both TH12 arms use 16 new controller seeds with the repaired clock. The controller seeds govern policy sampling. The original game RNG is uncontrolled. TH11 runs the publisher's 0.02a three-stage trial and spends Power to cast. [TH11 stage-register qualification](../th11-original-ppo/stage2-qualification.json)
