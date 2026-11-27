# Second joint PPO generation on five original games

The learner restored both the actor and critic of the first five-title PPO. That policy collected 80 new full training credits, eight per title and difficulty. The new checkpoint completed four epochs on 38,623 playable decisions. The largest final-epoch joint action KL among the ten cells was 0.00665, on TH10 Hard. [Optimizer log and checkpoint hash](policy/report.json) · [Deployable actor](policy/policy.ts) · [Verified original training records](verified-training-by-rank.json)

The previous and new policies each reached Stage 2 in 0/80 fresh full credits and exhausted their spare lives. TH17 advanced farther with the new policy on seven of eight paired seeds on both difficulties. Mean gains in last Stage 1 tick were +1,136 on Hard (paired 95% interval +425 to +1,846) and +884 on Lunatic (+368 to +1,400). TH13 showed a Hard gain of +1,158 (+414 to +2,078), with a Lunatic change of −224 (−798 to +350). TH10 and TH12 showed no clear improvement on either difficulty, and TH11 settled scores were highly variable. These results span ten title and difficulty comparisons. [All stocks, casts and stage outcomes](evaluation/README.md) · [Two generations of paired deployment effects](../learning-curves/five-title-paired-generations.png) · [Hard recording](evaluation/first-hard-full-credits.mp4) · [Lunatic recording](evaluation/first-lunatic-full-credits.mp4)

| Original game | Hard decisions | Lunatic decisions |
|---|---:|---:|
| TH10 | 1,816 | 1,637 |
| TH11 | 1,601 | 1,691 |
| TH12 | 4,257 | 3,893 |
| TH13 | 6,178 | 5,905 |
| TH17 | 7,560 | 4,085 |

On TH12, 19 physical games produced 16 eligible trajectories. Three runs lacked enough evidence to distinguish consumption of a repeated last input before or after the fatal tick. Each replacement used its predefined backup seed on the same physical machine and lane. The held evaluation keeps every originally assigned seed. [Selection and exclusions](th12-train-source-selection.json) · [Native evidence for all 19 attempts](th12-train-native-records.release.json) · [Full input and camera archive](release-archive.json) · [Optimizer reward corrections](optimizer-reward-corrections.zip)

The main Hard and Lunatic training series now covers 320 distinct physical credits and 141,765 playable decisions. That count includes all five games and deduplicates previous experiments that shared the same behavior trajectories. [Counts by game, difficulty and study](../learning-curves/hard-lunatic-training-coverage.json)

The coordinator took 73 min 50 s from registration to checkpoint. Original collection and qualification took 70 min 22 s (95.3%), packing and reward preparation took 64 s, and learner verification, optimization and export took 2 min 23 s. Momentum supplied 32 eligible games, Endeavour 16, Gamer WSL 16, Steam Deck 8 and Entropy 8. Other experiments shared these machines, and the collection interval includes waiting for the TH12 evidence and its technical replacements. A faster learner alone could remove at most the 2 min 23 s learner interval from this particular run. [Coordinator timestamps and signed source receipts by host](coordinator-timing.json)

Across the 16 accepted TH12 games, median setup to first action was 22.87 s and median native gameplay was 53.51 s. The median compressed input transfer took 0.26 s to receive after its header, with 0.90 s from request to header and 1.23 s for local unpacking and verification. The current worker restarts Wine after a verified terminal screen. Reusing the game process is a higher-value engineering experiment here than replacing the transport. [Per-lane timings and byte counts](th12-qualified-lane-costs.json)

On a dedicated Entropy lane, a later engineering trial kept the original game process alive through the ranking screen, title menu and next stock game. On Hard this reduced setup to first action from 21.63 to 14.78 s, and on Lunatic from 22.61 to 15.85 s. All 1,710 native inputs in the two measured pairs matched. The mean setup saving was 6.81 s (30.8%) across two pairs on this host. [Native reset comparison](../th12-warm-reset-engineering/paired-reset-timing.json)

TH10 finished all 16 independent evaluation pairs. Mean last Stage 1 tick changed from 1,935 to 1,848 on Hard and from 1,507 to 1,476 on Lunatic. The paired 95% intervals were −568 to +485 ticks and −263 to +196 ticks. Both policies finished 0/8 at Stage 2 at each difficulty. [TH10 matched original trials and stocks](native10/ranked-native-comparison.json)
All 160 held credits are complete. Both generations reached Stage 2 in 0/80. TH13 Hard advanced by 1,158 native ticks on average (paired interval +414 to +2,078). TH17 Hard advanced by 1,136 ticks (+425 to +1,846) and Lunatic by 884 (+368 to +1,400). Each TH17 comparison had seven gains in eight pairs. All three estimates are discovery results. [Five-title paired evaluation](evaluation/README.md) · [All native progress intervals](../learning-curves/five-title-paired-generations.json) · [Independent TH17 confirmation protocol](../th17-five-generation2-replication/camera-protocol.json)
