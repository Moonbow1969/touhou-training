# Second joint PPO generation on five original games

The learner restored both the actor and critic of the first five-title PPO. That policy collected 80 new full training credits, eight per title and difficulty. The new checkpoint completed four epochs on 38,623 playable decisions. The largest final-epoch joint action KL among the ten cells was 0.00665, on TH10 Hard. Real workers are running the registered 80 fresh paired starts against the previous PPO. [Optimizer log and checkpoint hash](policy/report.json) · [Deployable actor](policy/policy.ts) · [Verified original training records](verified-training-by-rank.json)

| Original game | Hard decisions | Lunatic decisions |
|---|---:|---:|
| TH10 | 1,816 | 1,637 |
| TH11 | 1,601 | 1,691 |
| TH12 | 4,257 | 3,893 |
| TH13 | 6,178 | 5,905 |
| TH17 | 7,560 | 4,085 |

On TH12, 19 physical games produced 16 eligible trajectories. Three runs lacked enough evidence to distinguish consumption of a repeated last input before or after the fatal tick. Each replacement used its predefined backup seed on the same physical machine and lane. The held evaluation keeps every originally assigned seed. [Selection and exclusions](th12-train-source-selection.json) · [Native evidence for all 19 attempts](th12-train-native-records.release.json) · [Full input and camera archive](release-archive.json) · [Optimizer reward corrections](optimizer-reward-corrections.zip)

The main Hard and Lunatic training series now covers 320 distinct physical credits and 141,765 playable decisions. That count includes all five games and deduplicates previous experiments that shared the same behavior trajectories. [Counts by game, difficulty and study](../learning-curves/hard-lunatic-training-coverage.json)
