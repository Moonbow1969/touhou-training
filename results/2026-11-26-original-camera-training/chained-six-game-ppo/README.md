# Six-title full-action PPO from the five-title human policy

The learner restored the same trained actor and critic as the five-title comparator. It consumed all 80 selected games from that comparator and 16 fresh TH08 games, with eight Hard and eight Lunatic TH08 starts. The shared six-title checkpoint learned from 47,608 playable decisions across twelve game/difficulty combinations.

Both recipes attempted 259 optimizer steps per epoch. The six-title fit accepted 4 epochs. Adding TH08 changes the fraction of the batch allocated to each previously trained work. The separate 64-game TH08 evaluation uses prespecified controller seeds and fixed policy weights.

[Source allocation](camera-protocol.json) · [Exact selected seeds](fit-protocol.json) · [Full optimizer log](policy/report.json) · [Deployable policy](policy/policy.ts) · [Existing five-title input archives](../chained-five-game-ppo/README.md)
