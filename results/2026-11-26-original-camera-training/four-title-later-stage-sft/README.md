# Four-title human imitation with later story stages

A matched 3,000-update comparison retained the original 47 synchronized stages and added 21 later stages from five of the same players. Available four-tick decision windows rose from 148,415 to 231,511. Both actors sampled 192,000 training windows. The TH10, TH12 and TH13 increment supplied 159 observed Bomb-key rising edges. The original six-replay acquisition roster deliberately favored runs with Bomb use. TH17 training data stayed fixed. The registered increment includes TH13 Hard and has no new TH13 Lunatic stages.

| Game | Development pilots/stages | Correct directions, previous → expanded | Correct changes | Correct movement starts |
|---|---:|---:|---:|---:|
| TH10 | 2/3 | 4,105 → 4,283 / 6,600 | 357 → 377 / 1,543 | 48 → 44 / 470 |
| TH12 | 2/3 | 5,158 → 5,259 / 9,806 | 699 → 778 / 3,275 | 94 → 76 / 969 |
| TH13 | 2/4 | 7,864 → 8,031 / 11,835 | 749 → 689 / 2,863 | 66 → 63 / 839 |

All three games improved on overall direction accuracy in the reused development set. Correct movement starts fell in all three, and correct TH13 changes fell from 749 to 689. These measurements are from human trajectory observations.

[Frozen source roster, hashes and authors](../human-train-six-complete-credits/later-train-first-21-frozen.json) · [Weights](policy/policy.ts) · [Training log](policy/loss.jsonl) · [Per-replay results](repeated-dev/held.json) · [Original TH10 and TH11 SFT-to-PPO registration](../later-human-online/camera-protocol.json)

## Previously unseen TH17 pilots

Four pilots supplied eight independently synchronized original stages and 24,643 decisions. The 47-stage and 68-stage policies scored 11,658 and 12,041 correct directions (47.31% and 48.86%). The expanded training set improved direction changes from 1,956 to 1,983 of 8,118, with starts declining from 131 to 108 of 2,476 and turns declining from 391 to 372 of 3,171. Correct stops rose from 1,434 to 1,503 of 2,471. Neither policy had a Bomb probability of at least 0.5 on any annotated onset. In the top 5% of frames per stage, the earlier model flagged 0/19 onsets and the later model 1/19, with 1,236 frames flagged by each. All 19 human onsets came from the two Lunatic players. These results informed the subsequent movement and Bomb-objective experiments.

[Per-stage movement metrics](fresh-four/held.json) · [All Bomb predictions and false alerts](fresh-four/bomb-onsets.json) · [Original held data and synchronization](../human-retail17-wave2-held/held-github-release.json)
