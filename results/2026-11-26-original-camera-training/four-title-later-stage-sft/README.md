# Four-title human imitation with later story stages

A matched 3,000-update comparison retained the original 47 synchronized stages and added 21 later stages from five of the same players. Available four-tick decision windows rose from 148,415 to 231,511. Both actors sampled 192,000 training windows. The TH10, TH12 and TH13 increment supplied 159 observed Bomb-key rising edges. The original six-replay acquisition roster deliberately favored runs with Bomb use. TH17 training data stayed fixed. The registered increment includes TH13 Hard and has no new TH13 Lunatic stages.

| Game | Development pilots/stages | Correct directions, previous → expanded | Correct changes | Correct movement starts |
|---|---:|---:|---:|---:|
| TH10 | 2/3 | 4,105 → 4,283 / 6,600 | 357 → 377 / 1,543 | 48 → 44 / 470 |
| TH12 | 2/3 | 5,158 → 5,259 / 9,806 | 699 → 778 / 3,275 | 94 → 76 / 969 |
| TH13 | 2/4 | 7,864 → 8,031 / 11,835 | 749 → 689 / 2,863 | 66 → 63 / 839 |

All three games improved on overall direction accuracy in the reused development set. Correct movement starts fell in all three, and correct TH13 changes fell from 749 to 689. These measurements are from human trajectory observations.

[Frozen source roster, hashes and authors](../human-train-six-complete-credits/later-train-first-21-frozen.json) · [Weights](policy/policy.ts) · [Training log](policy/loss.jsonl) · [Per-replay results](repeated-dev/held.json) · [Original TH10 and TH11 SFT-to-PPO registration](../later-human-online/camera-protocol.json)
