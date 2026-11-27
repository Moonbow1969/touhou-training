# Does a small penalty for spending a Bomb improve ordinary play?

Both six-game PPO policies start from the same actor and critic. They see all 96,728 identical recorded training decisions, in the same order and with equal weight per title and difficulty. In one arm, the TH12 and TH17 reward cost for a confirmed same-life Bomb expenditure changes from −0.03 to zero. Death, stage progression and survival rewards retain their registered definitions. Other games retain their original rewards. An independent endpoint clock audit found two TH10 actions sampled after the final observed life loss in this shared input. Both models were trained on the historical input including those two rows.

| Original title | Hard cast events | Lunatic cast events |
|---|---:|---:|
| TH12 | 43 | 44 |
| TH17 | 86 | 73 |

The new actor completed 4 accepted epochs. SHA-256 is `2e66ea08d675155e99f4b0b820bbd0bedebc50ea0d977aed513c68f570c9ee54`. Held evaluation is registered on eight pairs per difficulty and original game. Primary outcome is Stage 2, with remaining lives, Bombs spent and last playable Bombs as secondary outcomes.

[All original cast events by game](cast-event-audit.json) · [Training log](policy/report.json) · [Policy weights](policy/policy.ts) · [Full-credit test registration](camera-protocol.json)
