# TH14 Normal full-credit policy development

## Randomized Bomb decisions

Four Reimu A story credits provided 31 verified 16-frame windows. A random assignment at the beginning of each enrolled window chose press or release with equal probability. The remaining three four-frame decisions released Bomb. Enrollment required positive native inventory. The frozen PPO-397 actor supplied movement, focus and shooting.

| Native player state | Release | Press |
| --- | --- | --- |
| Ordinary state 1 | 0 life losses in 5 windows | 0 life losses in 8 windows, 7 Bombs consumed |
| Hit state 4 | 8 life losses in 8 windows | 0 life losses in 10 windows, 10 Bombs consumed |

The table and a separate linear two-action value model were fitted using native player state, previous Bomb key and inventory. This observation contract adds information to the original PPO actor. The table requires support from both randomized actions and uses the anchor policy for unsupported entries. The linear model predicts continuously from its fitted coefficients. Stock zero and previous-key-held contexts had no randomized support. [Interventions](bomb-randomized-collection.json) · [Fitted table](bomb-effect-table.json) · [Linear fit](bomb-value-fit.json)

| Policy sample seed | PPO-397 Game Over | PPO-397 + fitted table Game Over | Actual Bombs consumed, control / fitted |
| --- | ---: | ---: | ---: |
| 1701 | 6317 | 10973 | 1 / 9 |
| 1702 | 5467 | 10114 | 0 / 9 |

All four full credits ended in Stage 1. The fitted head passed the boss spell with native ID 4 in both trials. Stage 2 arrivals were 0/2 in each arm. Fixed policy sampling seeds do not control native game randomness. [Native paired evaluation](bomb-table-eval.json)

## Where the original late-stage curriculum failed

The fixed learned crossover was scheduled to hand control to an online PPO learner at frame 7600. It died at 7482, 7406, 5270 and 6170 in the four attempted prefixes. No learner transitions or updates occurred. A subsequent run selected frame 5000. [Rejected prefixes](rejected-7600-prefix.json) · [Reference action trace](reference-steps.csv)

## Testing the movement anchor

The same fitted linear Bomb value function was applied on every stage and every decision. Its base action came from PPO-397 or independently trained PPO-211. All four complete credits ended in Stage 1. The Game Over frames on policy seeds 1901 and 1902 were 10,168 and 10,069 for PPO-397, and 8,284 and 8,268 for PPO-211. Both movement actors exhausted all three starting lives on both seeds. These trials were scored at ordinary stock speed. A failed menu-image verification between credits required reacquisition of a verified new Stage 1 credit before the remaining trials. [Native results](bomb-value-story-transfer.json)

The retained native observation/action archives contain full boss HP and vulnerability. On 1,750 right-wall observations of the first PPO-397 credit, inward movement had mean probability 0.0152 under PPO-397, 0.3743 under PPO-211 and 0.0909 under their 50:50 parameter interpolation. These are per-observation probabilities. No native clearance was measured for the interpolated actor. [Wall probe](native-wall-probes.json)

The frozen human-data TH14 training split had 85,416 rows and 591 Bomb-positive four-tick chunks. All 85,416 training rows excluded the native hit state due to the movement-controllability filter. Future captures can retain aligned hit and recovery windows separately. A raw cadence-rejected archive was used only to verify extraction. [Input alignment audit](human-hit-window-audit.json)
