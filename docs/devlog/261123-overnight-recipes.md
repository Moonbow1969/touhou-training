# 261123 Overnight training recipes

## Movement intent

Separated movement intent from conditional direction while keeping the native action space. The deterministic decoder compares the total probability of the eight moving directions with the probability of idle, then selects a direction when movement wins. Training uses a binary intent loss and a conditional direction loss on four-tick soft labels. Existing checkpoints retain their decoder.

| Data and endpoint | Flat reference | Factorized candidate |
| --- | ---: | ---: |
| 126,781 paired train decisions, held-out direction changes | 17.17% | 14.65% |
| Training on TH13/14 only, unseen TH17 direction changes | 31.65% | 35.21% |
| Training on TH13/14 only, unseen TH17 overall movement accuracy | 57.90% | 48.77% |
| TH17 native first miss, Normal Reimu Wolf Stage 1, one trial each | 2,450 frames | 514 frames |

The factorized native trial began at frame 6 and made 11 moving decisions before losing a life. This candidate stays experimental. Increasing movement or matching more expert turns can still send the policy into danger on states reached by its own actions.

## World forecast objective

Trained for 20 epochs on the same 370,330 state–action transitions used by the previous world model, with an extra loss over the cells around the future replay player. The final evaluation uses a fixed near-player region and the same test trajectories.

| Forecast horizon | Reference near-player AP | Local-loss candidate AP |
| --- | ---: | ---: |
| 4 native frames | 0.959 | 0.952 |
| 12 native frames | 0.851 | 0.844 |
| 24 native frames | 0.646 | 0.654 |

The four-frame reference remains the short-horizon model. The evaluator can also score its predictions in shadow mode on on-policy native actions with exact four-frame transitions. Shadow metrics use observed future states and do not drive player inputs.

## Boss aiming

Trained on distinct native episodes and selected checkpoints on a Stage 4 Practice run. Reserved Stage 6 Practice for comparison. Adding Stage 5 examples alone changed held-out recovery of teacher corrections from 91.35% to 91.07%. Training with the same clearance mask used at inference raised that figure to 92.12% on 1,041 corrective examples. The metric measures imitation of the geometric teacher.

## Native PPO and replay capture

TH17 replay playback continued in the background while TH14 accepted foreground controller actions. Started a six-stage TH17 Lunatic Youmu Otter training replay (`th17_ud039f`). Its contributor remains in the training split and the existing holdouts stay fixed. A queued job freezes all verified stage pairs and compares adding the new data with title-balanced training.

A third title, TH13, returned a single-instance error during native launch. TH14 lost focus in this transition. The interrupted run retained its 5,888-step PPO checkpoint, then resumed toward the same 8,192-step budget. The original console, startup retry, and continuation have separate run records. The controller now explicitly leaves the game alive if its debugger exits, and detects native time skips during foreground recovery.

In a separate audit of 2,725 captured TH14 practice states, the imitation dodge head had mean action entropy 0.041 nats. Dividing the initialized actor logits by eight left its deterministic directions unchanged and raised mean entropy to 0.402 nats. On 1,561 mixed-clearance states, the resulting distribution assigned 5.09% mean probability to choices with negative analytic clearance. A native trial is needed for the exploration and performance effects.

The capture completed all six stages. It added 20,779 verified decisions, reaching 147,560 training decisions across 38 stages. Per-stage moving alignment ranged from 99.65% to 100%. The original validation and test shards remain fixed. Both expanded SFT candidates use 20 epochs and seed 61 with the same architecture and stage context as the 126,781-decision reference.

## Exploration across safe alternatives

Compared hard geometric imitation with a 35% target mixture over geometrically near-equivalent nonnegative-clearance alternatives. Both used the same 9,000 replay-derived training decisions, 29 original on-policy corrections, and held-out validation rows. Correction agreement rose from 72.34% to 76.60% on 47 validation corrections.

On a separate audit of 2,725 captured practice states, mean entropy rose from 0.066 to 1.318 nats compared with the matched hard-target head. Mean probability on negative predicted clearance fell from 0.166% to 0.103% among 1,561 mixed-clearance states. These margins are geometric estimates. This candidate superseded the queued uniform temperature intervention before that native trial started. The native experiment uses the same geometry, PPO update budget, and comparative controls as the prior recipe.

## Replicating the data result

Ran the same learner for 20 epochs at seeds 61, 97, and 137. The 51,421-row held-out set comprises TH17 Normal expert observations from players excluded by the split. These are replay metrics for a title present in training.

| SFT training set and sampling | Mean direction accuracy | Mean direction-change accuracy | Direction changes by seed |
| --- | ---: | ---: | --- |
| Original 126,781 decisions | 58.44% | 15.58% | 17.17%, 14.70%, 14.88% |
| Expanded 147,560 decisions | 63.35% | 18.71% | 24.29%, 16.28%, 15.56% |
| Expanded with title balance 0.5 | 63.90% | 27.52% | 27.51%, 32.26%, 22.79% |

The expanded and balanced recipe improved direction-change accuracy at each of the three seeds. Its native policy contribution still needs testing.

## Four-tick geometry regularization

Measured constant-direction kinematics directly from paired training replays. TH17 Marisa Wolf moved at about 5.0 units per native frame unfocused, compared with 4.5 for the other observed profiles. Each tested profile measured about 2.0 while focused.

Added weight 0.3 on predicted collision probability when the replay action was geometrically safe and at least one alternative action was unsafe. In the seed-61 ablation, both checkpoints selected 16 predicted collisions among the same 3,437 eligible test decisions. Direction-change accuracy moved from 27.51% to 27.32%. Keep the regularized checkpoint experimental.
