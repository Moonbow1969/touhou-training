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

## Native baseline with matched geometry

Completed 8,192 native TH14 Stage 1 PPO steps and 128 optimizer epochs across the interrupted job and its checkpoint continuation. All three controllers reached the same 9,618-frame cutoff without a miss in all three trials. Median displayed score gains were 228,210 for the frozen imitation head, 198,110 for PPO, and 234,100 for rule-based selection. The test stopped before a clear in all nine episodes, so the survival endpoint could not distinguish the arms.

The next native comparison trains the soft-clearance initialization for the same 8,192 steps. Its evaluation applies a common aiming policy to every arm, includes the first PPO checkpoint as a frozen reference, and allows two misses and 3,300 decisions so it can measure stage-clear outcomes.

## Additional on-policy checks

Evaluated the frozen SFT checkpoints on 2,725 states recorded during native TH14 Stage 4 controller runs. The expanded balanced checkpoint chose actions with negative geometric margin in 72 states, compared with 77 for the earlier stage-clock checkpoint. Applying the same rule left nine negative margins for each candidate and for the older deployed SFT. The balanced candidate fired in 91.73% of states with a positive boss HP observation, compared with 98.27% for the deployed base. Keep the native base fixed for the PPO comparison and the story pilot.

Adding 641 captured controller states to the soft-clearance imitation recipe preserved nearly identical entropy and negative-margin probability on a separate 2,084-state practice recording. Corrective action agreement changed from 111/128 to 110/128. The additional capture stays experimental.

## Progression benchmark and early stopping

The soft-clearance PPO recipe completed the same 8,192 native steps as the reference. With common geometric boss aim, the soft imitation head, soft PPO head, geometric selector, and earlier PPO each cleared Stage 1 in both native trials, with no lives lost. This benchmark shows the common controller can combine survival and boss damage. It did not demonstrate a progression advantage for PPO at this stage.

At seed 61, the balanced SFT run trained for 60 epochs selected the same epoch-2 checkpoint as the 20-epoch run, with identical SHA-256. Validation loss moved from 1.279 at selection to 2.026 at epoch 60. A new run with patience six stopped at epoch 8 and produced the same selected checkpoint. Replaying the stopping rule on the three balanced histories would have stopped at epochs 8, 10, and 9 while preserving their selected checkpoints. The latter two stop times are retrospective.

## Additional data for the world model

Adding the new replay raised the world-model training set from 370,330 to 432,456 exact transitions across the three forecast horizons. With the original architecture, objective and 20 epochs, held-out near-player occupancy AP moved from 0.959, 0.851, and 0.646 to 0.949, 0.835, and 0.641 at 4, 12, and 24 frames. The story pilot retains the earlier checkpoint for short-horizon shadow evaluation.

## Horizontal data augmentation

Mirrored half of the training examples, including projectile positions and horizontal velocities, player and boss coordinates, motion history, and the directional labels. The three seeds produced 28.25%, 23.01%, and 22.44% direction-change accuracy on the same held-out replays, a 24.57% mean compared with 27.52% without mirroring. The flag remains off by default.

## First native one-credit clear

Run `20261123T113237Z-overnight-th14-analytic-story-e2c63234` cleared all six stages of TH14 1.00b Normal with Reimu A. The original executable entered Ending/Credits with zero continues. It recorded a displayed score of **296,778,700**, six misses and 23 bombs over 25,702 decisions. The previous full-story benchmark reached Stage 4. This is one completed story trial with the frozen motion-history SFT base, geometric dodge, masked learned boss aim and emergency bombing.

The original four-frame world checkpoint scored 0.975 near-player occupancy AP against 0.936 for ballistic extrapolation on 2,960 exact on-policy transitions. The checkpoint observed the run in shadow mode throughout. Of 29 logged entries into the native hit state, the action geometry assigned a margin above four units to 25 and a negative margin to one. These entries include deathbomb opportunities, so they differ from the six lives lost. The samples show a clearance calibration gap without identifying the cause.

Separate passive recordings preserved 2,214 Stage 5 and 3,519 Stage 6 state and input windows for diagnosis. Among adjacent moving transitions included in the alignment audit, 89.02% and 74.44% respectively reproduced the observed displacement within half a unit. These captures are controller trajectories. The laser inventory saw no active laser entries across 1,321 sampled Stage 5 frames and 260 sampled Stage 6 frames early in the inventory window.

## Fully balancing title weights

With the expanded training corpus fixed, increased `--title-balance` from 0.5 to 1.0. Direction-change accuracy improved at each seed, from 27.51%, 32.26%, and 22.79% to 31.05%, 32.55%, and 26.85%. Mean direction-change accuracy moved from 27.52% to 30.15%. Mean overall direction accuracy moved from 63.90% to 64.18%. These measurements use the same reserved TH17 Normal experts on 51,421 paired decisions. The seed-61 checkpoint is the candidate for a matched native story trial.

## Selecting the world checkpoint by forecast quality

Kept the expanded 432,456-transition dataset, optimizer and loss fixed. Used the mean of the three validation near-player occupancy AP values to select weights. With a 32-epoch budget and patience five, the best checkpoint was epoch 5 and training stopped at epoch 10. The held-out replays used the same neighborhood and aggregation as the earlier 20-epoch comparison.

| Horizon | Held-out pairs | Expanded epoch 20 AP | Expanded selected AP |
| --- | ---: | ---: | ---: |
| 4 frames | 51,078 | 0.949 | 0.960 |
| 12 frames | 50,884 | 0.835 | 0.854 |
| 24 frames | 50,657 | 0.641 | 0.646 |

The selected checkpoint recovers the short-horizon accuracy of the original 370,330-transition model. Its held-out player displacement error changed from 1.716 to 1.671 units at four frames, from 2.734 to 2.795 at 12 frames, and from 4.098 to 4.487 at 24 frames. The selection criterion is occupancy AP. The native one-credit comparisons keep the earlier world checkpoint in shadow mode.

In a separate endpoint ranking audit, forecasted the bullet field under the recorded inputs and held those future hazards fixed while scoring 18 possible move-and-focus endpoints. Among 2,796 twelve-frame windows with both occupied and free candidate neighborhoods, the occupied-neighborhood rate of minimum-risk picks was 0.989% for ballistic extrapolation, 0.118% for the original world checkpoint and 0.119% for the selected checkpoint. The outcome is coarse 3×3 cell occupancy under the recorded actions. Both native story runs retained shadow-only world forecasts.

## Checking native motion against the saved replay

Fit the timestamp offset on the first half of the captured moving windows. The archived replay matched every sampled input in the held-out 592 Stage 5 windows and 894 Stage 6 windows at that offset. Reconstructing player motion from the exported replay inputs gave the same displacement errors as using the live input words. Shifting the input windows by one or two frames reduced alignment in both stages.

On the full native examples, the fixed-kinematics predictor averaged 0.587 and 1.731 game units of error on moving Stage 5 and Stage 6 windows. The selected world model's displacement head averaged 1.698 and 2.766 units. A 20-epoch residual predictor trained on 20,075 TH14 expert windows selected its initial zero correction on the separate 2,258-window validation split. Keep analytic player kinematics for the controller. The late-stage motion errors persisted when reconstructing the captured trajectories from the archived replay inputs.

## Causal speed calibration

Some moving windows fit one-third of the usual speed. Tested a two-speed estimator that uses only earlier observed displacements to choose the next speed and expires evidence after 40 frames. Stage 6 was exploratory. Stages 4 and 5 came from the subsequent story trajectory.

| Native stage | Moving windows | Fixed mean error | Adaptive mean error | Within 0.5 units, fixed → adaptive |
| --- | ---: | ---: | ---: | ---: |
| 4 | 438 | 4.482 | 2.156 | 48.17% → 58.90% |
| 5 | 705 | 0.811 | 0.320 | 92.34% → 94.33% |
| 6 | 1,788 | 1.731 | 1.032 | 74.44% → 80.98% |

A separate passive probe found the player position stable across repeated reads at the same native stage timer in all 11,552 sampled frames. Input words changed in 13 of the 4,251 sampled Stage 6 frames. These observations cover sampled times in the subsequent story run. The adaptive speed flag is experimental. The recorded comparisons use the earlier controls.

## Repeating title balance with TH17 excluded

Repeated the balance comparison across three matched seeds with TH17 excluded from both training and model selection. Both arms used the same 47,117 TH13/14 training decisions and 2,288 TH14 validation decisions. The test reused the 51,421 TH17 Normal decisions. Under full title balance, mean direction-change accuracy moved from 32.99% to 33.08%. Overall direction accuracy moved from 57.38% to 57.15%. The small average change varied in sign across seeds. There is no consistent improvement on the unseen title in this comparison.

## Native story with the expanded SFT policy

Run `20261123T120553Z-overnight-th14-title-equal-story-d39cd57a` replaced the base checkpoint with the seed-61 policy trained on 147,560 decisions with full title balance. It used the same TH14 build, Normal rank, Reimu A shot, geometric dodge, masked aim, emergency bomb logic and shadow world checkpoint as the first one-credit clear. The native restart was uncontrolled in its exact timing. The first controlled frame was 11 in the original run and 18 in the follow-up.

| Story trial | Stages cleared | Continues | Misses | Bombs used | Displayed score | Decisions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Original motion-history SFT | 6 | 0 | 6 | 23 | 296,778,700 | 25,702 |
| Expanded, fully title-balanced SFT | 6 | 0 | 5 | 22 | 253,575,700 | 26,359 |

The original executable entered Ending/Credits in both trials. The updated checkpoint used one fewer miss and one fewer bomb and recorded a lower score. Firing during vulnerable boss frames changed from 92.92% to 91.09%. On 3,047 exact four-frame transitions from the follow-up, the shadow world model scored 0.967 near-player occupancy AP against 0.930 for ballistic extrapolation. Each configuration has one complete trial, so the comparison establishes that each can clear on one credit.

## Native launch and capture follow-up

Closed both completed game sessions and retried TH16 through its official Steam launch. The game exited before gameplay and Windows logged an access violation in the game executable. Launching the installed executable and testing its default configuration also exited before gameplay. Restored the two original configuration files with matching SHA-256 hashes. The executable produced no paired TH16 data in these attempts. The exclusive TH13 Steam launch entered its Replay menu successfully.

## Policy width and action-only scale

Compared SFT hidden width 512 with 1024 at seeds 61, 97 and 137. The 147,560 training decisions, equal title weights, 20-epoch budget and replay splits were fixed. Total policy parameters including the value stack increased from 781,552 to 2,601,200. Mean direction-change accuracy rose from 30.15% to 32.43%, with gains at two of the three seeds. Mean overall direction accuracy fell from 64.18% to 62.30%. The recorded native story benchmark uses width 512.

The source corpus contains 2,000 complete action replays with 209,326,603 native input frames. Its training split contains 45,498,703 non-overlapping four-frame button windows. The action-only prior consumed 45,161,344 distinct windows per epoch after constructing contiguous 64-step sequences. A ten-epoch H100 run processed 451,613,440 window presentations and selected epoch 8 using validation direction changes. On 2,549,888 test windows, direction-change accuracy moved from 6.80% after one epoch to 11.43%, and overall direction accuracy moved from 70.97% to 71.80%. This prior takes previous button inputs and replay metadata. State-conditioned SFT uses the separate native state–action pairs.

## Adding a second full Lunatic replay

Captured six stages of TH13 1.00c Lunatic Marisa from the original executable. The training replay `th13_ud0bf1.rpy` contributed 20,576 paired decisions. Per-stage moving-input alignment ranged from 98.96% to 100%. The new snapshot has 168,136 training decisions from 44 stages. Both prior validation and test splits have identical shard fingerprints in the new snapshot.

| Training on the updated snapshot | Mean direction accuracy | Mean direction-change accuracy | Selected epochs for seeds 61, 97, 137 |
| --- | ---: | ---: | --- |
| Frozen 147,560-decision reference | 64.18% | 30.15% | 2, 4, 3 |
| 168,136 decisions, random initialization | 61.62% | 14.79% | 3, 2, 2 |
| Low-rate warm start | 63.17% | 18.47% | 2, 2, 0 |
| Warm start with movement anchor | 64.22% | 29.01% | 2, 0, 0 |

Randomly initialized runs used the same 20-epoch recipe and equal title weighting as the reference. Warm-start runs used learning rate 0.00004, the updated corpus, the same three initial checkpoints, a 16-epoch maximum, and patience six. The anchored variant added a weight-four KL term to retain each initial checkpoint's movement distribution on the sampled inputs. The starting checkpoint participated in model selection. The updated data remain available for future training, and the native story controller retains the 147,560-decision checkpoint.

Seed 137 in the anchored experiment reached its normal early stop before an SSH timeout interrupted artifact retrieval. The recovery fetched the saved checkpoint and epoch log with verified hashes. The original failure and subsequent evaluation have separate run records.
