# State-conditioned SFT and pure-policy PPO

## Current evidence

The raw corpus contains 2,000 decoded replays across TH13, TH14, TH16 and TH17, with 209,326,603 native input frames. The action-only prior trained on over 45 million four-frame windows per epoch. The state-conditioned policy needs an observation paired with each label. The legacy parent used 147,560 training decisions, including one target-configuration TH14 Normal Reimu A Stage 1 shard with 2,415 decisions. The intermediate width-ablation snapshot had 168,136 decisions. The latest frozen snapshot has 232,697 training decisions from TH13, TH14 and TH17, including 64,561 new native TH14 pairs from 18 stages. [Capture acceptance and snapshot](../results/2026-11-23-th14-full-policy/sft-expanded-admitted-captures.json)

The parent trained for 20 epochs from random initialization. Its best validation loss was 1.292 at epoch 2 and rose to 1.567 at epoch 20. The earlier 20- versus 60-epoch matched experiment selected identical weights. The new trainer records the weighted training objective, validation NLL per head and positive support. The training objective and validation NLL have different weights. Compare trends within each objective. [Earlier experiment](devlog/261123-overnight-recipes.md#progression-benchmark-and-early-stopping), [new epoch logs](../results/2026-11-23-th14-full-policy/sft-button-and-width-ablation.json)

- The TH14 validation replay scores 72.81% overall direction accuracy. For exact next-tick changes in held-out labels, the model predicts 2/58 starts and 0/95 active turns. Temporal alignment passed an independent displacement test. Of 850 informative four-tick windows, 99.53% produced the expected displacement within half a game unit. Shifting the labels one frame gave 16.24% or 15.29%. About 26% of training windows contain more than one direction, including 53% of adjacent-window starts. [Label and decoder audit](../results/2026-11-23-th14-full-policy/sft-label-decoder-audit.json), [tick alignment](../results/2026-11-23-th14-full-policy/sft-label-tick-alignment-audit.json)
- TH14 supplies zero Bomb positive windows in the paired parent. The source stages for those shards also have zero Bomb keypresses. The full validation split has zero Bomb positives. Reweighting button classes to their natural frequency improved validation NLL from 1.240 to 1.218 over three seeds, but TH17 direction-change accuracy fell from 29.83% to 26.67%. Against 99 Bomb positives from separate TH17 test players, the default weighting produced 0, 4, and 7 true positives across seeds. The unweighted head produced zero true positives at a 0.5 threshold for all three seeds. Keep the parent recipe for downstream controls. [Bomb support and test](../results/2026-11-23-th14-full-policy/sft-bomb-th17-test.json), [source button counts](../results/2026-11-23-th14-full-policy/sft-bomb-coverage-by-split.json)
- On a TH14 validation replay, shuffling projectile sets within an eight-second stage interval and similar projectile-count groups changed 3.28% of greedy directions. Direction accuracy moved from 72.81% to 72.33%. This tests dependence on projectile arrangement on that replay. Observed positions, velocities, history, stage timing, and other context were still available. Removing stage-clock features in the matched width-512 comparison improved mean validation movement NLL from 0.906 to 0.876. TH14 direction changes rose from 9.57% to 11.14%. The TH17 diagnostic slice rose from 13.10% to 20.58%. Neither condition identified movement starts on the single TH14 validation replay under this slice definition. [Projectile sensitivity](../results/2026-11-23-th14-full-policy/sft-projectile-dependence-audit.json), [clock ablation](../results/2026-11-23-th14-full-policy/sft-button-and-width-ablation.json)
- We also gave half of the four-tick soft movement target to the first tick and kept the same 512-width model without stage clock. On three matched seeds, TH14 first-tick NLL improved from 0.714 to 0.678. Immediate-start accuracy stayed at 3.45% and immediate-turn accuracy stayed at 0.70%. This change improves likelihood but leaves the rare decisions that motivated it almost unchanged. It stays outside the expanded SFT queue. [First-tick ablation](../results/2026-11-23-th14-full-policy/sft-first-tick-ablation.json)

### Capacity on a fixed split

Train all widths on the same 168,136 aligned decisions, use the same four validation stages for checkpoint selection, and use the same three seeds. TH17 test replays are from two excluded players on a title present in training. The project has inspected these TH17 results repeatedly, so use them for diagnosis and keep the newly captured TH14 test replay sealed for final comparisons. Parameters include the unused-in-SFT value branch, so they overstate the deployed actor's size.

| Actor MLP width | Total parameters | Mean best validation NLL | TH17 test direction accuracy | TH17 test direction-change accuracy |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 781,552 | 1.250 | 61.96% | 13.10% |
| 1024 | 2,601,200 | 1.265 | 61.49% | 16.05% |
| 2048 | 9,386,224 | 1.297 | 61.63% | 14.63% |

The wider models reduced the weighted training objective sooner, while mean best validation NLL increased. This is evidence of a limited-data generalization gap for this pooled encoder and training snapshot. The 1024 arm improved the TH17 direction-change slice and stays a candidate after the training set grows. In this test even the 2048 actor still received the same capped projectile set and pooled summary. Increasing MLP width leaves that information bottleneck unchanged. [Runs and per-seed metrics](../results/2026-11-23-th14-full-policy/sft-button-and-width-ablation.json)

The expanded 232,697-decision snapshot was also tested on Momentum's RTX 5090. Six runs compared widths 512 and 1024 with all other training settings fixed, three matched seeds, and validation-based checkpoint selection. Width 1024 increased validation NLL for every seed. Mean NLL was 1.278 at width 512 and 1.317 at width 1024. Overall movement accuracy was 75.05% and 74.06%. Active-direction accuracy was 64.39% and 64.96%. The small active-direction gain varied by seed. These are shared-validation results. The independent TH14 replay and live gameplay were outside this width comparison. [Six training traces, validation metrics and hashes](../results/2026-11-23-th14-full-policy/sft-expanded-width-5090.json)

### Independent TH14 replay

The reserved original-game replay contributes 26,279 evaluation pairs across all six Normal Reimu A stages from one excluded player. All nine checkpoints were selected on the unchanged validation split before scoring this replay. Width was 512 throughout. Each number below is the mean across training seeds 61, 97 and 137. [Training curves and checkpoint fingerprints](../results/2026-11-23-th14-full-policy/sft-expanded-training-curves.json), [test counts and per-seed deltas](../results/2026-11-23-th14-full-policy/sft-independent-th14-final-matched-seeds.json)

| Cohort | Four-tick direction | Direction while moving | Immediate start | Immediate turn | Bomb recall | Bomb false positive rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy with stage clock | 65.29% | 53.44% | 1.79% | 2.90% | 0.23% | 0.001% |
| Expanded with stage clock | 66.53% | 59.33% | 2.83% | 3.48% | 15.19% | 1.68% |
| Expanded with stage clock omitted | 67.23% | 61.39% | 3.07% | 3.54% | 14.74% | 1.25% |

The test contains 13,952 moving targets, 1,250 immediate starts, 574 immediate turns and 147 Bomb-positive windows. An immediate event uses the expert input at the first upcoming native tick. Bomb recall measures greedy button prediction in each positive window. For the expanded cohorts, mean Bomb-onset recall was 5.33% with clock and 2.00% with clock omitted, over 50 observed onsets. The result describes one expert trajectory and supports no player-population interval. Native replay capture and live policy control have separate endpoints. [Metric definitions and counts](../results/2026-11-23-th14-full-policy/sft-independent-th14-final-matched-seeds.json)

Reproduce the nine-checkpoint offline report with the frozen private artifacts and their existing run layout.

```powershell
python -m scripts.summarize_expanded_sft_independent --legacy-root RUNS --expanded-root RUNS/expanded --test-data RUNS/expanded/test/th14_ud0559 --out runs/th14-independent.json
```

### Native Stage 1 pure SFT

The three seed-61 checkpoints completed eight rotated blocks in stock TH14 1.00b Normal Reimu A. Each arm sampled its own actions using common policy seeds 101 through 108. Geometry, aim correction and automatic Bomb were disabled. Every episode started at stage frame 18 and ended at the first miss. [Complete native report and checkpoint fingerprints](../results/2026-11-23-th14-full-policy/sft-expanded-native-pure.json)

| Cohort | Median first-miss frame | Clears | More frames than legacy within block | Observed horizontal responses | Bombs used |
| --- | ---: | ---: | ---: | ---: | ---: |
| Legacy with stage clock | 758 | 0/8 | — | 803/804 | 1 |
| Expanded with stage clock | 494 | 0/8 | 2/8 | 738/761 | 7 |
| Expanded with stage clock omitted | 484 | 0/8 | 3/8 | 756/773 | 8 |

The expanded policies improved several offline direction metrics and died earlier in the original game on this sample. Horizontal response counts show physical motion in all three arms. The response denominator includes commands near playfield edges. The two expanded arms used more Bombs over fewer played frames. The full per-episode traces support further action and score analysis. This small live comparison supports using the legacy checkpoint as the frozen common PPO start. The next experiment uses fresh sampling seeds and keeps the existing hybrid backbone fixed.

The all-stage offline mean hides an opening-stage regression for the deployed seed. On the sealed Stage 1 replay, overall greedy movement agreement at seed 61 was 2,180/3,292 for the legacy checkpoint, 2,114/3,292 for expanded with clock, and 2,096/3,292 for expanded without clock. Agreement on the 1,690 already-moving labels rose from 858 to 987 and 1,015, respectively. The same three checkpoints correctly predicted 1, 1 and 4 of 218 immediate starts. Future checkpoint selection should include a separate development Stage 1 slice for immediate starts and turns, with companion accuracy on holds. [Per-stage counts for the sealed credit](../results/2026-11-23-th14-full-policy/sft-independent-th14-final-matched-seeds.json)

The native policies requested shooting on 98.33%, 88.55% and 87.62% of their decisions, and focus on 53.03%, 37.10% and 24.34%. They requested Bomb on one, 16 and 17 decisions. These pooled rates give longer episodes more weight. They describe what the policy requested and do not identify which head caused earlier misses. Offline replay scoring uses greedy decisions. A frozen-head native crossover could compare sampled and greedy buttons with a separate, shared movement RNG stream to isolate the effect of button decoding. [Native action histogram counts](../results/2026-11-23-th14-full-policy/sft-expanded-native-pure.json)

On the expanded snapshot and common development split, weighting movement-change targets 4× raised immediate turn accuracy from 1.40% to 5.96% across the three matched seeds. Immediate starts stayed at 2.87%. Idle-hold accuracy fell from 91.73% to 86.84%, active-hold accuracy fell from 82.25% to 77.84%, and shared validation NLL rose from 1.278 to 1.474. The weighted target compares the upcoming modal direction with the last measured displacement direction. The diagnostic uses the exact upcoming tick and previous native input. The sealed replay stayed outside this comparison. Keep movement starts, turns, and holds as separate endpoints when changing the movement objective. [Development-only counts and three training traces](../results/2026-11-23-th14-full-policy/sft-moving-transition-development.json)

An equal-size temporal context comparison replaced the three recent displacement intervals with cumulative velocities over 4, 12 and 36 native frames. Both arms retained the same 781,552 parameters and four-frame actions. The longer context increased mean validation NLL from 1.278 to 1.291 and reduced overall validation movement agreement from 75.05% to 74.34%. Immediate starts on the TH14 development replay stayed at 2.87%. This tests the selected cumulative velocity features on the same data. It leaves projectile histories and recurrent policies for separate experiments. [Per-seed temporal-context diagnostics](../results/2026-11-23-th14-full-policy/sft-motion-horizon-development.json)

### Native full-policy PPO pilot

The pilot initialized all four PPO actor heads from the legacy seed-61 SFT checkpoint. Its initial actor logits matched SFT exactly. The original game supplied 8,192 training decisions over a 3,300-decision Stage 1 horizon, with a survival-time reward and a fresh value branch. Eight evaluation blocks rotated frozen SFT, pure PPO and the existing deterministic hybrid. SFT and PPO sampled with policy seeds 211–218. The game RNG was unmanaged and geometry, aim correction and automatic Bomb were disabled for both learned arms. [All 32 training updates and 24 evaluated episodes](../results/2026-11-23-th14-full-policy/ppo-long-8192-sft-paired.json)

| Controller | Median first-miss frame | Clears | Reached frame 1,000 | Bombs used |
| --- | ---: | ---: | ---: | ---: |
| Frozen SFT | 570.5 | 0/8 | 3/8 | 0 |
| Pure PPO, 8,192 decisions | 1,354.5 | 0/8 | 6/8 | 10 |
| Hybrid baseline | Censored by clear in all 8 | 8/8 | 8/8 | 0 |

PPO reached a later frame in 7/8 paired blocks, with a median within-block gain of 518 frames. Its later first misses accompanied ten Bombs used across eight runs. The trial does not estimate how much of the survival gain came from dodging or Bomb use. Training ran at 14.06 decisions per second, and native stepping plus resets accounted for 96.7% of wall time. The timing histogram contains 8,184 exact four-frame transitions, four three-frame transitions and four five-frame transitions.

The 16,384-decision checkpoint resumed the exact saved PPO state at 8,192 decisions. Eight new paired blocks used policy seeds 401–408. Pure PPO won 7/8 again. Its median first miss was 1,430 frames, compared with 666 for frozen SFT. The median paired gain was 598 frames, with 15 Bombs used by PPO and zero by SFT. Both arms were 0/8 on no-miss Stage 1 clears. The two budgets were evaluated using separate sampling seed cohorts, so they each estimate a gain against SFT and leave the budget effect unmeasured. [Continuation trace and 16 original episodes](../results/2026-11-23-th14-full-policy/ppo-long-16384-sft-fresh-paired.json)

The two saved PPO checkpoints then ran head to head with policy seeds 701–708 and rotated order. Median first miss was 704 at 8,192 decisions and 1,194 at 16,384 decisions. The longer run won 5/8 paired blocks and the median paired difference was 564 frames. Both reached frame 2,000 in 2/8 and cleared in 0/8. Bombs used rose from five to eleven. This sample supports a survival signal and leaves the effect of additional training on later progression uncertain. [All 16 episodes and paired deltas](../results/2026-11-23-th14-full-policy/ppo-budget-8192-vs-16384.json)

## Execution order and gates

1. **Capture matching TH14 data.** The checked queue includes twelve Normal Reimu A, six Hard, and six Lunatic training credits, with distinct replay groups and 599,770 candidate four-tick windows before alignment. All 24 stages in the first tranche passed the minimum 95% moving alignment and 90% coverage. Its three training credits added 64,561 decisions including 591 Bomb-positive windows. The independent test replay stayed outside training and model selection. [Capture queue and hashes](../results/2026-11-23-th14-full-policy/sft-capture-queue.json), [completed first tranche](../results/2026-11-23-th14-full-policy/sft-expanded-admitted-captures.json)
2. **Measure the data effect in width 512.** Six expanded runs completed, with three seeds, a 20-epoch maximum and patience six, first with the current architecture and then with the stage clock omitted. The sealed evaluator verified replay and player exclusion and reproduced the same report from frozen checkpoints. The native comparison completed all eight rotated evaluation blocks with seed-61 checkpoints. Report native outcomes by episode and keep later cross-title and difficulty evaluations distinct from this TH14 Normal result.
3. **Change one modeling variable at a time.** Test a first-tick or motion-transition auxiliary loss at the four-tick control rate. If needed, compare actual two-tick captures with the same temporal history in native frames. Compare an entity-aware encoder and direct enemy and laser observations as separate changes. Run the width comparison again only on the same expanded data, split, optimizer and selection procedure. Increase width when improvements recur in validation and pure native gameplay across seeds.
4. **Warm-start full-policy PPO from the frozen SFT.** Keep a pure frozen SFT control and give PPO the exact same initial policy. Start a new task with a long enough horizon for a Stage 1 clear and record stage transitions, first miss, score, reactions and Bomb use. The former 512-step task measures early survival. The new comparison uses a first-miss episode with up to 3,300 decisions and enough native restarts to measure variance. Rule-based and hybrid scores remain separate comparative baselines. Keep the dataset, policy version, random draws, and build fixed for each comparison.

The capture pipeline uses the licensed Steam 1.00b runtime and its own replay menu and decoder. One capture window is currently verified. The multi-instance 0.01b trial has a different executable and observation schema. Test faster replay playback or an additional licensed original worker on the same replay before admitting its observations. Store raw archives and model weights in the run scope. Publish hashes, training curves, acceptance metrics and native evaluation traces.

Run the pure-SFT native comparison on the verified TH14 1.00b Normal Reimu A Stage 1 instance. The evaluator rotates checkpoint order across policy sampling seeds, saves every completed episode, and resumes the same schedule at its first uncommitted episode. The policy seed controls sampling and leaves the original game's RNG unmanaged.

```powershell
python -m scripts.evaluate_sft_native --policy parent=parent.pt --policy expanded=expanded.pt --policy no-clock=no-clock.pt --seeds 101 102 103 104 105 106 107 108 --parakit PATH --out runs/th14-pure-sft-native
```
