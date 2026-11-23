# 261123 Late-stage practice and title transfer

## TH14 late stages

The first story retest lost window focus during Stage 1 at frame 3,198. The new recorder preserved the full traceback. This incident establishes the cause of this retest’s exit. The earlier Stage 4 story failure has no surviving traceback.

Added a foreground wait that holds the original game paused and releases its keys while another window is active. The controller can resume at the recorded frame once TH14 returns to the foreground. The shared Windows desktop can deliver gameplay keys to one game at a time.

TH14 Normal Reimu A Stage 5 Practice returned the original game’s **STAGE CLEAR** result. The checkpoint stack took control at frame 212, reached the last logged frame 18,512, lost two lives, and used seven bombs. The displayed score was 56,162,230. Run `20261123T091712Z-th14-stage5-practice-94da4667` saved the report, console, source and artifact fingerprints. Practice starts with its own life, bomb, and power inventory.

Stage 6 Practice also returned **STAGE CLEAR**. The same stack took control at frame 207, reached the last logged frame 26,794, lost three lives, and used ten bombs. The displayed score was 76,859,610. Run `20261123T092444Z-th14-stage6-practice-1a4b29cc` preserves the full console and results. The deepest verified one-credit story run remains Stage 4.

## Leave-one-title-out TH17 evaluation

Trained a motion-history, stage-clock SFT policy for 20 epochs on H100 using 47,117 paired decisions from TH13 and TH14. Model selection used 2,288 TH14 validation decisions. TH17 was excluded from both sets. Evaluation used 51,421 decisions from 12 TH17 Normal stages across two Marisa shot configurations and two replay groups.

| Model | All directions | While moving | Direction changes |
| --- | --- | --- | --- |
| Previous direction | 59.91% | 56.87% | 0.00% |
| SFT with TH17 in training, other players held out | 58.66% | 56.37% | 17.17% |
| SFT trained and selected on TH13/14 only | 57.90% | 62.66% | 31.65% |

The reference trained on more examples and a different distribution. This tests frozen imitation on original replay states. Native survival and clears require a separate controller trial. The paired test covers Normal. TH15, TH18 and TH19 remain outside these training runs.

Training run `20261123T092515Z-leave-title-out-th17-sft-a189fc47` recorded the input bundle SHA-256, code and trainer configuration, full epochs, and checkpoint. Evaluation run `20261123T092608Z-th17-zero-shot-heldout-evaluation-104e8358` checked that the candidate’s training and validation manifests excluded TH17.

## Native TH17 transfer

Ran four independent native trials on TH17 1.00b, Normal Reimu Wolf, Stage 1. Each trial began at native frame 6 with two spare lives and ended at the first miss or 900 decisions. The SFT checkpoint was trained and selected on TH13/14. The frozen PPO dodge head was trained on TH14. The analytic and PPO arms used identical bullet and enemy-body clearance features with the TH14 motion model. Neither arm used boss alignment or the learned world model.

| Controller | Last logged frame | First miss | Moving decisions | Display score gain |
| --- | ---: | ---: | ---: | ---: |
| Stationary shooting | 2,450 | 2,450 | 0 | 41,210 |
| Frozen SFT | 2,450 | 2,450 | 0 | 38,060 |
| Frozen SFT and geometric action selection | 3,606 | Censored at decision limit | 41 | 51,990 |
| Frozen SFT and PPO action selection | 3,606 | Censored at decision limit | 74 | 12,960 |

This pilot supports transfer of collision-aware control through the benchmark horizon. PPO did not demonstrate an extra survival gain over geometric selection at that cutoff, and the geometric trial scored higher. Each row is one run. Stage clears and harder difficulty transfer remain unmeasured. The pure SFT episode selected an idle direction even though the offline candidate predicted some expert turns on expert-controlled states.

One analytic attempt attached to frame 4,749 during the native retry transition. It was excluded from the benchmark and its log was retained. The evaluator now rejects stale starts. TH15 Steam startup did not produce a gameplay window during its attempts, so the first live unseen-title benchmark used TH17.
