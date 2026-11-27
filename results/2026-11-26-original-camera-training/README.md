# Human replay cameras and original-game online PPO

The previous portable policy rarely changed its movement when we substituted another screen from the same credit. This run learns the action from synchronized original human gameplay pixels. All three original Normal trial workers use four heads (nine directions, focus, shoot and Bomb), original lives and terminal states. The seed controls policy sampling; the native game RNG is independent. Each tested pair runs on the same physical machine.

Two training paths answer different questions. The portable continuation initializes from PPO3 and collects 16 new full credits per title. Its plain PPO control and its demonstration-regularized PPO use identical behavior data. The demonstration loss uses actual replay actions together with frozen camera-policy probabilities. Separately, a full-resolution CNN initializes directly from human camera SFT. It acts in fresh original games and receives its own PPO update from eight complete credits per title, with an independent six-seed paired test. These paths have different starting weights and data budgets.

The spatial encoder takes a 96 × 112 grayscale playfield and the absolute difference from the preceding control observation. Its native training split uses actual TH10, TH12 and TH13 player positions as labels for pretraining. At evaluation or gameplay it receives only the camera. After 2,200 optimization steps its median held error in original-pixel units was 5.15 for TH10 (550 frames), 8.94 for TH12 (14 hand-labeled frames) and 4.74 for TH13 (2,465 frames). Repeated frames in a credit are correlated. Localizing the player is a separate endpoint from surviving a level. [Training loss and exact source fingerprints](encoder/report.json) · [Frozen encoder](encoder/encoder.pt)

[Portable training and comparison design](design.json) · [Full-resolution PPO protocol](camera-protocol.json)

### Fourth portable PPO update

The plain PPO arm trained on 48 new uninterrupted original-game credits, 16 per title and 36,488 policy decisions in total. All four epochs satisfied the per-title KL bound. Each trajectory includes the behavior actions, likelihoods, original reward and terminal mask. Re-scoring the saved behavior is a prerequisite for each update. The demonstration arm shares all 48 trajectories with the plain arm. The evaluation compares them on the same 12 controller seeds per title and matches physical hosts. [Behavior traces and checksums](batch4/credits.json) · [Training log](ppo4-control/report.json) · [Portable weights](ppo4-control/policy.bin)

### Replay synchronization and Bomb timing

TH13 replay sampling pairs the camera with the following native tick for movement, focus and shooting. A Bomb tap is retained anywhere within the next four-tick action window. Replays AIZ and MON each achieved exact agreement on all moving input words in the best clock alignment. On 778 high-confidence moving image pairs from AIZ, independent image registration agreed with the native displacement direction 98.84% of the time. [Camera and action schema](human/schema.json) · [AIZ audit](human/audit/th13_ud0c5d-s1.json) · [MON audit](human/audit/th13_ud0cd3-s1.json)

For TH12, 30 of the 32 spare-life losses in the PPO3 training batch had at least one Bomb in stock half a second before the HUD confirmed the loss. The mean Bomb probability in the preceding second was 0.173%, compared with 0.313% elsewhere. This measures policy behavior around HUD confirmation. Testing the effect of a reward change requires changing the reward and comparing fresh game outcomes. [Event counts, window definition and per-credit data](th12-training-bomb-timing.json)

### First human-camera policy

The visual SFT ran 900 updates on 4,492 usable decisions from two TH13 Stage 1 players, AIZ and MPS16. On 2,243 observations from held player MON it predicted the next movement on 1,192 decisions (53.1%). A half-replay circular shift of the screens reduced that to 451 decisions (20.1%). At the 543 direction changes it predicted 162 correctly (29.8%), versus 96 (17.7%) after shifting the screens. PPO3 received the actual preceding expert actions at evaluation. Its 1,688 correct movements included only 23 correct changes, and zero correct starts or turns. MON had two distinct Bomb onsets. These measurements cover one player, one title and one stage. [Independent figure and counts](held-player-first/held-camera.png) · [All metrics and source hashes](camera-sft/report.json) · [900-update training trace](camera-sft/loss.jsonl) · [Camera policy for the original games](camera-sft/policy.ts)

The portable demonstration-regularized PPO4 checkpoint passed the KL guard in all four epochs on the same 36,488 original decisions as the plain PPO4 control. The original-game comparisons evaluate this checkpoint separately from the full-resolution camera policy. [PPO4 optimizer record](ppo4/report.json) · [Demonstration protocol](protocol.json) · [Portable checkpoint](ppo4/policy.bin)

The first completed portable comparisons covered 12 matched seeds each in TH10 and TH12. PPO3 and PPO4 reached Stage 2 on zero credits in either title. In TH10 mean time to Game Over changed from 38.31 s to 43.61 s, with a host-stratified paired 95% bootstrap interval for the difference of [−0.84, +12.50] s. In TH12 it changed from 55.28 s to 46.86 s, interval [−18.68, +0.87] s. Mean remaining TH12 Bombs at Game Over changed from 1.83 to 2.00, with zero spare lives throughout. TH10 uses Power for spirit strikes. On the same TH10 seeds PPO4 differed from the plain PPO4 control by +1.10 s, interval [−5.30, +7.81] s. [Matched original endpoints and action counts](portable-interim/credits.json) · [Per-seed comparison](portable-interim/ppo4.json) · [Demonstration ablation](portable-interim/th10-regularized-vs-plain.json)

### Rollout throughput and storage

Each original game sends a 96 × 112 grayscale observation to its own host over a Unix socket. The host verifies the frozen TorchScript hash and returns the four action-logit vectors and scalar critic. The actor samples the controls and records the selected log probabilities locally. The learner receives compressed trajectories after each full credit and re-scores them before PPO. For TH10 the first 8 camera-SFT training credits contained 3,342 decisions. Momentum averaged 3.46 ms for local inference; the observed median decision interval was 124.84 ms with a 120 ms game hold. The credits transferred 34,991,047 bytes. [Measured timing and transport](runtime/runtime-cost.json)

Eight complete TH10 screen sequences, including final successor images, contained 3,350 frames. In an offline lossless comparison, gzip used 32.60 MB for the pixels, horizontal byte differences plus Zstandard used 27.08 MB, and lossless H.264 used 24.48 MB. All decoded frame bytes were checked against the originals. The native runs used their original archive format. [Byte counts and single-core timings](runtime/lossless-camera-codecs.json)

### Reproducing a learner update

Run the following commands from the repository root with the experiment directory in `RUN`. The first command rebuilds the compact learner input from native credit receipts, original pre-action images and actor log probabilities. The second validates the original behavior before training any parameters. Use an empty output directory.

```sh
python -m scripts.pack_camera_ppo_rollouts --root "$RUN" --behavior "$RUN/camera-sft/policy.ts" --protocol "$RUN/camera-protocol.json" --out "$RUN/repacked"
python -m scripts.train_full_resolution_camera_ppo --manifest "$RUN/repacked/credits.json" --protocol "$RUN/camera-protocol.json" --sft "$RUN/camera-sft/camera-policy.pt" --behavior "$RUN/camera-sft/policy.ts" --out "$RUN/retrained"
```

A subsequent PPO generation uses `--parent-actor-critic` to restore the previous critic together with the camera and action heads. Its 48 training seeds and 36 independent matched evaluation seeds are registered separately. [Second camera PPO protocol](camera-ppo2-protocol.json)

### First full-resolution PPO update

The frozen two-player human SFT generated 16,461 decisions in 24 complete original credits, eight each in TH10, TH12 and TH13. Replaying every pre-action image through the parent reproduced the selected native action log probabilities to a maximum absolute error of 0.00000489. All four PPO epochs passed the KL bound. Final joint KL was 0.00863 in TH10, 0.00915 in TH12 and 0.00396 in TH13. The training batch rebuilt from native receipts produced an identical 24-credit manifest. [Learner log](camera-ppo/report.json) · [Frozen CNN actor and critic](camera-ppo/policy.ts) · [Original behavior and human SFT input downloads](https://github.com/Moonbow1969/touhou-training/releases/tag/camera-training-2026-11-26)

For MON and SHAOfLan the expert changed movement on 24.2% and 24.9% of observations. Conditional on that same expert history and camera, sampling from the first SFT would change movement on 56.5% and 63.5% respectively. Its actual native runs changed movement on 60.0%, 69.8% and 67.7% of decisions across TH10, TH12 and TH13. The proposal used to train SFT oversampled direction changes. A matched 900-update correction kept the exact same data and draws and importance-weighted only the movement loss back to natural within-replay frequency. Held movement NLL improved from 1.575 to 1.360 on MON and 1.952 to 1.764 on SHAOfLan. Correct decisions at true changes fell from 330 to 313 out of 1,172. The corrected policy still sampled movement changes at 50.5% and 58.8%. These two held players informed this follow-up hypothesis. A fresh original-game comparison tests its practical effect. [Behavior profile](camera-batch/control-profile.json) · [Matched comparison figure](frequency-ablation/held-camera.png) · [Frequency ablation protocol](frequency-ablation-protocol.json)

The first camera PPO test completed all 36 original credits. Each pair shares a physical host, game build and policy-sampling seed. All credits ended in Stage 1 with no spare lives. Stock reported below is the mean at Game Over. TH10 uses Power to cast spirit strikes.

| Original Normal trial | Stage 2, SFT → PPO | Mean wall seconds, SFT → PPO | Terminal resource, SFT → PPO | Successful casts across six credits |
|---|---|---|---|---|
| TH10 | 0/6 → 0/6 | 44.94 → 53.24 | 0.00 → 0.00 Power | 11 → 15 |
| TH12 | 0/6 → 0/6 | 53.77 → 62.08 | 0.83 → 0.50 Bombs | 18 → 23 |
| TH13 | 0/6 → 0/6 | 71.44 → 83.78 | 0.33 → 0.00 Bombs | 31 → 34 |

TH13 also exposed the stage script timer. PPO reached a mean 382 additional Stage 1 ticks on these pairs, with a paired 95% bootstrap interval of [−295, +1,141] ticks. Rendering and dialogue affect elapsed time. None of the 18 test seeds established a Stage 2 advantage. [Independent endpoints, actions and resources](camera-evaluation/credits.json) · [Matched figure](camera-evaluation/paired-full-credits.png) · [Native script timer comparison](camera-evaluation/th13-native-progress.json) · [TH10 full-credit video](th10-video-754001/th10-normal-754001-full-credit.mp4) · [TH12 full-credit video](th12-full-credit-764001.mp4)

Expanding replay SFT to six distinct players provided 14,215 usable training observations, covering Normal, Hard and Lunatic. The model received 3,000 updates (192,000 sampled observations). On two held players, all-row movement NLL changed from 1.575 to 1.486 and from 1.952 to 1.765. Correct direction changes fell from 330 to 312 of 1,172. Both the data and update budget grew. [Frozen expanded SFT](camera-sft-expanded/policy.ts) · [Same-player evaluation](expanded-compare/held-camera.png) · [Source replay catalog](resources/replay-sources.md)
