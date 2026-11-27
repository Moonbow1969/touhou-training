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
