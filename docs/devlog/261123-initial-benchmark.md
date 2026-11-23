# 261123 Initial benchmark and development record

## Summary

The TH14 1.00b Normal Reimu A controller reached native Stage 4 on one credit twice. The longest logged trial reached frame 23,694 of Stage 4 with one miss and seven bombs used. The Python controller exited there, and the original game continued running. The complete traceback for that run is unavailable. Full-game and Lunatic clear rates remain unmeasured.

## Data and frame timing

- Verified 2,000 complete action replays for TH13, TH14, TH16 and TH17, containing 209,326,603 native input frames. The training split has 1,738 replays and 45,498,703 non-overlapping four-frame input windows before training eligibility filters.
- Corrected replay decoding when a dictionary backreference crosses the circular boundary or reads a byte written earlier in that reference. Rebuilt and audited labels from the original game states. An additional TH17 Lunatic run contributed about 20,700 verified decisions, bringing the paired training snapshot to 126,781 examples across 32 stages in TH13, TH14 and TH17.
- A separate action-only prior used over 45 million eligible windows. Its direction accuracy was 70.97% on held-out action labels, compared with 70.49% for previous-direction persistence. It achieved 6.80% accuracy at direction changes. It supplies evidence about the action sequence corpus and has no verified native policy gain.
- Kept decisions at four native frames. On the same held-out set, adding coarse 16/32/64-frame slopes to the existing short history moved direction-change accuracy from 18.45% to 16.49%. Keep the short history as the default for this encoder.
- TH16 failed to start in a normal local Steam launch. A Wine capture worker reached the normal Steam handoff, and no TH16 state–action capture was produced. TH15, TH18 and TH19 remain reserved for future title-level testing.

## Native ablations

| Change | Endpoint | Result |
| --- | --- | --- |
| Soft-label SFT versus stationary input | Stage 1, first-miss median, 3 trials each | 506 vs 486 frames. Zero clears in both groups. |
| Direction-change-weighted SFT | Same native test | 418 frames. Held-out direction-change accuracy reached 32.60%, while native survival fell versus 506 frames for the previous soft-label SFT. |
| Analytic clearance added to SFT | First-miss median, 2 trials per arm | 422 → 2,384 frames. |
| PPO added to a pre-existing imitation dodge head | First-miss median, 3 trials per arm | 3,470 → 3,626 frames. Zero clears in both groups. |
| Boss alignment added to frozen PPO dodge controller | Stage 1, 2 trials per arm | 0/2 → 1/2 clears. Damage per vulnerable second increased from 119 to 506 in this pilot. |
| Straight-laser and enemy-body checks, supervised aim and emergency bombs | Integrated story controller | Cleared stages 1–3 in two runs and reached Stage 4. Component-level increments for these later changes are still unmeasured. |

The World Model’s Stage-weighted held-out 24-frame occupancy AP near the replay player was 0.646, compared with 0.539 for constant-velocity extrapolation on the same evaluation. The live controller used analytic collision prediction. The model has yet to demonstrate a live clear or survival gain.

## Engineering changes and incidents

Native stage transitions are recognized by the stage marker so that the temporary results pause does not terminate a story trial. Boss damage accounting excludes large HP drops associated with phase timeout. The controller tracks emergency bomb commands and actual inventory consumption independently. A Stage 4 dialogue stall led to a change that pulses Z during dialogue. The manual Stage 4 reproduction and longer one-credit retest remain pending.

The project added a run recorder, per-epoch streams for SFT and world models, PPO CSV metrics, H100 console persistence, and persistent evaluation exceptions after the early experiments. Later records should include the original streams. Earlier experiments retain the JSON reports and checkpoints that were available for archival.

## First recorded H100 run

Trained the stage-clock SFT policy for 20 epochs on the same 126,781 pairs, changing only the within-title player balance to 0.5. Run `20261123T085800Z-th14-sft-stage-clock-player-balance-5c1f74e9` saved the full console, all epoch metrics, the checkpoint, and input and artifact hashes. On 51,421 unseen paired decisions, overall movement accuracy moved from 58.66% to 59.13% and direction-change accuracy from 17.17% to 12.72%. The previous checkpoint remains the reference. The held-out evaluation is run `20261123T090038Z-sft-stage-clock-player-balance-heldout-6e1c978c`.

## Stage 4 Practice diagnostic

Run `20261123T090358Z-th14-stage4-practice-diagnostic-60283765` adopted native Stage 4 Practice at frame 207. The original game showed **STAGE CLEAR** after 4,243 decisions, at the last logged stage frame of 17,175. The controller used four bombs and lost zero lives during the recorded section. The end screen showed 40,815,110 points including a 12,000,000 clear bonus. The controller exited normally. The previous story-run failure did not recur in this practice configuration, and a fresh one-credit story trial is required to verify progression to Stage 5.

The original practice report used the generic `native_end` outcome. The screenshot and a read-only memory check verified the clear. The evaluator now supports an explicit Practice Start mode with `practice_clear` as a distinct result. A passive observer collected 2,084 candidate training rows from the practice run. Only 73.3% of consecutive moving samples passed the 0.5-unit kinematic check, so this capture is retained as diagnostic evidence and excluded from expert SFT.
