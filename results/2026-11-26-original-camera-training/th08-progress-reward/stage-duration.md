# TH08 Normal Ghost stage durations

The five TRAIN players all reached both Stage 2 and Stage 3 in their original 0.02 replays. Their last sampled stage-local clocks yield **15,652 ticks for Stage 1** and **18,624 ticks for Stage 2** as conservative candidate normalization constants. Each median uses one completed stage per player. The actual transition can occur after the last eligible camera sample.

| TRAIN replay author | Stage 1 last observed tick | Stage 2 last observed tick | Stage 3 last observed tick |
|---|---:|---:|---:|
| PAWA | 11,644 | 15,312 | 15,404 |
| TAK. | 16,556 | 16,420 | 16,744 |
| KATiWO | 15,652 | 18,624 | 16,072 |
| WAI NYMO | 17,052 | 25,424 | 21,024 |
| Yoshi | 13,548 | 19,052 | 20,208 |

The observer samples every fourth eligible tick, validates current and previous action bits against the original replay, and checks that the input cursor equals its original per-stage offset plus twice the tick. The original Stage 2 and Stage 3 cameras establish ordinal progression for the preceding stages. Across these fifteen traces the dummy-inclusive source word count exceeds the last sampled tick by 4–9. Holding a key through the tail and releasing it before playback ends both occur. The source length, the last nonzero key and host time cannot supply an exact transition clock.

**Stage 3 has no registered normalization constant.** All five original USER records say “Stage 3”. All five post-playback images show the replay menu. The last cameras show end-of-boss presentation, including failed-spell text on some. Finishing a spell and returning from replay playback do not establish that the trial's objective ending was reached. The median last sampled tick of 16,744 is descriptive. A future three-stage reward needs independently confirmed trial-ending footage or an original clear indicator tied to native ticks.

Source URLs, original executable and replay hashes, raw camera hashes, per-stage observer receipts and following-stage witness ticks are in [the machine-readable candidate](stage-duration-candidate.json). The two held Normal authors were excluded from all duration calculations. The constants have not been installed in any learner. These clocks come from human replay mode and provide no substitute for validating native tick increments during live policy play.
