# Increasing the number of human players at fixed training compute

Both visual policies used the same RTX 5090, 44,149 optimizer updates and 2,825,536 sample presentations. The smaller roster comprised 77 credited training players, the larger 97. The actors saw 233,748 and 286,316 distinct windows respectively. Each was tested on the same 12 controller seeds per game and difficulty, yielding 96 independent physical credits across the four cells. Uncontrolled original-game RNG can differ within a pair. [Bound checkpoints and training reports](checkpoint-binding.json) · [Per-seed evaluation](evaluation/credits.json)

| Game and difficulty | Mean original progress, 77 to 97 players | Paired change, 95% interval | Stage 2 per arm | Terminal resources, 77 to 97 |
|---|---:|---:|---:|---|
| TH10 Hard | 2,059 to 1,706 | −353 [−764, +36] | 0/12 each | Power 0.050 to 0.238 |
| TH10 Lunatic | 1,707 to 1,329 | −377 [−628, −152] | 0/12 each | Power 0.350 to 0.213 |
| TH12 Hard | 2,713 to 2,499 | −214 [−897, +475] | 0/12 each | Bomb 0.917 to 0.750 |
| TH12 Lunatic | 2,019 to 1,956 | −64 [−252, +119] | 0/12 each | Bomb 1.333 to 1.250 |

All 96 credits ended with zero spare lives. Larger human coverage at this fixed optimization budget did not improve deployment in these two titles. TH10 Lunatic reached the final death earlier. TH12 included all 38 warm starts and 10 cold fallbacks, with reset mode and host recorded per physical credit. [TH10 original ticks and cast counts](native10/ranked-native-comparison.json) · [TH12 native events](th12-held-native-event-pairs.json) · [TH12 reset balance](th12-held-reset-balance.json) · [Full Hard games](evaluation/hard-full-credits.mp4) · [Full Lunatic games](evaluation/lunatic-full-credits.mp4)
