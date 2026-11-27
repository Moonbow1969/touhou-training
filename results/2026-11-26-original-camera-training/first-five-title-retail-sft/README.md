# Adding the fifth original game to human camera SFT

Both policies start from the same visual encoder and train all four action heads using 3,000 updates of 64 windows. The four-game control uses 74 synchronized stages from TH10, TH12, TH13 and TH17. The five-game actor adds TH11 Hard Marisa B and Lunatic Reimu B from two distinct training players. The 4,489 new four-tick windows have zero human Bomb onsets. The fixed sampling budget is redistributed over ten game/difficulty groups in the five-game fit.

| Human camera policy | Stages | Available windows | TH11 Hard draws | TH11 Lunatic draws | Held TH11 direction | Starts | Turns | Direction NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| four titles | 74 | 257,145 | 0 | 0 | 1,230/2,446 | 16/220 | 42/282 | 1.566 |
| five titles | 76 | 261,634 | 18,888 | 19,040 | 1,271/2,446 | 22/220 | 37/282 | 1.552 |

The TH11 development population is a single other player, Dagoth2hu, on Lunatic Marisa C. These values describe imitation on that player’s original screens. A larger distinct player roster provides the next check.

The prospective TH11 check qualified three of four additional players. Across their 8,425 original windows, movement accuracy rose from 4,131/8,425 (49.0%) with four-title training to 4,450/8,425 (52.8%) with all ten added TH11 stages. The largest gain came from one Lunatic player. On the same three players, correct movement starts fell from 48/717 to 34/717. [Full independent comparison and excluded capture](reviewed-independent-th11/README.md)

Two further five-title policies used the same initialization, 192,000 total sampled windows and the same 37,928 sampled TH11 windows as the first five-title actor. The larger source sets include real Bomb onsets. All four other games retain their original training stages.

| TH11 training stages | TH11 players | TH11 distinct windows | TH11 Bomb onsets | All five-title training windows | Held directions | Held starts | Held turns |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2 | 4,489 | 0 | 261,634 | 1,271/2,446 | 22/220 | 37/282 |
| 9 | 6 | 24,440 | 20 | 281,585 | 1,310/2,446 | 12/220 | 45/282 |
| 10 | 7 | 27,077 | 21 | 284,222 | 1,327/2,446 | 10/220 | 43/282 |

The tenth source matches the original inventory and score on one pre-reset clear tick. This criterion was applied across the entire roster. Its inclusion raised overall direction matches on the same development player and left movement starts at 10/220. [Nine-stage TH11 source additions](added-complete-th11.json) · [Ten-stage reviewed evidence](added-reviewed-th11.json) · [Nine-stage actor](five-complete/policy.ts) · [Ten-stage actor](five-reviewed/policy.ts) · [All four models on the same development players](reviewed-development/held.json) · [Seven additional trace archives](th11-seven-more-original-training-stages.release.json) · [Tenth trace archive receipt](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th11-retail-pre-reset-increment-one-train-camera-receipt.json)

Across the four previously trained works, overall held movement accuracy fell in every game after the TH11 addition. Correct turn counts rose in all four. The new TH11 player's overall movement accuracy rose and turn accuracy fell. The smaller expected number of draws per prior game is one possible explanation. [Per-game and per-player results](development/per-game-transfer.md) · [Matched title count and sampling budget experiment](../five-title-sampling-budget/protocol.json)

The 84-stage screen policy correctly anticipated 210 of the TH11 developer's 723 movement changes. It also changed direction on 606 of the 1,723 windows where that player continued the prior movement. All five games show more errors on held directions than correct predictions of changes. [Temporal predictability and movement errors by game](reviewed-development/expert-direction-persistence.md)

The [two TH11 training stages](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/first-five-title-retail-sft-th11-train-original-inputs.zip) and [one TH11 development stage](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/first-five-title-retail-sft-th11-development-original-inputs.zip) each include the original screen/action arrays and synchronization evidence. Extract both with the [three loader metadata files](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/first-five-title-retail-sft-th11-loader-metadata.zip). The [four-title inputs](../four-title-complete-epochs/training-data.md) supply the 74 common training stages and 18 common development stages. The three releases have [training](th11-train-original-inputs.release.json), [development](th11-development-original-inputs.release.json) and [metadata](th11-loader-metadata.release.json) checksums.

[Original two training stages and stock/score evidence](added-th11.json) · [Study registration](protocol.json) · [All 19 development stages](development/held.json) · [Four-title weights](four/policy.ts) · [Five-title weights](five/policy.ts) · [Five-title training log](five/loss.jsonl)
