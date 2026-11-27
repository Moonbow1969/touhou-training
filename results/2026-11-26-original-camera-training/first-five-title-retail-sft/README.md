# Adding the fifth original game to human camera SFT

Both policies start from the same visual encoder and train all four action heads using 3,000 updates of 64 windows. The four-game control uses 74 synchronized stages from TH10, TH12, TH13 and TH17. The five-game actor adds TH11 Hard Marisa B and Lunatic Reimu B from two distinct training players. The 4,489 new four-tick windows have zero human Bomb onsets. The fixed sampling budget is redistributed over ten game/difficulty groups in the five-game fit.

| Human camera policy | Stages | Available windows | TH11 Hard draws | TH11 Lunatic draws | Held TH11 direction | Starts | Turns | Direction NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| four titles | 74 | 257,145 | 0 | 0 | 1,230/2,446 | 16/220 | 42/282 | 1.566 |
| five titles | 76 | 261,634 | 18,888 | 19,040 | 1,271/2,446 | 22/220 | 37/282 | 1.552 |

The TH11 development population is a single other player, Dagoth2hu, on Lunatic Marisa C. These values describe imitation on that player’s original screens. A larger distinct player roster provides the next check.

The [two TH11 training stages](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/first-five-title-retail-sft-th11-train-original-inputs.zip) and [one TH11 development stage](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/first-five-title-retail-sft-th11-development-original-inputs.zip) each include the original screen/action arrays and synchronization evidence. Extract them beside the [four-title camera inputs](../four-title-complete-epochs/training-data.md) to reconstruct the 76 training stages and 19 development stages used here. The [training receipt](th11-train-original-inputs.release.json) and [development receipt](th11-development-original-inputs.release.json) record the remote checksums.

[Original two training stages and stock/score evidence](added-th11.json) · [Study registration](protocol.json) · [All 19 development stages](development/held.json) · [Four-title weights](four/policy.ts) · [Five-title weights](five/policy.ts) · [Five-title training log](five/loss.jsonl)
