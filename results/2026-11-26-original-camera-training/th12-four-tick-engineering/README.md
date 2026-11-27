# TH12 human replay and original controller timing

The exact-build human training set contains 25,392 four-tick decisions across nine stages and three pilots. The six held stages from separate pilots supply 18,735 decisions. The model sees a 96×112 grayscale frame and its absolute difference from the frame four original ticks earlier. Movement, focus and fire use the next game's input word. Bomb includes taps anywhere in the following four ticks.

| Physical host | G2 held credits | Wall-driven original ticks/decision, median | Exact clock decisions/s | Genuine ticks/s | All adjacent decisions four ticks apart |
|---|---:|---:|---:|---:|---:|
| deck | 24 | 4.84 | 8.54 | 34.62 | 406/406 |
| entropy | 24 | 6.53 | 10.85 | 43.85 | 609/609 |

The historical figures summarize whole-credit averages from a sparsely polled native clock. The prospective engineering worker waits for the next multiple of four, verifies the stage and tick before and after capture, and records the game's consumed direction, Shift, Z and X bits. Both independent Hard credits reached the original Game Over. Every end-of-hold mask matched, including 342 decisions with focus on. Across both credits the independent sampled-action log-probability check differed by at most 0.00000370.

The longest inter-decision wall intervals were 164 ms on Steam Deck and 398 ms on Entropy. The native clock recorded zero skipped or duplicated decision slots. Each host ran one engineering credit, which qualifies the timing and input path. A comparison of game performance under the two clocks would require a separately registered matched-seed evaluation.

## Held human temporal ablation

On the same 18,717 held expert decisions, the current image, next controls and frozen weights were kept identical. Only the source frame for the difference plane changed. Both held pilots had their lowest movement NLL at the trained four-tick gap.

| Prior image offset in original ticks | Movement NLL | Movement accuracy |
|---:|---:|---:|
| 0 | 2.310 | 41.44% |
| 4 | 1.386 | 54.45% |
| 8 | 1.536 | 49.20% |
| 12 | 1.654 | 45.63% |

A zero offset supplies a zero difference plane. These held archives contain frames sampled at four-tick intervals. The test measures 4, 8 and 12 ticks directly and supplies no exact 6-tick counterfactual.
