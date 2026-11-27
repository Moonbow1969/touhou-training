# Held movement with the previous human input available

On a recorded human demonstration, predicting the previous expert direction is correct on every unchanged direction and wrong on every change. The five-title camera policy receives the playfield image and its recent difference. This comparison separates correctly predicted human changes from choices that interrupt a human hold. The persistence column uses the recorded previous expert action. It measures temporal predictability of the demonstration and does not estimate autonomous survival.

| Game | Windows | Correct by repeating previous expert direction | Correct by screen policy | Expert changes correctly predicted | Policy errors on unchanged direction |
|---|---:|---:|---:|---:|---:|
| TH10 | 6,600 | 76.6% | 66.0% | 411/1,543 | 1,110/5,057 |
| TH11 | 2,446 | 70.4% | 54.3% | 210/723 | 606/1,723 |
| TH12 | 9,806 | 66.6% | 53.0% | 774/3,275 | 2,108/6,531 |
| TH13 | 11,835 | 75.8% | 67.6% | 781/2,863 | 1,754/8,972 |
| TH17 | 24,643 | 67.1% | 48.2% | 2,053/8,118 | 6,704/16,525 |

[Per-game counts and checkpoint identity](expert-direction-persistence.json) · [Original held development results](held.json)
