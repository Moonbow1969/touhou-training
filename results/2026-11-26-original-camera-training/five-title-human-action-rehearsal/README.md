# Five-title human action training at fixed compute

Starting from the same four-title human checkpoint, we compared a TH08-only continuation with a shared TH08, TH13, TH14, TH16 and TH17 continuation. Both consumed 5,587,200 training decisions in four passes of the same size. The mixed run saw 3,921,280 distinct decisions. It allocated half the budget to TH08 and divided the other half across the eight historical game and rank cells. A frozen four-title checkpoint supplied a third reference. Inputs comprised human key history and replay metadata. [Sampling protocol and exact hashes](protocol.json) · [Source roster](sources.json) · [Optimizer log](training.json) · [Mixed weights](epoch-4.pt)

| Held game and difficulty | Players | Scored decisions | Frozen | TH08-only | Mixed | Frozen on turns | TH08-only on turns | Mixed on turns |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TH08-Hard | 8 | 248,192 | 77.25% | 77.36% | 77.31% | 8.48% | 9.31% | 9.76% |
| TH08-Lunatic | 8 | 236,544 | 76.85% | 76.94% | 76.93% | 13.18% | 11.91% | 13.06% |
| TH13-Hard | 4 | 102,336 | 72.59% | 69.41% | 72.46% | 10.19% | 15.71% | 9.38% |
| TH13-Lunatic | 2 | 45,248 | 73.78% | 71.29% | 73.57% | 13.52% | 15.32% | 11.96% |
| TH14-Hard | 2 | 47,104 | 76.48% | 73.00% | 76.35% | 10.07% | 17.73% | 10.44% |
| TH14-Lunatic | 3 | 71,808 | 70.30% | 66.85% | 70.21% | 11.70% | 16.30% | 11.53% |
| TH16-Lunatic | 6 | 153,280 | 69.66% | 66.95% | 69.65% | 13.28% | 15.33% | 13.20% |
| TH17-Hard | 1 | 47,488 | 74.25% | 70.99% | 74.60% | 12.75% | 21.15% | 13.89% |
| TH17-Lunatic | 3 | 69,568 | 71.69% | 68.84% | 71.71% | 12.67% | 16.73% | 12.87% |

TH16 Hard had no historical test replay from a credited player absent from both training sources. The retention set is small in several cells. TH17 Hard contains two replays by one player. Each TH08 held author was absent from both continuation training rosters. Two of the 16 TH08 held names had appeared in the historical parent. The main table gives each author equal weight within a cell. [Author rows and paired differences](report.json) · [Figure](held-policy-retention.png)

On the 536,832 historical test decisions, pilots changed direction 29.03% of the time. The TH08-only model predicted a change on 13.58%, with 51.53% turn-detection precision and 24.10% recall. Mixing the original games restored next-direction accuracy on all seven historical cells while bringing predicted-change frequency down to 6.47%, precision to 63.77% and recall to 14.22%. Those figures pool decisions. This tradeoff explains why raising accuracy on turns alone can lower overall accuracy. The models have no camera input in this experiment. [Verified turn alarms and exact-direction counts](turn-detection.json)
