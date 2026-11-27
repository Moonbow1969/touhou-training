# Which Touhou games have trained a policy?

The human columns count distinct synchronized four-tick screen/action windows in the six-title visual SFT. The PPO columns count unique original Hard and Lunatic training games included in completed updates across the main experimental series. A single six-title PPO update also used all six online titles together. Reusing a game for another optimizer or epoch adds zero to these PPO counts.

| Game | Human Hard windows | Human Lunatic windows | PPO Hard games | PPO Hard decisions | PPO Lunatic games | PPO Lunatic decisions |
|---|---:|---:|---:|---:|---:|---:|---:|
| TH08 | 0 | 0 | 44 | 30,971 | 44 | 27,362 |
| TH10 | 27,478 | 19,087 | 116 | 28,444 | 116 | 22,705 |
| TH11 | 14,622 | 12,455 | 68 | 15,725 | 68 | 14,991 |
| TH12 | 30,553 | 34,473 | 88 | 47,731 | 88 | 39,604 |
| TH13 | 31,246 | 30,845 | 88 | 66,072 | 88 | 49,342 |
| TH16 | 2,163 | 2,951 | 0 | 0 | 0 | 0 |
| TH17 | 43,173 | 40,290 | 96 | 78,460 | 96 | 61,266 |
| **Total** | **149,235** | **140,101** | **500** | **267,403** | **500** | **215,270** |

TH08 also has separate action-history imitation research. TH16 season-release inputs are preserved in its human data. The six-title visual SFT predicts movement, focus, shooting and Bomb. The ledger counts a five-title SFT-to-PPO training game once the optimizer has consumed it.

[Six-title SFT source and weights](six-title-human-camera-th16/README.md) · [PPO input ledger](learning-curves/hard-lunatic-training-coverage.json) · [192 independent original full-stock tests](six-game-return-horizon/all-six-original-results.md) · [Five-title human SFT to online PPO](reviewed-five-human-online/README.md)
