# Effect of including TH11 on each game

Both camera policies trained for 192,000 sampled windows. Each row compares predictions on the same original human screens. Direction and turn accuracy pool decisions within a game. The movement negative log likelihood weights each player equally within that game. Lower loss is better.

| Game | Players | Stages | Correct movement, four → five | Correct turns, four → five | Movement loss, four → five |
|---|---:|---:|---:|---:|---:|
| TH10 | 2 | 3 | 4,322 → 4,246 / 6,600 | 150 → 153 / 600 | 1.043 → 1.071 |
| TH11 | 1 | 1 | 1,230 → 1,271 / 2,446 | 42 → 37 / 282 | 1.566 → 1.552 |
| TH12 | 2 | 3 | 5,291 → 5,058 / 9,806 | 225 → 236 / 1,338 | 1.388 → 1.422 |
| TH13 | 2 | 4 | 8,048 → 7,919 / 11,835 | 293 → 320 / 1,189 | 0.997 → 1.017 |
| TH17 | 4 | 8 | 11,977 → 11,740 / 24,643 | 378 → 386 / 3,171 | 1.536 → 1.555 |

TH11 contributes a single development player. Several players contributed multiple stages. The four earlier games have appeared in development comparisons throughout the project. Adding TH11 also redistributes the fixed sampling budget across games and difficulties. These observations describe expert action prediction. Ordinary campaign progress is measured separately.

[Player-level results](per-game-transfer.json) · [Matched checkpoints and source allocations](../README.md)
