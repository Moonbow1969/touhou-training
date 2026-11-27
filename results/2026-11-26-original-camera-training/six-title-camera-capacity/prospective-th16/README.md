# Four independent TH16 players on Hard and Lunatic

Two Hard and two Lunatic pilots were reserved before fitting. All four original-game captures qualified. None of the pilots appeared in the training sets. Overall movement accuracy is the fraction of four-tick windows in which the highest-probability direction matched the human command.

| Policy | Correct directions / 10,729 | Correct starts / 925 | Correct turns / 1,307 |
|---|---:|---:|---:|
| Five-title CNN, no TH16 training | 5,187 (48.35%) | 38 | 149 |
| Six-title CNN including TH16 | 5,373 (50.08%) | 64 | 221 |
| Six-title spatial attention including TH16 | 5,411 (50.43%) | 50 | 215 |

The six-title CNN gained 186 correct directions across the four pilots. The paired gains were +5, −86, +226 and +41 windows. Both six-title models improved on three of four pilots. The CNN and attention model shared their six-title input schedule. These measurements evaluate human-action prediction on held screens. Campaign success requires separate original-game evaluation.

[Per-player and per-action scores](held.json) · [Training membership and reserved-player hashes](../prospective-th16-coverage.json) · [Source archives and release digest](../../th16-visual-expansion/independent-held/held-github-release.json)
