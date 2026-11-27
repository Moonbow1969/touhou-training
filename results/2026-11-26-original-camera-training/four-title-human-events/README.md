# Predicting movement starts, turns and Bomb onsets from the game camera

The Bomb onset readout fitted 331 positive windows among 231,511 human decisions. Movement, focus and firing retain their original SFT weights. The direction experiment gives starts and turns twice the weight in the supervised movement objective.

## Repeated TH17 development comparison

| Bomb readout | Observed onsets | Correct among top 5% per stage | False alerts among top 5% | Probability ≥0.5, hits / all alerts | Mean average precision across players with onsets |
|---|---:|---:|---:|---:|---:|
| Original 68-stage SFT | 19 | 1 | 1235 | 0 / 0 | 0.0055 |
| Case-weighted Bomb head | 19 | 1 | 1235 | 0 / 0 | 0.0026 |

All 19 onsets came from two Lunatic players. The other two players contributed their Hard stages to the false-alert count. Each contributing player receives equal weight after averaging AP over their stages with at least one onset. All four players previously informed follow-up design. Six new TH10, TH12 and TH13 players were allocated before this fit for prospective validation.

[Bomb checkpoint](bomb/policy.ts) · [Training data and fit](bomb/report.json) · [Complete repeated TH17 scores](bomb/repeated-th17/onsets.json) · [Bomb protocol](bomb-protocol.json) · [Direction protocol](movement-protocol.json) · [Six-player registration](../unseen-later-stage-author-challenge/prospective-six-players-frozen.json)
