# Training movement with the preceding sampled direction

Both policies use the same human camera encoder and train a direction head on four consecutive decisions from the 68-stage split. The teacher model sees the preceding expert command in each supervised loss. The propagated model begins with the same expert command and then integrates over its own predicted commands for the remaining three steps. The deployed models receive their own sampled previous direction. Focus, shooting and Bombs keep the same learned weights.

| Training | Adjacent changes on held TH17 screens | Direct reversals | Start NLL | Turn NLL |
|---|---:|---:|---:|---:|
| teacher | 40.47% | 1.06% | 2.493 | 2.607 |
| propagated | 46.91% | 1.84% | 2.598 | 2.601 |

These four TH17 pilots previously informed the experiment design. The table uses fixed human screens and sums over all nine possible preceding sampled commands. A separate six-player roster was frozen before fitting. The original TH10 and TH17 evaluation uses new controller seeds allocated before any of those games ran.

[Training registration](protocol.json) · [Teacher fit](teacher/fit.json) · [Propagated fit](propagated/fit.json) · [Original-game assignment](../four-step-feedback-original/assigned-seeds.json)
