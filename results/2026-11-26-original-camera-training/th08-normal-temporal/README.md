# TH08 Normal Ghost causal camera history

Both fits start from the same frozen six-title replay SFT. The same 58,696 eligible original Normal camera windows span 15 stages from five Ghost Team players. Each arm uses identical sampled indices, 1,200 updates, 153,600 presentations and the same residual readout capacity and initial weights. The treatment sees four strictly ordered camera states and the control receives only the current state. Both update movement, shooting and Bomb. Both use the parent model's learned Focus head because the Normal competition replays have zero eligible positive Focus labels.

| Held Normal pilot | Current-only movement NLL | Four states movement NLL | Current-only movement accuracy | Four states movement accuracy |
|---|---:|---:|---:|---:|
| HOS.YN | 2.0167 | 2.0081 | 31.79% | 32.29% |
| Rota | 0.9801 | 0.9687 | 68.44% | 68.86% |

The 15,558 evaluation windows come from two players whose data were held out of this fit and used in earlier development comparisons. Four states modestly reduce overall NLL on both players. NLL on direction changes rises from 2.8860 to 2.8934 for HOS.YN and falls from 2.2866 to 2.2553 for Rota. A new, separately registered original-game comparison supplies the closed-loop evidence.

[Registered protocol](protocol.json) · [Per-stage metrics and checkpoint hashes](report.json) · [Learner log](loss.jsonl) · [Current-only policy](change-free/policy.ts) · [Four-state policy](temporal/policy.ts)
