# TH13 Easy training on two boss phases

An opening guide brought the original game to a registered spell and inventory. At handoff the screen-only policy received full control with zero spare lives and one Bomb. Each eligible trajectory continued to the original terminal. The allocation reserved 16 opening attempts per phase.

| Handoff phase | Attempts | Eligible | Policy decisions | Decisions sampled for PPO | Stage 2 after handoff |
|---|---:|---:|---:|---:|---:|
| Earlier spell | 16 | 11 | 1,535 | 1,535 | 0/11 |
| Final spell | 16 | 4 | 1,567 | 1,535 | 1/4 |

All 3,102 decisions contributed to computing trajectory advantages. The optimizer used 3,070 distinct decisions in four accepted epochs. The single Stage 2 result belongs to the policy that collected the training data after a guided opening. Evaluation of the fitted child starts from the ordinary Easy opening with full stock and no gameplay guide.

On 16 ordinary full-stock Easy pairs, the previous final-spell PPO and its two-phase child each reached Stage 2 in 0/16 with zero ending spare lives. Mean final Bombs were 0.250 and 0.3125, and each policy cast 80 times. The native Stage 1 clock advanced from 5,896.625 to 6,213.625 ticks on average. The paired difference was +317 ticks with a 95% interval of −821 to +1,452. [Full ordinary test data](th13-held-evaluation/paired-by-rank.json) · [Complete first matched pair video](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th13-multiphase-curriculum-first-ordinary-easy-pair.mp4) · [Original test traces](th13-held-native-logs.release.json)

[Acquisition and outcomes](acquisition-by-phase.json) · [Per-phase optimizer sampling](phase-balanced-optimizer.json) · [Optimizer report](policy/report.json) · [Training log](policy/train.jsonl) · [Fitted actor](policy/policy.ts) · [Recorded Stage 2 handoff](first-confirmed-stage2-suffix/README.md) · [Earlier-phase original records](th13-previous-native-suffixes.release.json) · [Final-phase original records](th13-final-native-suffixes.release.json)
