# Learning movement from the previous executed direction

Both camera policies started from the same human SFT checkpoint and used 70,078 consecutive four-tick decisions from 22 training players across TH10, TH12, TH13 and TH17. Each residual direction head trained for 4,000 minibatches of 128 examples, sampled equally across Hard and Lunatic in all four games. The image control uses a constant neutral direction. The feedback model receives the last direction the agent issued. Both retain the original focus, shooting and Bomb heads. [Training registration](protocol.json) · [Optimizer and player-level records](report.json)

Eight development players contributed 33,239 decisions, including 9,601 direction changes. The table gives movement accuracy at those changes. The predictions use the model's own probability distribution over previous directions, integrated exactly along each recorded human camera sequence. This measures the action predictor on the human's route. Survival requires an original game in which the agent's actions alter subsequent images.

| Held game | Image control | Previous direction | Change NLL, image → feedback |
|---|---:|---:|---:|
| TH10 | 24.97% | 25.94% | 2.271 → 2.121 |
| TH12 | 20.95% | 23.76% | 2.403 → 2.232 |
| TH13 | 25.84% | 26.15% | 2.302 → 2.150 |
| TH17 | 22.87% | 25.68% | 2.385 → 2.214 |

Conditioning on the actual human's previous direction improved mean all-decision log loss by 0.337 nats per title against the matched image control. Its top-choice accuracy on genuine changes under that conditioning ranged from 11.11% to 15.24%. The development result passed the registered threshold for original-game tests. The independent TH10 and TH17 trials use the direction actually sampled by the running agent. [Development figure](held-direction-changes.png) · [Two-title original-game registration](../four-title-direction-feedback-native/camera-protocol.json) · [Deployable image control](image/policy.ts) · [Deployable feedback policy](feedback/policy.ts)

A post-hoc control fit a nine-by-nine direction transition table to the same training players. Without images, its most likely direction on the held sequences missed every movement start and every turn between two moving directions, yet its aggregate accuracy on direction changes exceeded the feedback model on all four titles. It benefited from changes ending at neutral. The camera model correctly predicted 5–8% of starts and 14–23% of moving turns across titles. Moving its inputs halfway around each held stage worsened change NLL on all four. The aligned images contribute information about starting and steering. Predicting the expert's precise response remains difficult. [Training-only action baseline, shifted images and per-title subsets](image-alignment/report.json)
