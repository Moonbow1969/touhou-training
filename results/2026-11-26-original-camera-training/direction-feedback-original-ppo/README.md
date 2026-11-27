# Online PPO with a learned previous-direction input

A human SFT first learned four-title visual movement from replay cameras. This online update restored its entire visual policy and its learned previous-direction movement head. The critic began at zero. Both original games supplied on-policy actions to the same network, with all four action heads and the visual encoder trainable.

| Game | Hard training decisions | Lunatic training decisions |
|---|---:|---:|
| TH10 | 2,089 | 1,576 |
| TH17 | 6,316 | 5,156 |

Each cell contributed eight ordinary full credits. All four PPO epochs were accepted on 15,137 decisions from 32 new games. Replaying the native observations, previous directions and chosen actions gave a maximum absolute log probability discrepancy of 0.000011. The largest final joint KL was 0.00468. Each control seed in the registered evaluation receives a full SFT credit and a fresh full PPO credit on the same original, difficulty and lane. [Optimizer report](policy/report.json) · [Deployable actor](policy/policy.ts) · [Source archive](release-archive.json) · [Training and held seeds](camera-protocol.json)

Across 32 fresh paired starts, the human SFT and its PPO child both reached Stage 2 in 0/32 and finished with zero spare lives. On TH10 Hard the PPO child advanced 536 fewer native ticks on average (paired 95% interval −1,060 to −23). On TH17 Lunatic it advanced 1,121 more ticks (+242 to +2,000), doubled actual Bomb casts from 21 to 41, and ended with 1.25 Bombs versus 2.125 for SFT. TH10 Lunatic (+246) and TH17 Hard (+65) had intervals spanning zero. Each cell included eight matched pairs. [Per-credit outcomes and resources](evaluation/README.md) · [Hard footage](evaluation/first-hard-full-credits.mp4) · [Lunatic footage](evaluation/first-lunatic-full-credits.mp4)
