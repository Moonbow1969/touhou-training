# Online PPO with a learned previous-direction input

A human SFT first learned four-title visual movement from replay cameras. This online update restored its entire visual policy and its learned previous-direction movement head. The critic began at zero. Both original games supplied on-policy actions to the same network, with all four action heads and the visual encoder trainable.

| Game | Hard training decisions | Lunatic training decisions |
|---|---:|---:|
| TH10 | 2,089 | 1,576 |
| TH17 | 6,316 | 5,156 |

Each cell contributed eight ordinary full credits. All four PPO epochs were accepted on 15,137 decisions from 32 new games. Replaying the native observations, previous directions and chosen actions gave a maximum absolute log probability discrepancy of 0.000011. The largest final joint KL was 0.00468. Each control seed in the registered evaluation receives a full SFT credit and a fresh full PPO credit on the same original, difficulty and lane. [Optimizer report](policy/report.json) · [Deployable actor](policy/policy.ts) · [Source archive](release-archive.json) · [Training and held seeds](camera-protocol.json)
