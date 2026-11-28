# TH08 advantage-weighted replay and human imitation

Two policy training arms mix 64 human and 64 original policy decisions per update. The uniform arm treats the original decisions equally. The AWR arm weights them with clipped exponentiated, out-of-fold GAE advantages. Both start from the same deployed TH08 PPO policy, use identical minibatches for matched seeds and update the same four camera-based action heads. The image encoder is frozen.

The corpus contains 72 verified original credits and 66,428 decisions. Sixty-three credits and 57,753 decisions enter the fit. Nine complete credits are held out of the current fit for a behavior-divergence check. Five independently fit value readouts supply the advantages on disjoint original credits. The eligible human population contributes 58,696 training observations from five players and 15,558 held observations from two other players. Prior experiments have evaluated those two authors. The final exact clipped weights use β=0.149683 and an effective sample fraction of 30.035%.

Each of the six final policy fits used 2,000 updates. The lowest registered seed, 808401, determines the play-test pair before any policy results. These are offline checks.

| Fixed test seed | Human movement accuracy | Human switch accuracy | Mean original behavior KL |
|---|---:|---:|---:|
| Parent | 49.467% | 23.462% | — |
| Uniform | 49.666% | 22.977% | 0.00738 |
| Advantage-weighted | 49.556% | 23.007% | 0.01032 |

Both fixed-seed arms passed the preregistered guard. Human turn accuracy fell by 0.485 percentage points for uniform and 0.455 points for AWR. The maximum allowed decrease was 0.5 points. The offline guard allows the original-game test. It does not certify a gameplay gain.

[Protocol](protocol.json) · [Fit inputs](fit-inputs.json) · [All seeds and optimizer traces](report.json)
