# TH12 Stage 1 dialogue and survival

On Normal Reimu A in the original TH12 0.02a trial, the native HUD stays visible during the Nazrin conversation. The benchmark times a complete credit through the original terminal menu. A policy can spend longer in dialogue when it produces fewer distinct shoot-key presses. Two credits entered this conversation and subsequently returned to combat. The following durations come from the native screens immediately before, during and after the conversation. [Original boundary frames](dialogue-boundaries.jpg)

| Frozen credit | Whole-credit seconds | Dialogue duration bounds | Survival reward attributed to dialogue | Fraction of survival term |
|---|---:|---:|---:|---:|
| Entropy TRAIN 931204 | 189.33 | 55.85–58.87 s | 0.05312 | 30.4% |
| Steam Deck held 895206 | 185.58 | 33.29–36.31 s | 0.03253 | 19.0% |

The optional offline overlay recognizes the left portrait and dialogue box in the original 112 × 96 grayscale camera. Its 24 gradient prototypes came from the single Entropy TRAIN sequence. The threshold of 0.5611 was fixed using that sequence and 2,800 training negative frames from 35 other credits. Two adjacent positive decisions confirm dialogue. Intermediate scores retain the original reward and carry an uncertainty flag. The overlay removes only the survival component during confirmed dialogue. Each output includes the frozen model hash and the source trajectory hash. [Scoring implementation](../../../../tools/rollout-worker-probe/th12-full/dialogue.py) · [Frozen prototypes](dialogue-prototypes.npz)

The independent Steam Deck positive sequence contained 249 decisions wholly inside the native dialogue interval. The overlay detected 248. It marked zero of 1,117 decisions wholly outside it. Another 24 decisions fell in the uncertain visual transition intervals. Among 97 separate, short credits with 47,775 decisions, the overlay produced zero dialogue flags and retained 13 uncertain decisions. Inspection of the 24 highest-scoring negative frames showed active shooting, scenery or bullet effects. Across 141 screened credits outside negative calibration, only the two pictured credits produced confirmed flags. These checks cover one encounter, one player character and two host configurations. Detection of dialogue in later stages, other games or different shot types is untested. [Validation ledger](validation.json) · [High-scoring controls](high-score-negative-controls.jpg)

All published primary returns and trajectories use the original reward. A future paired experiment can write a separately versioned overlay with `python dialogue.py SOURCE.npz dialogue-prototypes.npz NEW-OVERLAY.npz`. The program refuses to overwrite an existing file.
