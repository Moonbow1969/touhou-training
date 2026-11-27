# How much does half a second of visual history help human imitation?

The same frozen SFT encoder and action logits supplied two identical recurrent heads. One head received eight consecutive original observations at four-native-tick intervals. Its control received the current embedding eight times. Both used 6,000 updates, 768,000 sampled presentations and the same 69,694 eligible human training windows across TH10, TH12, TH13 and TH17. The model had access to visual history alone. The four controls remained direction, focus, shooting and casting. [Registered design and stage fingerprints](protocol.json)

Eight held players contributed 33,083 observations, including 9,578 direction changes. Chronological history raised the fraction of correctly predicted direction changes in all four games. It also raised change-of-direction negative log likelihood in all four, so the primary test failed. Overall accuracy, averaged equally over titles, rose by 0.52 percentage points against the equal-capacity control. Both fitted residual heads had lower overall accuracy than the frozen SFT parent in every title. The candidate did not advance to the registered original-game test. [Per-stage and per-title results](report.json) · [Direction-change figure](held-changes.png)

| Game | Change accuracy, current → history | Change NLL, current → history |
| --- | ---: | ---: |
| TH10 | 23.86% → 24.77% | 2.252 → 2.276 |
| TH12 | 21.29% → 22.91% | 2.390 → 2.415 |
| TH13 | 25.51% → 26.39% | 2.332 → 2.354 |
| TH17 | 22.90% → 23.89% | 2.471 → 2.567 |
