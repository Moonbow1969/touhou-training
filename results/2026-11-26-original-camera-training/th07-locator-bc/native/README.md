# TH07 official 0.11 Normal full-credit test

Six valid independent controller seeds compared action-only SFT with the auxiliary player-XY SFT. Every pair played Normal Reimu A with the ordinary two spares and three Bombs. Both arms sampled their four action heads without assistance. First stage reached its original Game Over in all 12 valid credits. Each scorecard showed zero continues and three misses.

| Native full-credit outcome | Action SFT | Action SFT + XY |
|---|---:|---:|
| Reached Stage 2 | 0/6 | 0/6 |
| First-stage deaths / native ticks | 18 / 33,590 | 18 / 37,387 |
| Deaths per 10,000 native first-stage ticks | 5.359 | 4.815 |
| Mean first-stage native ticks | 5,598.3 | 6,231.2 |
| Mean progress on the official result screen | 4.638% | 5.240% |
| Actual Bombs on official result screens | 27 | 30 |
| Deaths with Bomb stock visible on the preceding HUD | 12/18 | 14/18 |
| Final displayed spare lives, all credits | 0 | 0 |
| Mean final displayed Bomb stock | 1.67 | 0.67 |
| Mean original score | 420,928 | 510,168 |

The registered death-rate ratio was **0.8984**. Promotion required a ratio of 0.75 or lower and first-stage progress at least as good. Both arms exceeded 30,000 first-stage ticks at the six-valid-pair minimum. Controller sampling seeds and physical lanes matched. Original in-game randomness was uncontrolled. The difference in the two rates is descriptive.

Fourteen original credits ran for seven seed pairs. The action-only credit for seed 2130755 missed 43 scheduled decision boundaries. Its 95.13% exact-window rate fell below the registered 99.5%. The bundle retains both original credits from that pair. The first replacement seed, 2130771, supplied a full valid pair. All twelve scored credits reached genuine one-credit Game Over with no entirely stale complete action window. Minimum full-window accuracy was 99.807% for action SFT and 99.618% for action+XY. Terminal windows retain the number of original playable ticks observed. Last displayed Bomb inventory before each death is a camera measurement with decision-interval timing.

The independent CPU reforward reproduced all four sampled actions in every one of 17,708 scored decisions. The local clock advanced 59.88 and 59.99 Stage 1 native ticks per active wall second respectively. The two-slot campaign spanned 18.09 wall minutes from first original boot file to last confirmed scorecard, including rejected credits and the replacement. The stored example movie plays the real policy-camera sequence at measured wall speed. Full-color stock opening, Continue prompt and official final scorecards appear at the endpoints.

Detailed machine-readable files include `scored.json`, `reforward.json`, `complete.json`, `qualification.json` and the fixed `protocol.json`. `native-credits.release.json` resolves to the 260-entry raw bundle, including actual newly recorded original replays, complete action traces, read-only native key evidence, all registered attempted credits, both qualifying engineering credits, exact worker sources, and the paired film. No game executable or original data archive is included. `matched-film.release.json` lists the individually downloadable movie for the first registered seed. `scorecards.jpg` shows the 12 validated original result screens.