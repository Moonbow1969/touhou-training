# TH14 Normal Reimu A full-credit experiments

Stock Steam TH14 1.00b. Each credit starts at Stage 1 frame 30 with two spare lives, three Bombs and zero Continues. The frozen actors sample movement, focus, shooting and Bomb every four native frames. All 25 accepted credits ended in native Game Over. One baseline attempt lost foreground and was rerun from a fresh credit. No accepted credit timed out. Stage 2 entry inventories are missing for all arms because none reached Stage 2.

| Frozen checkpoint | Credits | Stage 2 | Median first miss | Median Game Over | Bombs consumed | Bombs at Game Over, by policy seed |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| SFT | 4 | 0/4 | 978 | 2321.5 | 2 | 3, 3, 3, 2 |
| PPO-211 | 4 | 0/4 | 1504 | 4775.5 | 34 | 0, 0, 0, 0 |
| PPO-397 | 4 | 0/4 | 1252 | 5039 | 1 | 3, 3, 3, 3 |

The original comparison rotated the three actors over policy sampling seeds 1201–1204. Each credit used the native game RNG, which was uncontrolled. The final live Bomb inventory is read at Game Over. Bombs consumed count observed inventory drops on steps without life loss. [Full native log](report.json) · [Per-credit data](stage1-credits.csv) · [Figure](figures/stage1-first-miss-and-endpoints.png)

## Isolating the Bomb head

The candidate uses PPO-397 for movement, focus and shooting and PPO-211 for Bomb. Both frozen networks receive the same observation. Offline checks verified exact head probabilities and sampling order against the source policies. In fresh rotating trials the candidate reached a later Game Over than both parents in each of the three blocks. Its margin over PPO-211 on seed 1403 was five frames. [Head verification](bomb-crossover-verification.json) · [Native trials](bomb-crossover-native.json) · [Figure](figures/bomb-crossover.png)

| Policy sampling seed | PPO-397 end frame | PPO-211 end frame | Crossover end frame |
| --- | ---: | ---: | ---: |
| 1401 | 2961 | 4930 | 8350 |
| 1402 | 3551 | 3702 | 8489 |
| 1403 | 3558 | 5178 | 5183 |

## Testing single-network distillation

We fit the two final Bomb output rows in PPO-397 to PPO-211 probabilities using 45,118 replay observations. On two held-out full replays the other three heads retained their original logits. On the Stage 1 subsets the student removed 99.44% and 99.75% of the original teacher-to-student Bomb KL divergence. A two-block native test measured the cost of that approximation. [Training and validation](bomb-distillation.json) · [Native trials](bomb-distilled-native.json) · [Figure](figures/bomb-distilled.png)

| Seed | Exact crossover end frame | Distilled end frame | Exact / distilled Bombs consumed |
| --- | ---: | ---: | --- |
| 1501 | 8236 | 6122 | 7 / 4 |
| 1502 | 8712 | 4837 | 8 / 3 |

The exact crossover remains the reference. Across its five credits, four reached the Stage 1 boss spell with native ID 4. In all four it entered with zero spare lives and zero Bombs. This identifies a repeatable late-stage evaluation point for testing both spell execution and the cost of resources spent earlier. The small trial counts and uncontrolled game RNG limit estimates of general clear probability.

A separate supervised movement experiment on three training players tested event-weighted and unweighted output-layer updates. Neither recovered an additional exact movement start or turn on the two development replays. [Movement ablation](movement-head-offline.json)
