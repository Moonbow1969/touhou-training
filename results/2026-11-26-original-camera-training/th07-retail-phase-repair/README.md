# TH07 retail input timing qualification

Two independent Normal Reimu A games on Momentum consumed every scheduled four-tick action. Both reached the original Game Over screen, saved an official replay, and used zero continues. Polling the active retail record slot replaced a fixed 1.5 ms wait. Each game and its inference process also received dedicated CPU affinity with unlimited bandwidth. The two changes were applied together.

| Credit | Exact windows | Skipped | Native ticks/s of active play | Longest native read | Longest freeze |
|---|---:|---:|---:|---:|---:|
| 2171701, parent | 826/826 | 0 | 59.91 | 3.44 ms | 11.67 ms |
| 2171702, earlier human SFT | 866/866 | 0 | 59.93 | 6.30 ms | 8.48 ms |

Current and preceding input words agreed with the official saved replay at the registered offset in every compared tick. Across 235 and 243 cgroup snapshots, respectively, the throttled-period count stayed at zero. Both credits ended in Stage 1 with zero spare lives and three Bombs. These two independently seeded credits qualify precise control for future policy comparisons. They do not measure the effectiveness of either policy.

The official replay, terminal score cards, four-head decisions, native snapshots, timing trace, and individual SHA-256 digests are indexed in [the release receipt](raw.release.json).
