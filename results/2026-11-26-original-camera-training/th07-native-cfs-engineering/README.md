# Two original TH07 CPU-scheduler measurements

Two separate engineering starts used the same Gamer CPU affinity (4, 6, 8 and 14), model, opening, rank and stock. Each ran to a real 2,048-native-tick boundary with a retained next camera and independently re-forwarded critic. The two sampled-action seeds differed. These measurements are engineering only.

| Per-credit result | Two-CPU quota | No CFS quota |
|---|---:|---:|
| Original ticks/s | 57.12 | 56.90 |
| Sampled and consumed actions | 511/511 | 512/512 |
| Exact windows | 510/511 | 512/512 |
| Freeze duration p99 | 18.37 ms | 14.61 ms |
| Longest freeze | 28.67 ms | 18.88 ms |
| Increase in throttled periods | unavailable | 0 |

The capped run preserved its initial cgroup snapshot. Its post-game serializer failed after the real cutoff and the private container closed before the second CPU snapshot. The native trajectories and true bootstrappable next states survived for both allocations. Each allocation has a single observation. [Per-game results, input hashes and raw counters](scheduler-comparison.json) · [Download the two native sources](published-sources.json)

