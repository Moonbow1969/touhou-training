# 261125 · Steam Deck TH12 rollout timing

Raising the per-instance CPU quota from two to three increased rendered FPS from **33.4 to 46.1**, averaging each credit's median, across two simultaneous TH12 instances. The scripted Stage 1 title also appeared earlier in both instances. It was first sampled at **12.5 s** with three CPUs and at **16.2–19.8 s** with two. The largest Alpha–Beta difference was 0.92 s, about one camera interval. This limits the case for a persistent speed difference between the two lanes as an explanation for the earlier survival crossover.

The benchmark ran six fresh 40 s credits in a two → three → two CPU sequence on the same eight-logical-CPU Steam Deck. Each block used the original TH12 Trial 0.02a, Normal Reimu Type A, fixed Shoot, zero Bomb and zero Continue. Both lanes ran the same frozen initial controller with seed `122501`. The controller has state-independent logits, and replay verification matched the first 115 emitted actions across all six credits. The two higher-quota credits each had room for two more decisions. All six reached the time cap.

| Block | CPUs per lane | Lane | Median FPS shown by game | Stage 1 title appears, wall s | Action to capture, median ms | CFS periods throttled |
|---|---:|---|---:|---:|---:|---:|
| A1 | 2 | Alpha | 34.5 | 15.26–16.16 | 344.4 | 100% |
| A1 | 2 | Beta | 34.7 | 16.18–17.08 | 348.2 | 100% |
| B | 3 | Alpha | 46.2 | 11.62–12.51 | 342.5 | 4.9% |
| B | 3 | Beta | 46.0 | 11.61–12.50 | 342.4 | 5.7% |
| A2 | 2 | Alpha | 32.0 | 18.92–19.82 | 344.7 | 100% |
| A2 | 2 | Beta | 32.3 | 18.89–19.79 | 347.9 | 100% |

Original-color captures taken about every 0.90 s bracket the first visible Stage 1 lettering. The visible sample span also contracted from 7.42 s to 5.38 s, a measurement that is independent of reset-to-first-action alignment. Renderer FPS comes separately from confidence-filtered matches against the game's FPS HUD, starting at wall second five. Each credit contributed 29–36 accepted readings. Exact game tick counts and game RNG remain unavailable. Cgroup counters span the reset, active credit and final pause. Median time from one action onset to the next was 347.9–350.0 ms at two CPUs and 344.5 ms at three.

**Deck rollout profile.** For two simultaneous instances, use `--cpus 3 --memory 2g --network none` with one private Xvfb display per container and at least eight logical CPUs on the host. Both owned scopes accepted the three-CPU profile after the reversal. Average CPU use per lane rose from 2.00 to 2.92 CPU equivalents. The faster profile improves per-lane gameplay progress. The experiment did not measure the total throughput of additional lanes. Keep CPU profiles fixed across comparable wall-clock evaluations because identical wall holds span different amounts of game progress when the quota changes.

The timing samples, action sequences, cgroup deltas and fingerprints are in [the companion data](th12-deck-native-clock-ab.json).
