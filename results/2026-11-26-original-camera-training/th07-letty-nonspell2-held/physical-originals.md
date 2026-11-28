# TH07 Normal practice from Letty’s second nonspell

Forty independent starts of retail *Perfect Cherry Blossom* v1.00b tested a frozen camera policy and its curriculum PPO update. Each of the 20 controller seeds ran both policies on the same physical host and lane. Arm order was counterbalanced within the three lanes. Original game randomness remained uncontrolled.

| Frozen policy | Verified clears | Endpoints verified | Strict control | Spare lives at end, median | Bombs at end, median | Bomb units spent | Actions sampled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Frozen parent | 11/20 | 20/20 | 5/20 | 0 | 0 | 123 | 18,526 |
| Curriculum PPO | 13/20 | 20/20 | 10/20 | 0 | 1 | 122 | 19,869 |

The candidate missed the registered requirements of at least 50% clears and a gain of 20 percentage points. The difference in original section clear rates was +10 percentage points (paired-bootstrap 95% interval -20 to +40). Both arms met the stricter native input criterion in 3 pairs. The independent intake reports all physical outcomes alongside these strict pairs.

Every actor started from the official practice tool’s Normal Reimu A Boss Nonspell 2 selection, with 128 Power, two spare lives, three Bombs and zero continues. Four heads sampled movement, focus, shot and Bomb from the original visual input. The game’s own attack callbacks led from Sub39 into final spell ID 7 when the player survived. We required an original Stage Clear tally, a matching original result screen and a native stage-finished transition to score a section clear. Ordinary Game Start and arrival at Stage 2 require separate original tests.

Three original lanes on two hosts produced 62.2 complete starts per wall hour, including cold Wine setup, menu traversal, original play, post-game replay capture and sealing. The policy ran at 55.1 source game ticks per second during active control. The archive index records SHA-256 and byte counts for all four verified Release downloads (658.9 MB total).

[First assigned pair from entrance through its two native endpoints](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-retail-normal-nonspell2-first-pair.mp4) uses original 96×112 grayscale policy observations and a native clock overlay at 15 frames per second. A panel holds its original outcome when that run ends.

[Forty individual stocks, timing and four-head actions](per-original-stocks-and-actions.csv) · [Signed original cameras, inputs, alerts, tally images and game-generated replays](original-recordings.json) · [Original host throughput](actual-rollout-throughput.json) · [Prospective paired allocation](paired-registration.json) · [Checkpoint binding](released-binding.json)
[Independent paired confidence intervals and native input sensitivity](independent/results.json) · [Clear rates and first-loss figure](independent/clears-and-first-loss.png)
