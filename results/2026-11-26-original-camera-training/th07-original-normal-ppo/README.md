# TH07 Normal, first original-game PPO update

Four new full credits in ZUN's official Perfect Cherry Blossom 0.11 trial supplied 3,217 decisions for one joint four-head PPO update. All credits started in ordinary Normal with Reimu A, two spare lives and three Bombs. The six-game parent saw the 96 × 112 grayscale playfield and its difference from the previous observation. Two independent Xvfb/Wine instances on Momentum ran the game. The original 2070701 qualification credit was excluded. [Preregistration](protocol.json) · [Original-game qualification](../th07-original-normal/qualification.json)

The optimizer restored the parent's learned critic. The reward was 0.002 per control step and −1 per confirmed life loss, including the final one. Two epochs produced 52 Adam steps and final approximate joint KL 0.00180. The scores and explicit Bomb presses had zero reward weight. All four categorical heads changed. The training report records source hashes and accepted gradient steps. [Fit report](policy/report.json) · [Frozen actor](policy/policy.ts)

| Fresh control seed | Parent → updated progress | Parent → updated score | Successful Bombs | Last visible Bombs |
|---|---:|---:|---:|---:|
| 2080751 | 5.34% → 5.84% | 714,310 → 651,470 | 8 → 8 | 0 → 0 |
| 2080752 | 3.77% → 3.26% | 411,300 → 261,340 | 5 → 3 | 2 → 1 |

The original results showed zero continues and three misses in all eight credits. Neither arm reached Stage 2 in its two held games. Native progress moved in opposite directions across the two pairs. Each pair shared a physical host, with opposite arm orders between pairs. Policy sampling used matched seeds while the original game's RNG remained uncontrolled. The result screens also record simulation slowdown, which ranged from 0.17% to 10.58% in the held games. [Eight original scorecards](eight-original-scorecards.png) · [Per-credit results and actor checks](result.json)

The release contains the eight native replays and scorecards, pre-action camera frames, actions and four log probabilities, values, event-aligned rewards and terminal flags, two full original recordings for held seed 2080751, and the trained weights. The physical collector uses the shared launcher and the registered seed schedule. The independent verifier replays the actors on every archived camera state. The host fitter imports the repository's `replays/inspect_rpy.py` as `inspect_rpy`. [Raw evidence](https://github.com/Moonbow1969/touhou-training/releases/download/camera-training-2026-11-26/th07-original-normal-ppo-eight-original-credits.zip) · [Verified release SHA-256](eight-original-credits.release.json) · [Reproduction sources](source)
