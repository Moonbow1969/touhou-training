# TH14 full-policy PPO pilot

## Setup

All three controllers run Stage 1 in the Steam build of TH14 1.00b, on Normal with Reimu A. The original game supplies state and enacts inputs. Each evaluation starts from the native Retry command and ends at the first miss, the next stage, or 3,300 decisions. One decision advances four nominal original-game frames. The first tests use greedy actions. The fixed evaluator samples the learned actors and rotates the arms between blocks.

| Arm | Movement / Focus / Shoot / Bomb | Geometric dodge | Boss aim correction | Automatic bomb |
| --- | --- | --- | --- | --- |
| SFT | Frozen SFT actor | Off | Off | Off |
| Full-policy PPO | PPO actor initialized from the same SFT checkpoint | Off | Off | Off |
| Hybrid baseline | Existing SFT actor with controller overrides | On | On, masked learned head | On |

PPO samples all four categorical heads during training and updates the shared encoder and all actor heads. The critic starts fresh. The runner asserts that the initialized PPO distributions match SFT exactly on a sampled input. The three arms share the native dialogue and Retry handling. No world model controls these runs.

The offline SFT checkpoint comes from the seed-61 title-balanced run on 147,560 decisions. The hybrid uses the masked aim checkpoint. Both SHA-256 hashes are in the run reports. The native executable SHA-256 is `4609e701b677b8317e43be92041f0c7075a292182848a7c11215c3a226034f90`. ParaKit offsets came from `2f2db40f045fd73bec4679c419d755a25230316c`.

## Initial budget

The first run uses 1,024 decisions and a first-miss survival task capped at 512 decisions per training episode. It awards +1 for surviving the cap or clearing Stage 1, −1 for a miss or native end, and 0 otherwise. Every completed training episode ended in a miss, so the run never observed a positive return. The actor sampled zero Bomb commands. Movement, Focus, Shoot, and Bomb output weights all changed during PPO.

| Single original-game trial | Outcome | Native frame | Decisions | Score delta, native units | Bombs used |
| --- | --- | ---: | ---: | ---: | ---: |
| SFT | First miss | 546 | 132 | 206 | 0 |
| PPO, 1,024 decisions | First miss | 510 | 123 | 177 | 0 |
| Hybrid | Stage 1 cleared without a miss | 9,894 | 2,470 | 1,084,402 | 0 |

The first SFT and PPO trials lost their first life around nine seconds into Stage 1. A 36-frame difference in one trial per arm gives no evidence of a PPO improvement. The [per-update metrics and native trial records](../../results/2026-11-23-th14-full-policy/sparse-1024-report.json) and the [SB3 CSV](../../results/2026-11-23-th14-full-policy/sparse-1024-progress.csv) include the four training episodes and exact reset and frame counts.

The next run increases the budget to 4,096 decisions and uses a separate PPO initialization from the same SFT. It rewards each surviving decision until the first miss or horizon. The native game objective supplies the reward without collision heuristics, aim corrections or Bomb overrides. The evaluation endpoints remain the same.

## Expanded budget

Foreground focus changed during training and again during the first set of evaluations. The first training segment logged an update at 2,304 decisions, but the durable checkpoint was at 2,048. The continuation restored that checkpoint and trained for 2,048 more decisions. It completed 4,096 decisions across 16 PPO updates and saved its final checkpoint before the second interruption. Of the 13 completed training episodes feeding the retained updates, all ended at the first miss. The furthest reached frame 1,970, and the policy sampled two Bomb commands over the retained decisions.

An evaluation-only run loaded the 4,096-step checkpoint and rotated the two learned arms over three more native Retry blocks.

| Arm | First miss frames | Median | Stage 1 clears |
| --- | --- | ---: | ---: |
| Frozen SFT | 390, 554, 450 | 450 | 0/3 |
| Full-policy PPO, 4,096 decisions | 578, 542, 478 | 542 | 0/3 |

The 92-frame difference measures survival while stationary. Greedy SFT emitted idle movement in all 335 decisions and greedy PPO did so in all 386 decisions, with zero horizontal or vertical moves and zero bombs across the six trials. The separately measured hybrid cleared Stage 1 twice with zero misses and zero bombs, at last observed frames 9,894 and 9,866. Each learned evaluation starts at frame 18. The second recipe changes both reward and budget together.

The machine-readable records preserve the discarded update, checkpoint hashes, incomplete evaluations and the full replicated comparison. [First training segment](../../results/2026-11-23-th14-full-policy/survival-4096-segment-1-report.json), [training continuation](../../results/2026-11-23-th14-full-policy/survival-4096-segment-2-report.json), [paired original-game trials](../../results/2026-11-23-th14-full-policy/survival-4096-paired-eval.json), [hybrid replication](../../results/2026-11-23-th14-full-policy/hybrid-replication.json)

## Movement verification after 8,192 decisions

PPO continued from 4,096 to 8,192 decisions using the same survival-time reward. The additional 4,096 decisions took 291.4 seconds, with 267.8 seconds in native stepping. Another foreground interruption left two completed greedy pairs and a valid checkpoint. Those greedy actors still chose idle on every completed decision.

The evaluator now samples both learned actors' categorical distributions by default, using matching trial seeds 61, 62 and 63. Each arm controls its own movement, focus, shoot and Bomb. The episode logs include commanded actions and the observed horizontal position range. Horizontal response counts consider commands for which the player is active both before and after the step, and use the sign of the measured displacement.

| Sampled arm | First miss frames | Median | Horizontal responses | Stage 1 clears |
| --- | --- | ---: | ---: | ---: |
| Frozen SFT | 1,690, 1,706, 1,574 | 1,690 | 581/592 | 0/3 |
| Full-policy PPO, 8,192 decisions | 914, 2,478, 4,666 | 2,478 | 689/805 | 0/3 |

PPO won two of three pairs. It used three policy-selected bombs in the longest trial. Both learned arms moved across the playfield. The 46.6% median difference is an initial result from one trained PPO seed and three native retry pairs. [8,192-step training log](../../results/2026-11-23-th14-full-policy/survival-8192-train.json), [sampled native trials](../../results/2026-11-23-th14-full-policy/survival-8192-sampled-eval.json)

## Collection throughput

The 1,024-decision smoke run took 75.5 seconds end to end for training, or 13.56 decisions/s. Native stepping took 66.9 seconds, Retry and the initial reset took 6.3 seconds, and the four learner updates took about one second combined. The game advanced exactly four frames on 1,022 of 1,024 decisions. A single worker at this measured rate would take roughly 20.5 hours to collect 1 million decisions.

The Windows runner sends foreground keyboard input. Concurrent controlled instances need independent input and game state. First establish a second worker on an x86 Linux host or Steam Deck through Proton or Wine with an independent display and prefix. Verify TH14 1.00b launch, memory observations, native clock, action delivery and 20 consecutive Retry cycles while the Windows worker is also running. An isolated Windows guest with its own virtual display is another option if the Linux route fails the compatibility check. The current Windows 11 Home host does not provide the Hyper-V role. [Valve Proton compatibility](https://partner.steamgames.com/doc/steamhardware/compat), [Xvfb](https://www.x.org/X11R6/), [Microsoft Hyper-V requirements](https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/get-started/install-hyper-v)

For the two-worker benchmark, record aggregate decisions per second, the native-frame delta histogram, resets, interrupted trials and CPU/RAM per worker. A starting target is at least 1.6× the single-worker throughput while preserving the four-frame rate and avoiding input cross-talk. Additional workers should collect against the same policy version for a PPO batch. H100 becomes relevant if a larger encoder or more workers make learner time material. Mac Studio, Mac mini and MacBook Pro can help decode replays or run offline jobs. A Windows-on-Arm game worker needs a separate timing and input check. [PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html), [Microsoft's Windows on Apple silicon options](https://support.microsoft.com/en-gb/windows/experience/platform-variants/options-for-using-windows-11-with-mac-computers-with-apple-m1-m2-and-m3-chips)

## Worker isolation tests

The Apple M3 Ultra host booted a diskless two-vCPU ARM64 guest to UEFI under QEMU 10.2.2 with explicit `accel=hvf`. QMP confirmed it ran and shut down cleanly. Windows and TH14 remain untested in that VM. [QEMU accelerators](https://www.qemu.org/docs/master/system/introduction.html#virtualisation-accelerators)

An x86 Linux host ran two concurrent Docker containers with Wine 8, independent virtual displays and independent prefixes. Both [Win32 fixtures](../../tools/rollout-worker-probe/README.md) presented a Direct3D 9 frame and received one key each through foreground DirectInput. Their memory counters froze at 4 and 7 ticks, remained fixed through the hold, and resumed to 14 and 17 ticks. The first attempt hung in the Xvfb wrapper startup. Explicit display readiness polling fixed it. The production gate is launching the Steam build of TH14 1.00b and verifying the original observations, exact native frame histogram and resets. [Probe results](../../results/2026-11-23-th14-full-policy/worker-probes.json)

The Linux host also ran two simultaneous copies of ZUN's genuine TH14 trial 0.01b into Normal Stage 1 at 640×480. The initial graphics-only setup stalled on the Stage 1 loading screen and emitted ALSA device errors. A private PulseAudio null sink per worker let both copies load. Sending Left to one and Right to the other put Reimu near opposite edges of the playfield, reproduced from fresh containers with the checked-in trial launcher. These game tests use the trial binary. Production PPO still uses the Steam 1.00b binary and awaits the same test with the full-version state reader. [Trial launcher](../../tools/rollout-worker-probe/README.md#th14-trial), [developer download page](https://www16.big.or.jp/~zun/html/th14dl.html)
