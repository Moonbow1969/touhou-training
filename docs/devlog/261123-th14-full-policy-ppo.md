# TH14 full-policy PPO pilot

## Setup

All three controllers run Stage 1 in the Steam build of TH14 1.00b, on Normal with Reimu A. The original game supplies state and enacts inputs. Each evaluation starts from the native Retry command and ends at the first miss, the next stage, or 3,300 decisions. One decision advances four nominal original-game frames. Evaluation uses deterministic actions and rotates controller order between blocks.

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

## Collection throughput

The 1,024-decision smoke run took 75.5 seconds end to end for training, or 13.56 decisions/s. Native stepping took 66.9 seconds, Retry and the initial reset took 6.3 seconds, and the four learner updates took about one second combined. The game advanced exactly four frames on 1,022 of 1,024 decisions. A single worker at this measured rate would take roughly 20.5 hours to collect 1 million decisions.

The Windows runner sends foreground keyboard input. Concurrent controlled instances need independent input and game state. First establish a second worker on an x86 Linux host or Steam Deck through Proton or Wine with an independent display and prefix. Verify TH14 1.00b launch, memory observations, native clock, action delivery and 20 consecutive Retry cycles while the Windows worker is also running. An isolated Windows guest with its own virtual display is another option if the Linux route fails the compatibility check. The current Windows 11 Home host does not provide the Hyper-V role. [Valve Proton compatibility](https://partner.steamgames.com/doc/steamhardware/compat), [Xvfb](https://www.x.org/X11R6/), [Microsoft Hyper-V requirements](https://learn.microsoft.com/en-us/windows-server/virtualization/hyper-v/get-started/install-hyper-v)

For the two-worker benchmark, record aggregate decisions per second, the native-frame delta histogram, resets, interrupted trials and CPU/RAM per worker. A starting target is at least 1.6× the single-worker throughput while preserving the four-frame rate and avoiding input cross-talk. Additional workers should collect against the same policy version for a PPO batch. H100 becomes relevant if a larger encoder or more workers make learner time material. Mac Studio, Mac mini and MacBook Pro can help decode replays or run offline jobs. A Windows-on-Arm game worker needs a separate timing and input check. [PyTorch MPS](https://docs.pytorch.org/docs/main/notes/mps.html), [Microsoft's Windows on Apple silicon options](https://support.microsoft.com/en-gb/windows/experience/platform-variants/options-for-using-windows-11-with-mac-computers-with-apple-m1-m2-and-m3-chips)
