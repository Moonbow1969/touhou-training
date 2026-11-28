# Shared original Normal training credits

Two physical hosts collected 16 qualifying full Game Starts for the same frozen four-head visual parent. Each started the original TH07 retail 1.00b on Normal with Reimu A, two spare lives, three Bombs and zero assists. All 16 reached the actual Game Over and saved an official zero-Continue replay. The archive contains 20,324 on-policy decisions and 81,338 original Stage 1 ticks. Each four-tick action window was exact; every retained mask was consumed. The maximum native observer read over the admitted set was 7.028 ms. None reached Stage 2. The parent spent 107 Bombs. These trajectories are designated TRAIN.

| Physical host | Qualifying full credits | Median original ticks / wall second | Range across credits |
|---|---:|---:|---:|
| Accelerator | 9 | 59.96 | 59.89–59.97 |
| Endeavor | 7 | 46.56 | 42.23–51.16 |

One native game and one CPU policy ran on each host. Values use in-game observation time and exclude setup, menus and transfer. Across all qualified credits the two hosts spent 853 s and 701 s respectively on the full worker lifecycle from first dispatch of gameplay input through recording its terminal. The startup of each isolated Wine profile adds separate wall time.

Seventeen original credits started. Seed 21749104 stopped advancing during Stage 1 at tick 872 with two spare lives and one Bomb. Two independently captured game screens were byte-identical, and the original replay manager clock, committed input slots and stock remained fixed. Its main Wine thread was running and a standard resume signal did not advance it. The original still had no Game Over after 192.72 seconds without another playable observation. Its partial trajectory and technical evidence are archived separately. The first preregistered technical reserve, 21749117, supplied the sixteenth qualifying credit.

Seed 21749103 had an additional startup attempt: the disposable Wine instance disappeared while focusing the window before any Game Start or policy input. The initial setup evidence is archived in its own directory. Its single actual Game Start subsequent to a focus-readiness repair controls the source row for that seed. That early provisioning attempt is outside the count of 17 games.

The release manifest authenticates the original replay, independent native inputs, applied actions, startup and result screens, per-step grayscale cameras, four behavior log probabilities, critic values, native times, resource events and checksummed archives. Every source resides in its seed-specific subdirectory. The candidate and subsequent official evaluations must keep these TRAIN seeds separate from held data.
