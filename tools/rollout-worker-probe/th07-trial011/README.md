# TH07 0.11 four-native-tick original worker

This worker was qualified on the official 0.11 trial with Normal Reimu A, two spare lives and three Bombs. The two-channel policy sees the 96×112 grayscale playfield and its absolute change from the preceding decision. On the first action the difference channel is zero. Original clocks, held keys, HUD and replay files support validation after the action. The actor receives camera pixels as its only input.

The Linux host needs a licensed or official trial installation, the registered Docker image, a private Wine prefix and X display, XTest, CPU PyTorch, Pillow, NumPy and host ffmpeg for rendering the prespecified example pair. The registered source installation and original menu references are described under `results/2026-11-26-original-camera-training/th07-original-normal` and `th07-original-normal-ppo`. The new protocol binds the existing install and two trained models by digest. The target used two independent one-game containers on Momentum.

## Control timing

The native observer waits for a positive original clock divisible by four. After 1.5 ms for that tick's current input poll, it stops the TH07 process, verifies the process has stopped, brackets the X11 image with the same original tick and input state, and sends the exact camera to the resident CPU actor. It commits the sampled keys in one XTest batch, waits for XSync, checks the clock again, and resumes the original process. A watchdog resumes a stalled or orphaned game. The role of XSync is to confirm delivery to the X server. The separate original replay and live-key reconciliation establishes actual consumption by the game.

The stage clock is at 0x11c9ae8. The two input words are at 0x11c9d24 and 0x11c9d2c. During live recording the settled word at native tick t agrees with the new official replay word at index t+1. The operator can reproduce this calibration using every settled native word across offsets −2 through +2 in each credit. During playback of the older human source files, current word at t matched archived word t. The two collectors use these independently measured phases.

A full eligible window covers native ticks t+1 through t+4. The verifier records gaps, wrong masks, complete stale actions and the amount of playable terminal residue. It compares actual original replay words with the four sampled requests. The reference screenshot independently verifies zero continues, three misses on the observed Game Over path, and Normal difficulty. Postgame glyph verification uses prior hand-transcribed original scorecards and independently verifies the score from the replay header. A scorecard or game state outside this calibrated grammar requires separate original readback.

## Build and restore

From the source directory on Momentum, these commands build the bridge and the offline watchdog witness.

```sh
gcc -O2 -Wall -Wextra -o tick4_actor tick4_actor.c -ldl
gcc -O0 -c -o watchdog-reader.o tick4_actor.c
objcopy --redefine-sym main=bridge_main --globalize-symbol=mono_ns --globalize-symbol=freeze_game --globalize-symbol=protect_parent watchdog-reader.o watchdog-under-test.o
gcc -O2 -Wall -Wextra -o watchdog_fixture watchdog_fixture.c watchdog-under-test.o -ldl
timeout 8 ./watchdog_fixture
```

The fixture only creates and suspends its own dummy subprocess. In the recorded run the independent watchdog released it in 753 ms after a controller stall and 51 ms after forced controller termination.

The registered host root is `/var/tmp/moonbow-th07-trial-261127`. The run scripts read their external root from the host mapping in their source. Copy `run_credit.py`, `run_pairs.py` and the study's native protocol into `locator-bc/native/` (protocol file name `protocol.json`). The native executable, credit/verification/render scripts and the original Continue-text mask belong in `locator-bc/native/tool/`. The mask is 28×192 bytes in row-major order, cropped from original rectangle (114,175,306,203), grayscale greater than 218, encoded as uint8 0/1. The previous original Transcribed scorecard file is `reference-scorecards.json`. Place frozen weight directories at `locator-bc/fit/71281-{action,action-xy}/policy.ts`.

Each engineering run and each enrolled arm uses a new output directory, network-disabled container and stock original game. Warming the PyTorch graph on a blank image occurs at the menu. `run_pairs.py` verifies both prescribed engineering reports before enrolling a cohort. Every enrolled seed retains both arm directories and its scheduled opener. Source clock alignment, terminal snapshots, scorecards, traces, controller seeds and model SHA-256 are kept together. Rendering uses stored real camera frames with a one-times wall playback timeline and full-color original opening/terminal stills, keeping encoding outside the controller loop.
