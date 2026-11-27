# TH17 original trial worker

The four-action camera policy plays ZUN's standalone 0.01a trial in a separate Wine prefix on a private 640 × 480 X11 client. The SHA-256 of the game executable is `3ffa3963003cce6dc02b1dae5adeb4411138772834efdb195a05b9081027aaba`. Each worker has a separate X server, game process, output directory, game profile and controller RNG. The inference service can share immutable TorchScript weights. Every client connection has private preceding-frame state. A held seed assigns both policies to the same physical host and private game instance. The seed controls policy sampling only.

## Environment and actor

Gamer WSL uses the stock Wine trial runtime with two CPU cores and a 1,800 MB limit per game. The Linux actor uses X11 and XTest through dynamic library loading. Compile in C11 mode for compatibility with the deployment image's glibc.

```sh
cc -std=c11 -O2 -I../th12-full -I../../full_action worker.c -o worker -ldl -lm
cc -std=c11 -O2 -I../th12-full record_original.c -o record_original -ldl -lm
i686-w64-mingw32-gcc-posix -O2 sensor.c -o sensor.exe
```

The game client starts at private screen position (3,22). Capture its native playfield at client coordinates [32,16,416,464], area-average each 4 × 4 block, then feed the resulting 96 × 112 luminance image and its absolute difference from the previous decision. The first difference is zero. The policy samples movement from nine directions, and samples focus, shoot and Bomb separately. The control hold is 120 ms. All seven X11 key states are read back after application. The actor logs each input frame, action, per-head log probability, critic output and control timestamps.

Open the game through Start, select Hard or Lunatic on the original difficulty screen, then choose the default Reimu and Wolf. Save the initial and final rank-menu images and the opening HUD. In the directory for a fresh credit start `sensor.exe` under Wine, then invoke `TOHO_CAMERA_SHA=<checkpoint-sha256> TOHO_CAMERA_SOCKET=<socket> worker <controller-seed> <rank>` before entering Stage 1. The actor waits for the native initial stock. It releases all keys on stock exhaustion. Collect the original terminal screenshot. A failed or incomplete acquisition requires review.

## Native observations and endpoints

The observer reads the immutable trial's original game-status block at 0x4b1634. In units of four bytes the checked offsets are stage at 0, start stage at 1, stage clock at 3, character at 6, beast at 7, score at 8, difficulty at 9, continues at 10, power at 21, spare lives at 25, life pieces at 26, Bombs at 28 and Bomb pieces at 29. Signed life stock is 2 at a fresh game, 0 on the last playable life and −1 at death terminating the credit. QA independently compared both rank menus and HUDs, the full stock, the Game Over menu in the Hard control, and an original highlighted Stage 1 score-ranking row in Lunatic. The policy receives only playfield pixels. A loss triggers a native Bomb refill even on the last life. Report last playable Bomb stock and post-death Bomb stock independently. Stage 2 inventory is the original stock at the first native transition. Sparse progression rewards count the first new stage, each native life gained or lost, and a same-life Bomb consumption. The frozen protocol omits a dense time reward because combat-specific clock validity is unestablished.

`verify_camera.py` checks the pre-action frame and model hashes and evaluates every sampled action's per-head log probability and critic on a second machine. `render_credit.py` reconstructs the policy camera and controls on the original recorded timestamps. Selected runs also save an independent full-color original-client video at 5 FPS with monotonic times bracketing every decision. The study reports observer coverage and preserves technical attempts separately.
## Collect a credit from a local game installation

Run the following from the repository root on an x86-64 Linux system or WSL2 with Docker. The host needs Bash, a C compiler, Python 3 and GNU coreutils. Put the extracted, locally obtained original trial files `th17.exe`, `th17tr.dat` and `thbgm_tr.dat` in their own directory. A trained TorchScript must implement the four categorical outputs [9, 2, 2, 2] with an optional fifth scalar critic. Without a critic the actor records zero. The output directory must be fresh. Keep the temporary socket on a Linux filesystem when using WSL.

```bash
docker build -t toho-worker-probe tools/rollout-worker-probe
docker build -f tools/rollout-worker-probe/Dockerfile.trial -t toho-worker-trial tools/rollout-worker-probe
docker build -t toho-camera-cpu tools/distributed-ppo-pilot

bash tools/rollout-worker-probe/th17-full/run-credit.sh \
  --game /licensed/th17tr --model /checkpoints/four-title/policy.ts \
  --output /data/th17/train-four-sft-1211701 --seed 1211701 \
  --rank hard --arm four-sft --split train --lane alpha --cpus 2
```

The command compiles the native reader and Linux controller, checks the original binary, starts private X11/Wine and inference containers with networking disabled, and drives the original difficulty, character and beast menus. It saves the full credit, original photographs, resource traces and SHA-256 receipts. It removes its two temporary containers at exit. For Lunatic select `--rank lunatic`. To collect independent credits concurrently, use distinct episode directories. For multiple game instances sharing one frozen local camera model, pass `--socket /path/to/actor.sock` to reuse a running `tools/full_action/camera_service.py` that advertises the requested model hash. Set `--wine-image` and `--camera-image` to select previously built images. The registered study used 2 CPU cores per game and held the lane, executable and input cadence fixed within every comparison.

The published sensor binary was built with MinGW under Debian bookworm and `SOURCE_DATE_EPOCH=1795755599`. The pinned formula reproduced SHA-256 `a561ed8f077b9d75fc380777722a67e59f9dbbad1dcbc1108c1083be8f1dff69` in an independent container. Each new run records its compiler output's actual SHA. The calibration file contains luminance patches from original menu observations; the stage-reference file contains two binary glyph shapes from an original highlighted ranking row. A death endpoint requires signed stock exhaustion and a recognized original terminal screen. A trial clear or any unrecognized ending needs separate evidence review and remains unsealed.

The launcher runs the native controller once in the selected Wine image before starting the game. This checks that the deployment image can load the compiled executable. When the host compiler targets a newer libc, pass `--actor-builder <image>` to compile inside a Linux image with a matching libc and C compiler. Debian bookworm is the runtime used by the published Wine image.

```bash
python tools/rollout-worker-probe/th17-full/verify_camera.py \
  /data/th17/train-four-sft-1211701 /checkpoints/four-title/policy.ts
python tools/rollout-worker-probe/th17-full/render_credit.py \
  /data/th17/train-four-sft-1211701 /data/th17/whole-credit.mp4
python scripts/report_th17_parallelism.py /data/th17 /data/th17/throughput.json
```

Each sample in the NPZ contains the raw pre-action camera, movement/focus/shoot/Bomb, four action log probabilities, critic, event reward and final marker. To independently reconstruct an NPZ from preserved native rows and original screenshots, invoke `receipt.py EPISODE --identity EPISODE/identity.json`. Reconstructing all eight training NPZs of the four-title study yielded identical file hashes across 5,348 decisions.
The raw legacy TSV column named `tick` is the stage clock, `level_tick` is the chapter clock, and `total_tick` is the music clock. Music or chapters can restart without stage advancement. Perform event-to-action matching using the ordered pair (original stage, stage tick), and use the monotonic acquisition clock for wall duration. The stage clock is validated inside each stage; a transition from Stage 1 to Stage 2 can legitimately restart it at zero. The regression can be run with `python tools/rollout-worker-probe/th17-full/test_stage_clock.py`. Its synthetic timer wraps exercise the reader and do not count as an original-game Stage 2 witness.
The generation-two acquisition can select `--observer v2`. It records state changes even while the music clock is stopped. With the pinned compiler its SHA-256 is `803154279225a61f80ed1b6a8b7a1b8a094b94bfdabd2f08052c549f7898b2ee`. In simultaneous original Hard and Lunatic full-credit controls the two observers agreed on all 4,985 shared stage/ticks and all ordered native stock changes. Version 2 also explicitly saw the zero-filled Stage 1 initializer before the game assigned 2 spare lives and 3 Bombs. These pre-play rows are retained in raw TSV, counted in the receipt, and excluded from reward. Qualification receipts, build command and paired observer traces are in `results/2026-11-26-original-camera-training/four-game-ppo-generation2/th17-observer-qualification`.
