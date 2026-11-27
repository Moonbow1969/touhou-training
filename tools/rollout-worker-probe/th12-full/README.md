# TH12 full-action original-game collector

The worker runs the official *Undefined Fantastic Object* trial 0.02a in an isolated 640×480 Wine window. A policy acts from the 384×448 playfield downsampled to 96×112 grayscale. At each control it samples a nine-way direction, Focus, Shoot and Bomb. `XQueryKeymap` checks the seven physical key states after input delivery. The HUD, native Game Over menu and Score Ranking serve as evidence for the experiment. The actor receives only playfield pixels.

## Build and stage

From this directory on an x86-64 Linux build host, compile against the repository's portable actor. Xlib and Xtst are resolved dynamically inside the Wine container.

```sh
cc -std=c11 -O2 -Wall -Wextra -Werror -DHUMAN_ACTOR -I../../full_action worker.c -lm -ldl -o th12-full-actor
cc -std=c11 -O2 -Wall -Wextra -Werror worker.c -lm -ldl -o th12-full-qual
```

Create `input`, `game` and `capture` under `TH12_ROOT`. Stage the two workers and a frozen portable `policy.bin` inside `input`. Copy `credit.sh`, `reset.sh`, `return_title.sh` and `desktop.sh` there under the names `th12-full-credit.sh`, `th12-full-reset.sh`, `th12-full-return-title.sh` and `th12-full-desktop.sh`. Fill `game` from your local TH12 trial installation. The launcher checks the original EXE hash. Set `TH12_CONTAINER_IMAGE` to the locally built Wine/Xvfb image with `xdotool`, `scrot`, `flock`, `pactl` and the null PulseAudio sink.

Collect seven small screenshot references from the same game and HUD skin. For each image, run `python template.py KIND SCREENSHOT input`. `ranking-full` is the score entry screen, `pause` is the pause menu, `gameover` is the normal Game Over menu, `retry` is the highlighted retry option, `confirm` is the retry confirmation dialog, `hud` is the active HUD and `title` is the initial title screen. Use the intact original 640×480 window or a 1024×768 screen with the game at (3,22). Running `sh launch.sh` starts two private 3-CPU, 2-GiB workers. Each uses a separate ephemeral game copy and writable capture directory.

## Original credit

Use a fresh label and the SHA-256 of the frozen weight file. The seeded RNG belongs to the policy. TH12's native RNG remains uncontrolled. Within a held seed, rotate policy assignment between machines' local alpha and beta lanes on alternate seeds.

```sh
scope=toho-th12-full-alpha-261125
podman exec "$scope" sh /input/th12-full-credit.sh qualify calibration 719980 policy.bin ignored
podman exec "$scope" sh /input/th12-full-credit.sh held held-sft-721001-alpha 721001 policy.bin "$policy_sha256"
```

Controls are held for 120 ms, with capture and inference overhead recorded separately. Qualification deliberately exercises all directions and modifiers. The real policy controls each gameplay input. A credit starts from the original Normal story title with Reimu A, two reserve lives and two Bombs. It runs to native Game Over or native Score Ranking. The 1,200-second technical watchdog yields an incomplete trajectory. The actor releases its pressed keys at termination. Score Ranking is recognized even when a held shoot key crosses the terminal transition. Reset runs after the prior credit is sealed. No credit consumes a Continue.

For each recognized terminal, the worker saves `terminal-at-control.ppm` directly from the XImage that triggered the final observation. Its measured 608×448 region is placed at (32,16) in the 640×480 original-game coordinate system, with uncaptured borders displayed as black. `score_stage.py` can verify a highlighted native score-row stage against a manually certified original screenshot. Each isolated desktop holds a per-credit lock throughout reset and gameplay.

The raw journal includes action, per-head behavior log probabilities, value, sampled features, physical key masks, native spare/Bomb counts, monotonic times and an original color camera every ~1.5 seconds. Training credits also retain each grayscale observation. `summarize.summarize(raw, output, stage2_step=...)` seals an NPZ used by the shared PPO learner. Review native progression before passing a Stage 2 decision; rankings or the original color chronology give independent stage evidence. A full-credit loss and a full game clear are distinct endpoint classes. This collector seals the former and flags other endpoints for review.
## Direct visual policy and timing

Build `worker.c` with `-std=c11 -DHUMAN_ACTOR -DCAMERA_ACTOR`, include `tools/full_action`, and link `-ldl -lm`. Install that executable as `/input/th12-camera-actor`. Each native lane uses one private socket from `tools/full_action/camera_service.py`. Set `TH12_ACTOR_KIND=camera`, `TOHO_CAMERA_SOCKET` and `TOHO_CAMERA_SHA` on the original game's credit process. The hash identifies the frozen TorchScript passed to the service. Each decision sends the exact pre-action 112 × 96 grayscale frame. The service supplies direction, focus, shot and Bomb logits and the optional critic value. Reset the connection at each credit boundary to zero the first frame difference. Both training and held camera credits preserve every frame. The sealed NPZ retains `frames[T,112,96]` and `camera_logits[T,15]`. The summarizer checks the four realized action log probabilities against those logits.

`python display_fps.py evidence/00020.jpg` reads the original engine's displayed frame rate from a native 640 × 480 game capture. Its output includes the digit match margins and a readability flag. The supplied glyphs were calibrated on the 0.02a display at native size. For a cohort report, select color captures whose timestamps fall inside the active credit, discard unreadable readings, compute each credit's mean, then weight credits equally within the same host and policy arm. This measures presentation speed. Native Stage 2, resources and original terminal screens provide the game-progress endpoints.
