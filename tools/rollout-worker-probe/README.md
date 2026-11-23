# Wine rollout worker probe

This 32-bit Win32 fixture validates input, cross-process memory reads and debugger stepping under Wine. It runs with a private X virtual display and Wine prefix per container.

```sh
docker build -t toho-worker-probe tools/rollout-worker-probe
docker run --init --rm --network none --cpus 1 --memory 900m toho-worker-probe &
first=$!
docker run --init --rm --network none --cpus 1 --memory 900m toho-worker-probe &
second=$!
wait "$first" && wait "$second"
```

Each process must render and present a Direct3D 9 frame, receive exactly one Z key-down through foreground DirectInput, keep its tick counter stable under `DebugActiveProcess`, and advance it after `DebugActiveProcessStop`. `probe.c` reads the counter via `ReadProcessMemory`. The fixture uses a Win32 message timer. TH14 needs separate native timing, observation and Retry checks.

## TH14 trial

Install the original 0.01b trial locally and expose its `th14.exe`, `th14tr.dat` and `thbgm_tr.dat` through a read-only mount. Run one container per direction. Each one boots into Normal mode at 640×480, supplies a private null audio device and writes its own game screenshot. [Original trial download](https://www16.big.or.jp/~zun/html/th14dl.html)

```sh
docker build -f tools/rollout-worker-probe/Dockerfile.trial -t toho-trial tools/rollout-worker-probe
game=/absolute/path/to/installed/trial
mkdir -p capture/left capture/right
docker run --init --rm --network none --cpus 3 --memory 1536m \
  -v "$game:/input/game:ro" -v "$PWD/capture/left:/capture" toho-trial Left &
first=$!
docker run --init --rm --network none --cpus 3 --memory 1536m \
  -v "$game:/input/game:ro" -v "$PWD/capture/right:/capture" toho-trial Right &
second=$!
wait "$first" && wait "$second"
```

The captures demonstrate game launch and input isolation. Use the full Steam 1.00b build with its matching telemetry to benchmark online sample throughput.

## Frame-stepped trial worker

Pass `serve` to the same image in an interactive stdin pipe. It emits JSON for the original Stage 1 observation at startup, then accepts `S MASK` for a four-tick action, `R` for a native Retry, and `Q` to close. The seven mask bits are Left, Right, Up, Down, Focus, Shoot and Bomb, from least to most significant. The observer records original ticks, stage frame, lives, player position and up to 96 nearby projectiles.

Use a private JSON file that maps each worker name to the argument array of its `ssh -T` command. The SSH command runs `docker run -i` or rootless `podman run -i` with an independent capture directory and the same read-only trial mount. Start several in parallel with the benchmark command.

```sh
python -m scripts.benchmark_trial_workers \
  --workers local-workers.json --episodes 6 --decisions 128 \
  --out runs/th14-trial-workers.json
```

Each worker keeps its own X display, Wine prefix and audio sink. The worker checks the trial executable hash before reading memory. Keep the Steam 1.00b offset table and benchmarks separate.

For a heterogeneous pool, pass `--worker-episodes WORKER=COUNT` for each faster or slower worker that needs a different number of episodes. The output records each worker's cold startup, first stage frame, step latency, clock deltas and Retry frames. A run with a failed worker keeps the other workers' rows and leaves the aggregate rate empty.

The [Steam Deck DXVK setup](steamdeck-dxvk.md) uses an isolated hardware-rendered virtual display. Keep the software renderer available when qualifying the DXVK path or investigating a display failure.
