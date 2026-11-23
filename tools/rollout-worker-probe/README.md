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
