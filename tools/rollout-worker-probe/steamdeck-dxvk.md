# Steam Deck hardware rendering for the TH14 trial

The Steam Deck ran the TH14 0.01b Normal Stage 1 probe at **11.96 decisions/s** using 32-bit DXVK and AMD RADV. The Xvfb and WineD3D run on the same Deck reached **6.76 decisions/s**. The current native-v6 GPU launcher independently reproduced **12.01 decisions/s**. All three completed runs accepted 256 four-tick decisions and all eligible horizontal inputs.

## Build and start

The measured base image was `localhost/toho-trial:native-v5`. The current checkout builds `native-v6` with clock-driven menu selection. Both use Debian Wine 8.0 and the 32-bit prefix setup. The Steam Deck host provides Gamescope 3.16.30.1, Xwayland, `xdpyinfo`, Python and rootless Podman. The user running the launcher must be able to open `/dev/dri/renderD128` for reading and writing. The graphics dependencies install inside the image.

```sh
probe=tools/rollout-worker-probe
podman build --jobs=1 --tag localhost/toho-worker-probe:native-v6 "$probe"
podman build --jobs=1 --build-arg PROBE_IMAGE=localhost/toho-worker-probe:native-v6 \
    --tag localhost/toho-trial:native-v6 --file "$probe/Dockerfile.trial" "$probe"
podman build --jobs=1 --build-arg BASE_IMAGE=localhost/toho-trial:native-v6 \
    --tag localhost/toho-deck-vulkan:v6-dxvk-2.3.1 \
    --file "$probe/Dockerfile.steamdeck-dxvk" "$probe"

project="$HOME/.local/share/toho-ml"
install -d "$project/steamdeck-dxvk" "$project/runs"
install -m 700 "$probe/steamdeck-dxvk.sh" "$project/steamdeck-dxvk/steamdeck-dxvk.sh"
TOHO_DXVK_IMAGE=localhost/toho-deck-vulkan:v6-dxvk-2.3.1 "$project/steamdeck-dxvk/steamdeck-dxvk.sh"
```

To reproduce the measured v5 run on a Deck with `localhost/toho-trial:native-v5` already built, pass `BASE_IMAGE=localhost/toho-trial:native-v5` to the DXVK build, tag the resulting image as `localhost/toho-deck-vulkan-diagnostic:dxvk-2.3.1` and run the same wrapper with its default image.

Put the legally obtained 0.01b trial assets under `$project/trial` and keep that directory read-only in the container. The underlying trial launcher also checks the executable hash. The tested image resolved Debian Mesa RADV 22.3.6 for both architectures. The Dockerfile verifies the upstream DXVK 2.3.1 archive against SHA-256 `c51416d873ced17a6a52d1fb288f887148e9ab7841f25233b6aae24853956d2c`. The launcher overrides `d3d9.dll` in the disposable 32-bit prefix. [Trial setup](README.md), [DXVK release](https://github.com/doitsujin/dxvk/releases/tag/v2.3.1), [32-bit prefix instructions](https://github.com/doitsujin/dxvk/blob/v2.3.1/README.md)

The launcher runs Gamescope with `--backend headless --expose-wayland -w 1024 -h 768 -r 60` and starts a separate **rootful** Xwayland display at `:89`. Here rootful describes the X11 root window. Both X servers run as the regular user. Podman uses `--network none --device=/dev/dri/renderD128 --cpus 3 --memory 1536m`, with only the nested display's X socket mounted. The wrapper expands the fixed startup taps for v5 and preserves native-tick menu selection for v6. [Gamescope options](https://github.com/ValveSoftware/gamescope/blob/master/src/main.cpp), [Podman device mapping](https://docs.podman.io/en/latest/markdown/podman-run.1.html)

The worker speaks the existing line-based trial protocol on stdin and stdout. Send `Q` on its own line to close the container and compositor. The default session deadline is 120 seconds. Set `TOHO_SESSION_SECONDS` for a longer test or `TOHO_X_DISPLAY` to use a different isolated display. Renderer logs and failure captures are written to the run's directory under `$project/runs`. Check `th14_d3d9.log` for `AMD Radeon Graphics (RADV VANGOGH)`.

## Matched short trace

For the GPU case, set the Steam Deck worker command to `ssh -T steamdeck 'exec "$HOME/.local/share/toho-ml/steamdeck-dxvk/steamdeck-dxvk.sh"'`. For the CPU case, call `localhost/toho-trial:native-v5 serve` through rootless Podman with the same CPU and memory limits, and mount the same trial assets and a separate capture directory. Run each case independently.

```sh
python -m scripts.benchmark_trial_workers \
    --workers private-worker.json --out report.json --episodes 2 --decisions 128
```

| Metric | v5 Xvfb + WineD3D | v5 rootful Xwayland + DXVK/RADV | v6 rootful Xwayland + DXVK/RADV |
| --- | ---: | ---: | ---: |
| Decisions/s including one Retry | 6.76 | 11.96 | 12.01 |
| Step p50 / p95 | 128.71 / 210.90 ms | 81.55 / 101.04 ms | 81.65 / 102.55 ms |
| Retry / restarted frame | 2.63 s / 6 | 1.16 s / 6 | 1.13 s / 6 |
| Exact four-tick decisions | 256/256 | 256/256 | 256/256 |
| Eligible horizontal responses | 208/208 | 224/224 | 208/208 |

A synthetic 32-bit Direct3D 9 window returned success from device creation, clear and present on Gamescope's DRI3-capable Xwayland display. TH14 itself repeatedly received a 1×1 swapchain with Gamescope's embedded display. The separate rootful Xwayland display kept its 640×480 game window.

The measured **1.77×** v5 throughput applies to the tested display and renderer combination over one short fixed-action trace per backend. Both cases used the same game build, difficulty, stage and actions. Their initial states differed with launcher timing. A later v5 repeat timed out in the worker protocol after startup. The single v6 verification used the clock-driven menu and both episodes started at stage frame 6. Replay parity, full-version TH14, concurrent Deck workers and training outcomes remain unmeasured in this comparison.
