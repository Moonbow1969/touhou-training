#!/bin/sh
set -eu

if [ "${1:-}" = "--rootful-xwayland" ]; then
    ready=$2
    display=$3
    socket="/tmp/.X11-unix/X${display#:}"
    if [ -e "$socket" ]; then
        printf 'Display already in use: %s\n' "$display" >&2
        exit 1
    fi
    Xwayland "$display" -ac -geometry 1024x768 -nolisten tcp > "$ready.xwayland.log" 2>&1 &
    xwayland_pid=$!
    trap 'kill "$xwayland_pid" 2>/dev/null || true' EXIT
    for attempt in $(seq 1 40); do
        if DISPLAY="$display" xdpyinfo >/dev/null 2>&1; then
            printf '%s\n' "$display" > "$ready"
            wait "$xwayland_pid"
            exit
        fi
        sleep .15
    done
    cat "$ready.xwayland.log" >&2
    exit 1
fi

root=${TOHO_ROOT:-"$HOME/.local/share/toho-ml"}
image=${TOHO_DXVK_IMAGE:-localhost/toho-deck-vulkan-diagnostic:dxvk-2.3.1}
display=${TOHO_X_DISPLAY:-:89}
seconds=${TOHO_SESSION_SECONDS:-120}
source=$(realpath "$0")
mkdir -p "$root/runs"
capture=$(mktemp -d "$root/runs/steamdeck-dxvk.XXXXXXXX")
runtime="$capture/runtime"
mkdir -m 700 "$runtime"
printf 'Steam Deck DXVK logs: %s\n' "$capture" >&2

# Keep the verified trial-clock, asset validation, audio and input setup.
# Swap the Xvfb display and insert the native DLL into the disposable prefix.
podman run --pull=never --rm --network none --entrypoint cat "$image" /probe/run-th14-trial.sh \
    > "$capture/trial-stock.sh"
python3 - "$capture" <<'PY'
from pathlib import Path
import sys

capture = Path(sys.argv[1])
stock = (capture / "trial-stock.sh").read_text()
markers = ("export DISPLAY=:66 ", "timeout 40 wineboot --init")
if any(stock.count(marker) != 1 for marker in markers):
    raise SystemExit("Unsupported stock trial launcher.")
fixed_menu = 'set -- "$@" z; fi'
adaptive_menu = ('xdotool windowfocus --sync "$game"\n'
                 'if test "$mode" = serve; then\n'
                 '    if wine /probe/trial-clock.exe; then exit; fi\n')
if stock.count(fixed_menu) == 1:
    stock = stock.replace(fixed_menu, 'set -- "$@" z z z; fi')
elif fixed_menu in stock or adaptive_menu not in stock:
    raise SystemExit("Expected a native-v5 fixed menu or native-v6 adaptive menu.")
lines = stock.splitlines()
if not any(line.startswith("Xvfb ") for line in lines):
    raise SystemExit("The stock launcher has no Xvfb process to replace.")
kept = [
    line.replace("export DISPLAY=:66 ", "export ").replace('kill "$display_pid" "$audio_pid"', 'kill "$audio_pid"')
    for line in lines if not line.startswith(("Xvfb ", "display_pid="))
]
setup = "\n".join([
    "unset LIBGL_ALWAYS_SOFTWARE",
    "export XDG_RUNTIME_DIR=/tmp/toho-trial-xdg",
    'mkdir -p -m 700 "$XDG_RUNTIME_DIR"',
    "export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.i686.json",
    "export DXVK_LOG_PATH=/capture DXVK_LOG_LEVEL=info",
    "timeout 40 wineboot --init",
    'cp --remove-destination /opt/dxvk/x32/d3d9.dll "$WINEPREFIX/drive_c/windows/system32/d3d9.dll"',
    "export WINEDLLOVERRIDES='mscoree,mshtml=;d3d9=n'",
])
trial = "\n".join(kept) + "\n"
trial = trial.replace("timeout 40 wineboot --init", setup)
(capture / "trial-dxvk.sh").write_text(trial)
PY

env -u DISPLAY -u WAYLAND_DISPLAY XDG_RUNTIME_DIR="$runtime" timeout -k 2 "$seconds" \
    gamescope --backend headless --expose-wayland -w 1024 -h 768 -r 60 -- \
    "$source" --rootful-xwayland "$capture/display" "$display" > "$capture/gamescope.log" 2>&1 &
gamescope_pid=$!
cleanup() {
    kill "$gamescope_pid" 2>/dev/null || true
    wait "$gamescope_pid" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
for attempt in $(seq 1 60); do
    if [ -s "$capture/display" ]; then break; fi
    sleep .2
done
if [ ! -s "$capture/display" ]; then
    tail -20 "$capture/gamescope.log" >&2
    exit 1
fi
display=$(cat "$capture/display")
socket="/tmp/.X11-unix/X${display#:}"
test -S "$socket"
podman run --pull=never --init --rm -i --network none --cpus 3 --memory 1536m \
    --device=/dev/dri/renderD128 --env "DISPLAY=$display" -v "$socket:$socket:rw" \
    -v "$root/trial:/input/game:ro" -v "$capture:/capture" \
    -v "$capture/trial-dxvk.sh:/tmp/trial-dxvk.sh:ro" \
    --entrypoint sh "$image" /tmp/trial-dxvk.sh serve
