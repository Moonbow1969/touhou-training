#!/bin/sh
set -eu
case "${1:-}" in Left|Right|serve) mode=$1 ;; *) exit 2 ;; esac

mkdir -p /tmp/game
cd /tmp/game
cp /input/game/th14.exe /input/game/th14tr.dat /input/game/thbgm_tr.dat .
printf '%s  %s\n' '6c62bff1e6be644086c5811c36eb938818f14012f94ad5fce784ca7889c66d48' th14.exe | sha256sum -c >&2

export DISPLAY=:66 WINEDLLOVERRIDES=mscoree,mshtml= LP_NUM_THREADS=2
export PULSE_SERVER=unix:/tmp/toho-pulse.sock
Xvfb "$DISPLAY" -screen 0 1024x768x24 -nolisten tcp &
display_pid=$!
pulseaudio --daemonize=no --exit-idle-time=-1 --use-pid-file=no -n \
    --load="module-native-protocol-unix socket=/tmp/toho-pulse.sock auth-anonymous=1" \
    --load=module-null-sink > /capture/pulse.log 2>&1 &
audio_pid=$!
trap 'wineserver -k 2>/dev/null || true; kill "$display_pid" "$audio_pid" 2>/dev/null || true' EXIT
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if xdpyinfo >/dev/null 2>&1 && pactl list short sinks >/dev/null 2>&1; then break; fi
    sleep .3
done
xdpyinfo >/dev/null
pactl list short sinks >&2
timeout 40 wineboot --init
wine reg add 'HKCU\Software\Wine\Drivers' /v Audio /t REG_SZ /d pulse /f >&2
wine ./th14.exe > /capture/game-stdout.log 2>&1 &

dialog=$(timeout 20 xdotool search --sync --onlyvisible --name . | tail -1)
test -n "$dialog"
xdotool mousemove --window "$dialog" 20 34 click 1 mousemove --window "$dialog" 101 144 click 1
game=$(timeout 20 xdotool search --sync --onlyvisible --name 'Double Dealing Character. ver 0.01b' | tail -1)
test -n "$game"
xdotool getwindowname "$game" >&2
xdotool windowfocus --sync "$game"
if test "$mode" = serve; then
    if wine /probe/trial-clock.exe; then exit; fi
    scrot --overwrite /capture/failed.png
    exit 1
fi
sleep 5
for key in z z z z; do
    xdotool keydown "$key"
    sleep .075
    xdotool keyup "$key"
    sleep 1.3
done
sleep 1.5
xdotool keydown "$mode"
sleep 1.3
xdotool keyup "$mode"
scrot --overwrite /capture/game.png
printf 'Captured the trial with %s input.\n' "$mode"
