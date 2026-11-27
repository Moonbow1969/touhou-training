#!/bin/sh
set -eu
export DISPLAY=:66 WINEPREFIX=/tmp/th17-wine WINEARCH=win32 WINEDEBUG=-all
export WINEDLLOVERRIDES=mscoree,mshtml= LP_NUM_THREADS=2
export PULSE_SERVER=unix:/tmp/th17-null-pulse.sock XDG_RUNTIME_DIR=/tmp/th17-xdg
mkdir -m 700 -p "$XDG_RUNTIME_DIR" /tmp/th17-game
Xvfb :66 -screen 0 1024x768x24 -nolisten tcp > /capture/xvfb.log 2>&1 &
xpid=$!
pulseaudio --daemonize=no --exit-idle-time=-1 --use-pid-file=no -n \
    --load="module-native-protocol-unix socket=/tmp/th17-null-pulse.sock auth-anonymous=1" \
    --load=module-null-sink > /capture/pulse.log 2>&1 &
apid=$!
trap 'wineserver -k 2>/dev/null || true; kill "$xpid" "$apid" 2>/dev/null || true' EXIT
for i in $(seq 1 50); do
    if xdpyinfo >/dev/null 2>&1 && pactl list short sinks >/dev/null 2>&1; then break; fi
    sleep .2
done
xdpyinfo >/dev/null
pactl list short sinks > /capture/null-sink.txt
timeout 45 wineboot --init > /capture/wineboot.log 2>&1
wine reg add 'HKCU\Software\Wine\Drivers' /v Audio /t REG_SZ /d pulse /f > /capture/wine-registry.log 2>&1
for file in th17.exe th17tr.dat thbgm_tr.dat; do cp "/game/$file" /tmp/th17-game/; done
cd /tmp/th17-game
sha256sum th17.exe > /capture/executable.sha256
wine th17.exe > /capture/game.log 2>&1 &
pid=$!
wait "$pid"
