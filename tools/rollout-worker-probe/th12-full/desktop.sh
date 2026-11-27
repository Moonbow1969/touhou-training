#!/bin/sh
set -eu

export DISPLAY=:66 WINEPREFIX=/tmp/th12-wine WINEARCH=win32
export WINEDLLOVERRIDES=mscoree,mshtml= LP_NUM_THREADS=2
export PULSE_SERVER=unix:/tmp/th12-pulse.sock
export WINEDEBUG=-all
mkdir -p /tmp/runtime
chmod 700 /tmp/runtime
export XDG_RUNTIME_DIR=/tmp/runtime

Xvfb "$DISPLAY" -screen 0 1024x768x24 -nolisten tcp > /capture/xvfb.log 2>&1 &
xpid=$!
pulseaudio --daemonize=no --exit-idle-time=-1 --use-pid-file=no -n \
    --load="module-native-protocol-unix socket=/tmp/th12-pulse.sock auth-anonymous=1" \
    --load=module-null-sink > /capture/pulse.log 2>&1 &
apid=$!
trap 'wineserver -k 2>/dev/null || true; kill "$xpid" "$apid" 2>/dev/null || true' EXIT
for attempt in $(seq 1 30); do
    if xdpyinfo >/dev/null 2>&1 && pactl list short sinks >/dev/null 2>&1; then break; fi
    sleep .25
done
xdpyinfo >/dev/null
pactl list short sinks > /capture/audio-sink.txt
timeout 45 wineboot --init > /capture/wineboot.log 2>&1
wine reg add 'HKCU\Software\Wine\Drivers' /v Audio /t REG_SZ /d pulse /f > /capture/registry.log 2>&1

case "${1:-}" in
    setup)
        sha256sum /input/th12tr002a_setup.exe > /capture/installer.sha256
        wine /input/th12tr002a_setup.exe > /capture/installer.log 2>&1 &
        printf '%s\n' "$!" > /capture/launcher.pid
        ;;
    play)
        mkdir -p /tmp/th12-game
        cp -a /game/. /tmp/th12-game/
        cd /tmp/th12-game
        game=$(find . -maxdepth 1 -name 'th12.exe' -o -name 'th12tr.exe' | head -1)
        test -n "$game"
        sha256sum "$game" > /capture/executable.sha256
        wine "$game" > /capture/game.log 2>&1 &
        printf '%s\n' "$!" > /capture/launcher.pid
        ;;
    *) exit 2 ;;
esac
# Keep this bounded test desktop alive for private control and snapshots.
sleep "${TH12_DESKTOP_LIFETIME:-14400}"
