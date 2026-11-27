#!/bin/sh
set -eu
export DISPLAY=:66 WINEPREFIX=/tmp/th11-wine WINEDEBUG=-all WINEDLLOVERRIDES=mscoree,mshtml= LP_NUM_THREADS=2
export PULSE_SERVER=unix:/tmp/th11-null-pulse.sock
mkdir -p /tmp/th11-private
cd /tmp/th11-private
for item in /input/*.exe /input/*.dat; do test -f "$item" || continue; ln -s "$item" .; done
Xvfb :66 -screen 0 1024x768x24 -nolisten tcp >/capture/display.log 2>&1 &
pulseaudio --daemonize=no --exit-idle-time=-1 --use-pid-file=no -n --load="module-native-protocol-unix socket=/tmp/th11-null-pulse.sock auth-anonymous=1" --load=module-null-sink >/capture/pulse.log 2>&1 &
for i in $(seq 1 40); do if xdpyinfo >/dev/null 2>&1 && pactl list short sinks >/dev/null 2>&1; then break; fi; sleep .3; done
timeout 50 wineboot --init
wine reg add 'HKCU\Software\Wine\Drivers' /v Audio /t REG_SZ /d pulse /f >/capture/winereg.log
game=$(find /input -maxdepth 1 -iname 'th11*.exe' | head -1)
test -n "$game"
sha256sum "$game" >/capture/actual-executable.sha256
wine "./$(basename "$game")" >/capture/game.log 2>&1 &
pid=$!
echo "$pid" >/capture/linux-game-pid
wait "$pid"
