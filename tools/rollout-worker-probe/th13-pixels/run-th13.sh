#!/bin/sh
set -eu
export DISPLAY=:66 WINEDLLOVERRIDES=mscoree,mshtml= LP_NUM_THREADS=2
export PULSE_SERVER=unix:/tmp/th13-pulse.sock
mkdir -p /tmp/th13-runtime
cd /tmp/th13-runtime
ln -s /input/th13tr.dat /input/thbgm_tr.dat .
cp /input/th13.exe .
sha256sum th13.exe > /capture/executable.sha256
Xvfb :66 -screen 0 1024x768x24 -nolisten tcp >/capture/display.log 2>&1 &
pulseaudio --daemonize=no --exit-idle-time=-1 --use-pid-file=no -n --load="module-native-protocol-unix socket=/tmp/th13-pulse.sock auth-anonymous=1" --load=module-null-sink >/capture/pulse.log 2>&1 &
for i in $(seq 1 20); do if xdpyinfo >/dev/null 2>&1 && pactl list short sinks >/dev/null 2>&1; then break; fi; sleep .3; done
timeout 50 wineboot --init
wine reg add 'HKCU\Software\Wine\Drivers' /v Audio /t REG_SZ /d pulse /f >/capture/winereg.log
wine ./th13.exe >/capture/game.log 2>&1 &
game_pid=$!
echo "$game_pid" >/capture/linux-game-pid
wait "$game_pid"