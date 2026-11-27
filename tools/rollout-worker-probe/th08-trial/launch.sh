#!/bin/sh
set -eu
mkdir -p /tmp/eiya
cp /input/game/* /tmp/eiya/
cd /tmp/eiya
sha256sum th08tr.exe th08tr.dat > /capture/build.sha256
export DISPLAY=:81 WINEDLLOVERRIDES=mscoree,mshtml= LP_NUM_THREADS=1 PULSE_SERVER=unix:/tmp/native-pulse
Xvfb "$DISPLAY" -screen 0 800x600x24 -nolisten tcp > /tmp/display.log 2>&1 &
pulseaudio --daemonize=no --exit-idle-time=-1 --use-pid-file=no -n --load="module-native-protocol-unix socket=/tmp/native-pulse auth-anonymous=1" --load=module-null-sink >/tmp/pulse.log 2>&1 &
for n in $(seq 1 35); do if xdpyinfo >/dev/null 2>&1 && pactl list short sinks >/dev/null 2>&1; then break; fi; sleep .2; done
timeout 40 wineboot --init > /capture/wine-init.log 2>&1
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
WINEDEBUG=err+all wine explorer /desktop=study,800x600 th08tr.exe > /capture/game.log 2>&1 &
for n in $(seq 1 50); do
 game=$(xdotool search --class '^th08tr.exe$' 2>/dev/null | while read -r w; do sz=$(xdotool getwindowgeometry "$w" 2>/dev/null); case "$sz" in *640x480*) echo "$w";; esac; done | tail -1)
 [ -z "$game" ] || break
 sleep .3
done
test -n "$game"
xdotool windowfocus "$game"
sleep 10
tap(){ xdotool keydown "$1"; sleep .13; xdotool keyup "$1"; sleep .45; }
snap(){ scrot -o "/capture/$1.png"; }
snap title
tap z; sleep 1; snap default-rank
tap Down
case "$1" in Hard) : ;; Lunatic) tap Down ;; *) exit 3 ;; esac
sleep .8; snap chosen-rank
tap z; sleep 1.8; snap team
tap z; sleep 1
exec /input/stream
