#!/bin/sh
set -eu
export DISPLAY=:73 WINEDEBUG=err+all WINEDLLOVERRIDES=mscoree,mshtml=
export LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=1
export PULSE_SERVER=unix:/tmp/native-pulse
mkdir /tmp/th07
cp /input/th07.exe /input/th07tr.dat /tmp/th07/
cd /tmp/th07
sha256sum th07.exe th07tr.dat > /capture/build.sha256
Xvfb "$DISPLAY" -screen 0 800x600x24 -nolisten tcp > /capture/xvfb.log 2>&1 &
pulseaudio --daemonize=no --exit-idle-time=-1 --use-pid-file=no -n --load="module-native-protocol-unix socket=/tmp/native-pulse auth-anonymous=1" --load=module-null-sink > /capture/audio.log 2>&1 &
for i in $(seq 1 50); do
    if xdpyinfo >/dev/null 2>&1 && pactl list short sinks >/dev/null 2>&1; then break; fi
    sleep .2
done
wineboot --init > /capture/boot.log 2>&1
wine reg add 'HKCU\Software\Wine\Drivers' /v Audio /t REG_SZ /d pulse /f
wine explorer /desktop=study,800x600 th07.exe >/capture/game.log 2>&1 &
for i in $(seq 1 100); do
    window=$(xdotool search --class '^th07.exe$' 2>/dev/null | while read -r id; do
        case "$(xdotool getwindowgeometry "$id")" in *640x480*) echo "$id";; esac
    done | tail -1)
    if [ -n "$window" ]; then
        echo "$window" > /capture/window
        xdotool windowfocus "$window"
        break
    fi
    sleep .2
done
wait
