#!/bin/sh
set -eu
export DISPLAY=:73 WINEDEBUG=-all LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe LP_NUM_THREADS=1
export WINEDLLOVERRIDES='mscoree,mshtml=;d3d8=b' PULSE_SERVER=unix:/tmp/native-pulse
mkdir -m 700 /tmp/game
cp /input/th07.exe /input/th07.dat /tmp/game/
cd /tmp/game
sha256sum th07.exe th07.dat /tool/thprac.exe >/capture/initial-hashes.txt
find . -type f | sort >/capture/initial-files.txt
Xvfb "$DISPLAY" -screen 0 800x600x24 -nolisten tcp >/capture/xvfb.log 2>&1 &
pulseaudio --daemonize=no --exit-idle-time=-1 --use-pid-file=no -n --load="module-native-protocol-unix socket=/tmp/native-pulse auth-anonymous=1" --load=module-null-sink >/capture/audio.log 2>&1 &
for i in $(seq 1 60); do
    if xdpyinfo >/dev/null 2>&1 && pactl list short sinks >/dev/null 2>&1; then break; fi
    sleep .2
done
wineboot --init >/capture/boot.log 2>&1
wine reg add 'HKCU\Software\Wine\Drivers' /v Audio /t REG_SZ /d pulse /f
wine explorer /desktop=practice,800x600 th07.exe >/capture/game.log 2>&1 &
for i in $(seq 1 350); do
    window=$(xdotool search --onlyvisible --class '^th07.exe$' 2>/dev/null | while read -r id; do
        case "$(xdotool getwindowgeometry "$id")" in *640x480*) echo "$id";; esac
    done | tail -1)
    if [ -n "$window" ] && xdotool windowfocus "$window" 2>/dev/null; then break; fi
    window=""
    sleep .2
done
test -n "$window"
echo "$window" >/capture/window
pid=$(xdotool getwindowpid "$window")
grep -qx th07.exe "/proc/$pid/comm"
wine winedbg --command 'info proc' >/capture/wine-procs.txt 2>&1
hex=$(awk '/\x27th07\.exe\x27/{print $1}' /capture/wine-procs.txt)
test "$(echo "$hex" | wc -w)" -eq 1
wp=$(printf '%d' "0x$hex")
printf 'linux=%s\twine=%s\twindow=%s\n' "$pid" "$wp" "$window" >/capture/target-only.tsv
wine /tool/thprac.exe --attach "$wp" >/capture/thprac.log 2>&1 &
for i in $(seq 1 100); do
    if grep -F '/tool/thprac.exe' "/proc/$pid/maps" >/capture/loaded.txt; then break; fi
    sleep .1
done
test -s /capture/loaded.txt
touch /capture/ready
wait
