#!/bin/sh
set -eu
mode=${1:-}
wire=0
difficulty=1
case "$mode" in
    serve) wire=1 ;;
    serve-hard) wire=1; difficulty=2 ;;
esac
if test "$wire" -eq 1; then
    exec 3>&1 1>&2
fi
export DISPLAY=:59 WINEPREFIX=/tmp/toho-th10-proof WINEDEBUG=-all WINEDLLOVERRIDES=mscoree,mshtml=
export LANG=C.UTF-8 LC_ALL=C.UTF-8 LP_NUM_THREADS=2 PULSE_SERVER=unix:/tmp/toho-th10-pulse.sock
expected=5430fac601d645798ccf794362ec7d4cd753814f09d160247dc628b7c6943fd8

mkdir /tmp/th10-game
cp /input/game/th10tr.exe /input/game/th10tr.dat /input/game/thbgm_tr.dat /tmp/th10-game/
cd /tmp/th10-game
printf '%s  %s\n' "$expected" th10tr.exe | sha256sum -c
i686-w64-mingw32-gcc-posix -Wall -Wextra -Werror -O2 -Wl,--no-insert-timestamp \
    /tools/th10-trial-observe.c -o /tmp/th10-observe.exe -luser32
sha256sum /tmp/th10-observe.exe > /capture/probe-sha256.txt
if test "$wire" -eq 1; then
    i686-w64-mingw32-gcc-posix -Wall -Wextra -Werror -O2 -Wl,--no-insert-timestamp \
        /tools/th10-trial-step.c -o /tmp/th10-step.exe -luser32
    sha256sum /tmp/th10-step.exe >> /capture/probe-sha256.txt
fi

Xvfb "$DISPLAY" -screen 0 1024x768x24 -nolisten tcp >/capture/display.log 2>&1 & display_pid=$!
pulseaudio --daemonize=no --exit-idle-time=-1 --use-pid-file=no -n \
    --load='module-native-protocol-unix socket=/tmp/toho-th10-pulse.sock auth-anonymous=1' \
    --load=module-null-sink >/capture/pulse.log 2>&1 & audio_pid=$!
cleanup() {
    wineserver -k 2>/dev/null || true
    kill "$display_pid" "$audio_pid" 2>/dev/null || true
}
trap cleanup EXIT

timeout 45 wineboot --init
wine reg add 'HKCU\Software\Wine\Drivers' /v Audio /t REG_SZ /d pulse /f >/capture/audio.log
wine explorer /desktop=MoonbowTH10,640x480 'Z:\tmp\th10-game\th10tr.exe' >/capture/game.log 2>&1 &
dialog=
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16; do
    for id in $(xdotool search --onlyvisible --class th10tr.exe 2>/dev/null || true); do
        if xdotool getwindowgeometry "$id" | grep -q 'Geometry: 199x88'; then
            dialog=$id
            break
        fi
    done
    test -z "$dialog" || break
    sleep .5
done
scrot /capture/00-startup.png
test -n "$dialog"
# The isolated, verified first-launch game prompt has a selected Yes button at this offset.
xdotool mousemove --window "$dialog" 68 64 click 1

game=
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    game=$(xdotool search --onlyvisible --name 'Mountain of Faith' 2>/dev/null | tail -1)
    test -z "$game" || break
    sleep .5
done
scrot /capture/01-menu.png
xwininfo -root -tree > /capture/windows.txt
test -n "$game"
xdotool windowfocus --sync "$game"
xdotool getwindowname "$game" | tee /capture/window-title.txt
sleep 7
if test "$difficulty" -eq 2; then
    rank=
    for index in 1 2 3 4 5 6; do
        scrot -o -a 28,18,180,48 /tmp/th10-rank.ppm
        if perl /tools/th10-rank-screen.pl /tmp/th10-rank.ppm; then
            rank=1
            break
        fi
        xdotool keydown z; sleep .22; xdotool keyup z
        sleep 1.3
    done
    test -n "$rank"
    xdotool keydown Down; sleep .18; xdotool keyup Down
    sleep .65
    scrot /capture/hard-rank.png
fi
active=
for index in 1 2 3 4 5 6 7 8; do
    xdotool keydown z; sleep .22; xdotool keyup z
    sleep 1.1
    scrot "/capture/02-confirm-$index.png"
    if wine /tmp/th10-observe.exe --check > /capture/opening.csv; then
        awk -F, -v expected="$difficulty" 'NR == 2 { found = ($3 == expected) } END { exit !found }' /capture/opening.csv
        active=1
        break
    fi
done
test -n "$active"

if test "$wire" -eq 1; then
    sleep 1.5
    xdotool windowfocus --sync "$game"
    if test "$difficulty" -eq 2; then scrot /capture/hard-stage.png; fi
    /bin/bash /tools/serve-th10-trial.sh "$game" "$difficulty" >&3
    exit
fi

if test -s /input/actions.csv; then
    mkdir /capture/pixels
    capture_pixel() {
        read -r uptime other </proc/uptime
        ns=${uptime%.*}${uptime#*.}0000000
        printf '%s,%s,%s\n' "$1" "$ns" "$(date +%s%3N)" >> /capture/pixel-timing.csv
        scrot -a 0,0,640,480 "/capture/pixels/$1.png"
        test "$(wc -c < "/capture/pixels/$1.png")" -le 2097152
    }
    wine /tmp/th10-observe.exe >/capture/native.csv & sampler=$!
    sleep .4
    xdotool windowfocus --sync "$game"
    xdotool keydown z
    capture_pixel 0
    expected_index=1
    while IFS=, read -r index dx dy focus; do
        test "$index" -eq "$expected_index" && test "$index" -le 32
        buttons=
        case "$dx" in -1) buttons=Left ;; 0) ;; 1) buttons=Right ;; *) exit 3 ;; esac
        case "$dy" in -1) buttons="$buttons Up" ;; 0) ;; 1) buttons="$buttons Down" ;; *) exit 3 ;; esac
        case "$focus" in 0) ;; 1) buttons="$buttons Shift" ;; *) exit 3 ;; esac
        # Every token in this list is a fixed X key selected by the cases above.
        test -z "$buttons" || xdotool keydown $buttons
        printf '%s,%s,%s,%s,%s,%s\n' "$index" "$dx" "$dy" 1 "$focus" "$(date +%s%3N)" >> /capture/actions.csv
        sleep .085
        capture_pixel "$index"
        test -z "$buttons" || xdotool keyup $buttons
        expected_index=$((expected_index + 1))
    done </input/actions.csv
    xdotool keyup z
    test "$expected_index" -eq 33
    wait "$sampler"
    printf 'captured 33 pixels and 32 executed actions\n'
    exit 0
fi

mark() { printf '%s,%s\n' "$(date +%s%3N)" "$1" >> /capture/inputs.csv; }
wine /tmp/th10-observe.exe >/capture/native.csv & sampler=$!
sleep .5
xdotool windowfocus --sync "$game"
mark shoot_on
xdotool keydown z
sleep 2
scrot /capture/03-before.png
mark right_on
xdotool keydown Right; sleep .42; xdotool keyup Right
mark right_off
scrot /capture/04-right.png
mark left_on
xdotool keydown Left; sleep .74; xdotool keyup Left
mark left_off
scrot /capture/05-left.png
mark recenter_on
xdotool keydown Right; sleep .26; xdotool keyup Right
mark recenter_off
sleep 8
scrot /capture/06-shoot.png
mark shoot_off
xdotool keyup z
wait "$sampler"
printf 'captured %s native rows\n' "$(($(wc -l < /capture/native.csv) - 1))"
