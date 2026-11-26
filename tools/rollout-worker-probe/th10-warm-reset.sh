#!/bin/sh
# One private lane. Reset the game process without restarting Wine, Xvfb or audio.
set -eu
game=$1
seed=$2
cd /capture
date +%s%3N >cold-open.ms
i686-w64-mingw32-gcc-posix -O2 /tools/th10-full-stock.c -o /tmp/th10-stock.exe
check_open() {
    awk -F, 'NR==2{ok=($2+0==1 && $3+0==1 && $4+0==0 && $5+0==0 && $6+0==2 && $9+0==0 && $10+0==0)} END{exit !ok}' opening.csv
    wine /tmp/th10-stock.exe --endpoint >opening-inventory.csv
    awk -F, 'NR==1{ok=($2+0==1 && $3+0==2 && $4+0==0)} END{exit !ok}' opening-inventory.csv
}
for run in 0 1 2; do
    mkdir "cycle-$run"
    if test "$run" -ge 1; then
        date +%s%3N >reset-start.ms
        # The container uses an isolated PID namespace, virtual desktop and prefix.
        wine taskkill /F /IM th10tr.exe >"cycle-$run/restart.log" 2>&1
        episode="/tmp/th10-warm-$run";mkdir "$episode"
        for file in th10tr.exe th10tr.dat thbgm_tr.dat; do ln "/tmp/th10-game/$file" "$episode/$file";done
        game_path=$(winepath -w "$episode/th10tr.exe")
        (cd "$episode" && wine explorer /desktop=MoonbowTH10,640x480 "$game_path") >"cycle-$run/title.log" 2>&1 &
        game=
        for poll in $(seq 1 100); do
            for id in $(xdotool search --onlyvisible --class th10tr.exe 2>/dev/null || true); do
                if xdotool getwindowgeometry "$id" 2>/dev/null | grep -q 'Geometry: 199x88'; then
                    xdotool mousemove --window "$id" 68 64 click 1
                fi
            done
            game=$(xdotool search --onlyvisible --name 'Mountain of Faith' 2>/dev/null | tail -1)
            test -z "$game" || break
            sleep .2
        done
        if test -z "$game"; then scrot -a 0,0,640,480 "cycle-$run/no-title.png"; exit 21; fi
        xdotool windowfocus --sync "$game"
        active=
        for poll in $(seq 1 40); do
            # Wine may publish the title window before the modal startup prompt has appeared.
            for id in $(xdotool search --onlyvisible --class th10tr.exe 2>/dev/null || true); do
                if xdotool getwindowgeometry "$id" 2>/dev/null | grep -q 'Geometry: 199x88'; then
                    xdotool mousemove --window "$id" 68 64 click 1
                fi
            done
            game=$(xdotool search --onlyvisible --name 'Mountain of Faith' 2>/dev/null | tail -1)
            if test -n "$game"; then xdotool windowfocus --sync "$game";fi
            xdotool keydown z; sleep .17;xdotool keyup z;sleep .55
            if wine /tmp/th10-observe.exe --check >opening.csv && check_open; then active=1;break;fi
            case "$poll" in 4|12|24|40) scrot -a 0,0,640,480 "cycle-$run/confirm-$poll.png";;esac
        done
        if test -z "$active"; then xwininfo -root -tree >"cycle-$run/windows.txt"; exit 22; fi
        date +%s%3N >first-play.ms
    else
        cp cold-open.ms first-play.ms
        cp cold-start.ms reset-start.ms
    fi
    check_open
    scrot -a 0,0,640,480 "cycle-$run/entry.png"
    xdotool windowfocus --sync "$game"
    wine /tmp/th10-stock.exe >native.csv & observer=$!
    /tools/full-actor /input/actor.bin "$((seed+run))" 120
    touch actor-closed
    wait "$observer"
    test "$(tr -d '\r\n' <native-stop)" = native_final_life_lost
    sleep 1.7
    wine /tmp/th10-stock.exe --endpoint >settled-inventory.csv
    scrot -a 0,0,640,480 "cycle-$run/endpoint.png"
    mv native.csv native-stop opening.csv opening-inventory.csv settled-inventory.csv first-play.ms reset-start.ms actor-closed features.f32 frame.ns pixels.u8 action.csv "cycle-$run/"
done
