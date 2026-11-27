#!/bin/sh
set -eu
export DISPLAY=:66
mode=$1
label=$2
seed=$3
weight=$4
hash=$5
case "$mode" in qualify|train|held) ;; engineering) test "${TH12_TICK4_ENGINEERING:-0}" = 1;; *) exit 2;; esac
case "$weight" in ""|*[!a-zA-Z0-9._-]*) exit 2;; esac
if test "$mode" != qualify; then printf '%s  %s\n' "$hash" "/input/$weight" | sha256sum -c >&2; fi
case "$label" in ""|*[!a-zA-Z0-9_.-]*) exit 2;; esac
exec 9>"/capture/$label.lock"
flock -x 9
if test -s "/capture/full/$label/outcome.txt" && grep -q '^game_over ' "/capture/full/$label/outcome.txt"; then
    grep -qxF "$mode $seed $hash $label" "/capture/full/$label/identity.txt"
    cat "/capture/full/$label/outcome.txt"
    exit 0
fi
# The display belongs to exactly one active credit, including reset and evidence capture.
exec 8>/capture/active-credit-v2.lock
flock -x 8
clock_reader=${TH12_CLOCK_READER:-/input/th12-full-actor}
if [ "${TH12_NATIVE_STAGE_SENSOR:-0}" = 1 ]; then credit_start=$("$clock_reader" clock); fi
sh "${TH12_RESET_SCRIPT:-/input/th12-full-reset.sh}" "$label"
cd "/capture/full/$label"
test ! -e decisions.tsv
mkdir evidence
if [ "${TH12_NATIVE_STAGE_SENSOR:-0}" = 1 ]; then printf '%s\n' "$credit_start" > credit-start-clock.txt; fi
sensor_pid=
rank_pid=
engineering_pid=
if [ "${TH12_NATIVE_STAGE_SENSOR:-0}" = 1 ]; then
    test "$(cut -d' ' -f1 /capture/executable.sha256)" = c6d15044394a33d6116c09eac44b33fdec814541c78518e29f60d54f49bdbd3a
    cp /capture/executable.sha256 game-exe-sha256.txt
    sha256sum /input/th12-stage-sensor.exe | cut -d' ' -f1 > stage-sensor-sha256.txt
    "$clock_reader" clock > stage-sensor-clock.txt
    WINEPREFIX=/tmp/th12-wine WINEARCH=win32 WINEDEBUG=-all wine /input/th12-stage-sensor.exe \
        "Z:/capture/full/$label/stage-native.tsv" "Z:/capture/full/$label/stopped" > stage-sensor.log 2>&1 &
    sensor_pid=$!
    for count in $(seq 1 100); do test ! -s stage-native.tsv || break; sleep .03; done
    test -s stage-native.tsv
    "$clock_reader" clock >> stage-sensor-clock.txt
fi
if [ -n "${TH12_DIFFICULTY:-}" ]; then
    case "$TH12_DIFFICULTY" in 2|3) ;; *) exit 2;; esac
    sha256sum /input/th12-difficulty.exe | cut -d' ' -f1 > difficulty-observer-sha256.txt
    WINEPREFIX=/tmp/th12-wine WINEARCH=win32 WINEDEBUG=-all wine /input/th12-difficulty.exe \
        "Z:/capture/full/$label/difficulty-native.tsv" "Z:/capture/full/$label/stopped" > difficulty-sensor.log 2>&1 &
    rank_pid=$!
    for count in $(seq 1 100); do test ! -s difficulty-native.tsv || break; sleep .03; done
    test -s difficulty-native.tsv
fi
printf '%s\n' CLOCK_MONOTONIC > evidence/clock-source.txt
if [ "${TH12_TICK4_ENGINEERING:-0}" = 1 ]; then
    test "$mode" = engineering
    WINEPREFIX=/tmp/th12-wine WINEARCH=win32 WINEDEBUG=-all wine /input/th12-tick4-sensor.exe \
        "Z:/capture/full/$label/engineering-native.tsv" "Z:/capture/full/$label/stopped" \
        "Z:/capture/full/$label/engineering-clock.map" > engineering-sensor.log 2>&1 &
    engineering_pid=$!
    for count in $(seq 1 100); do test ! -s engineering-native.tsv || break; sleep .03; done
    test -s engineering-native.tsv
fi
# Hold a timestamped color record through every stage and the original terminal menu.
(
    i=0
    while [ ! -e stopped ]; do
        file=$(printf '%05d.jpg' "$i")
        t=$("$clock_reader" clock) || break
        scrot --overwrite -q 60 -a 3,22,640,480 "evidence/$file" || break
        printf '%s %s\n' "$t" "$file" >> evidence/clock.txt
        i=$((i+1)); sleep 1.5
    done
) &
camera=$!
trap 'touch stopped; wait "$camera" || true' EXIT
printf '%s %s %s %s\n' "$mode" "$seed" "$hash" "$label" > identity.txt
actor=/input/th12-full-qual
if test "$mode" != qualify; then actor=/input/th12-full-actor; fi
if test "${TH12_ACTOR_KIND:-portable}" = camera; then
    test "$mode" != qualify
    actor=${TH12_CAMERA_ACTOR:-/input/th12-camera-actor}
    if [ -n "${TH12_DIFFICULTY:-}" ]; then actor=/input/th12-rank-camera-actor; fi
    if [ "${TH12_TICK4_ENGINEERING:-0}" = 1 ]; then actor=/input/th12-tick4-camera-actor; fi
fi
rc=0
"$actor" "/input/$weight" "$seed" 120 "$mode" > outcome.txt 2> worker.log || rc=$?
touch stopped
if [ -n "$engineering_pid" ]; then wait "$engineering_pid"; fi
if [ -n "$sensor_pid" ]; then wait "$sensor_pid"; fi
if [ -n "$rank_pid" ]; then wait "$rank_pid"; fi
wait "$camera" || true
trap - EXIT
scrot --overwrite -a 3,22,640,480 endpoint.png
case "$(/input/th12-full-actor status)" in
    score_ranking) cp endpoint.png stage-ranking.png ;;
    gameover|gameover_retry)
        # X is the original menu cancel. Only after the terminal stock clock has stopped.
        xdotool keydown x; sleep .15; xdotool keyup x
        for attempt in 1 2 3 4 5; do
            sleep .2
            if test "$(/input/th12-full-actor status)" = score_ranking; then
                scrot --overwrite -a 3,22,640,480 stage-ranking.png
                break
            fi
        done ;;
esac
cat outcome.txt
flock -u 8
flock -u 9
exit "$rc"
