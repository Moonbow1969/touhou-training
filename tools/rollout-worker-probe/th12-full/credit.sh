#!/bin/sh
set -eu
export DISPLAY=:66
mode=$1
label=$2
seed=$3
weight=$4
hash=$5
case "$mode" in qualify|train|held) ;; *) exit 2;; esac
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
sh /input/th12-full-reset.sh "$label"
cd "/capture/full/$label"
test ! -e decisions.tsv
mkdir evidence
# Hold a timestamped color record through every stage and the original terminal menu.
(
    i=0
    while [ ! -e stopped ]; do
        file=$(printf '%05d.jpg' "$i")
        read t other </proc/uptime
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
    actor=/input/th12-camera-actor
fi
rc=0
"$actor" "/input/$weight" "$seed" 120 "$mode" > outcome.txt 2> worker.log || rc=$?
touch stopped
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
