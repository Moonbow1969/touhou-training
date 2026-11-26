#!/bin/sh
# Execute inside a private Wine/Xvfb TH13 official Trial 0.01a container at the in-game pause menu.
# Read-only actor inputs and trained weights are available at /capture/lib/.
set -eu
test "$#" = 3 || { echo 'usage: collect-credit.sh RUN_ID SEED POLICY_FILE' >&2; exit 2; }
tag=$1
seed=$2
policy=$3
case "$tag" in ''|*[!a-zA-Z0-9_-]*) exit 2;; esac
case "$seed" in ''|*[!0-9]*) exit 2;; esac
test -s "$policy"
test "$(cat /capture/executable.sha256 | cut -d' ' -f1)" = 09818929cceea76ff4c65630e649bbccdb85d92b2e28bb4dfe6526749e6f4fb6
mkdir "/capture/$tag"
cd "/capture/$tag"
xdotool keydown r
sleep .18
xdotool keyup r
sleep 3.9
scrot -o initial.png
sha256sum "$policy" > policy.sha256
/capture/lib/actor "$policy" "$seed" 1800 > actor.log &
player=$!
i=0
while kill -0 "$player" 2>/dev/null; do
    scrot -q 75 -o "$(printf 'view-%03d.jpg' "$i")"
    i=$((i+1))
    sleep 4
done
wait "$player"
scrot -o endpoint.png
sleep 2
scrot -o settled.png
