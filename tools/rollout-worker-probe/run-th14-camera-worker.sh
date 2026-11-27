#!/bin/sh
# Start one isolated original TH14 Normal game and serve clock-locked camera observations.
set -eu
test "$#" = 5 || test "$#" = 6
name=$1
case "$name" in v26th14-*) ;; *) exit 3;; esac
out=$2
assets=$3
clock=$4
image=$5
easy_menu=${6:-}
src=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
test -f "$clock" && test -f "$assets/th14.exe"
mkdir -p "$out"
set -- docker run --init --rm -i --network none --name "$name" --cpus 2.5 --memory 1400m
if test -n "$easy_menu"; then
    case "$easy_menu" in [1-9]|10|11|12) ;; *) exit 3;; esac
    set -- "$@" --env "TH14_EASY_MENU_TAP=$easy_menu"
fi
set -- "$@" --mount "type=bind,source=$assets,target=/input/game,readonly"
set -- "$@" --mount "type=bind,source=$clock,target=/probe/trial-clock.exe,readonly"
set -- "$@" --env TH14_SETTLED_CAPTURE=1 --mount "type=bind,source=$src/run-th14-trial.sh,target=/probe/run-th14-trial.sh,readonly" --entrypoint sh
exec "$@" --mount "type=bind,source=$out,target=/capture" "$image" /probe/run-th14-trial.sh serve
