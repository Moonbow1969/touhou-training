#!/bin/sh
set -eu
mkdir -p /capture/neutral-tail
xdotool keydown Escape; sleep .10; xdotool keyup Escape
for n in $(seq 0 145); do
    file=$(printf '/capture/neutral-tail/%04d.jpg' "$n")
    scrot -o -q 82 "$file"
    printf '%s\t%s\n' "$n" "$(date +%s%N)" >> /capture/neutral-tail.tsv
    sleep .8
done
