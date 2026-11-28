#!/bin/sh
set -eu
mkdir -p /capture/boss-frames
i=0
while [ ! -e /capture/stop-boss-pictures ]; do
    # The game display lives in the upper left 640 by 480 region of its private X server.
    prefix=$(printf '%05d' "$i")
    stamp=$(cut -d ' ' -f 1 /proc/uptime)
    scrot -q 82 -a 0,0,440,480 -o "/capture/boss-frames/$prefix.jpg"
    printf '%s,%s\n' "$prefix" "$stamp" >> /capture/boss-frame-times.csv
    i=$((i+1))
    sleep 2
done
