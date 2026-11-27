#!/bin/sh
set -eu
/capture/original-stream
# Policy has stopped and all seven physical keys were released by its worker.
# Menu interaction is recorded only after the first confirmed stock endpoint.
snap(){ scrot -o "/capture/post-credit-$1.png"; }
tap(){ xdotool keydown "$1"; sleep .16; xdotool keyup "$1"; sleep .38; }
sleep .9
snap settled
tap x
snap cancel
tap z
sleep .8
snap confirm
for i in 1 2 3 4 5 6; do
 sleep .6
 snap "$i"
done
