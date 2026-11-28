#!/bin/sh
# Run inside the private original Wine/Xvfb image. Mount a user-supplied official
# trial at /game, an original replay at /source.rpy, this directory at /reader,
# and an empty host capture directory at /capture.
set -eu
cd /capture
test ! -e game-start
test "$(df -Pk . | tail -1 | awk '{print $4}')" -gt 8000000
printf '%s  %s\n' \
  3a6b4a0de24f916e669cf642ffd4f7c22ada1d319d9ce8ca8494fe96a2f5f5c1 /game/th08tr.exe \
  ec9ad3091e437ba41bd453ed2637093e4bb02906b7f8dc9862f287a5e59b6388 /game/th08tr.dat \
  | sha256sum -c -
sha256sum /source.rpy /game/th08tr.exe /game/th08tr.dat /reader/camera.c > sources.sha256
date -Ins > game-start
mkdir -p /tmp/eiya/replay
cp /game/th08tr.exe /game/th08tr.dat /tmp/eiya/
if [ -f /game/th08.cfg ]; then cp /game/th08.cfg /tmp/eiya/; fi
cp /source.rpy /tmp/eiya/replay/th8_01.rpy
i686-w64-mingw32-gcc-posix -std=c11 -O2 -Wl,--no-insert-timestamp \
  /reader/camera.c -o /capture/reader.exe -lgdi32 -luser32
sha256sum /capture/reader.exe >> /capture/sources.sha256
export DISPLAY=:81 WINEDLLOVERRIDES=mscoree,mshtml= LP_NUM_THREADS=1
export PULSE_SERVER=unix:/tmp/native-pulse LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe
Xvfb :81 -screen 0 800x600x24 -nolisten tcp >/tmp/x.log 2>&1 &
pulseaudio --daemonize=no --exit-idle-time=-1 --use-pid-file=no -n \
  --load="module-native-protocol-unix socket=/tmp/native-pulse auth-anonymous=1" \
  --load=module-null-sink >/tmp/p.log 2>&1 &
for n in $(seq 1 40); do xdpyinfo >/dev/null 2>&1 && pactl list short sinks >/dev/null 2>&1 && break; sleep .2; done
timeout 45 wineboot --init >/capture/wine-init.log 2>&1
cd /tmp/eiya
wine explorer /desktop=study,800x600 th08tr.exe >/capture/game.log 2>&1 &
window=""
for n in $(seq 1 200); do
  window=$(xdotool search --onlyvisible --class '^th08tr.exe$' 2>/dev/null |
    while read -r w; do xdotool getwindowgeometry "$w" | grep -q '640x480' && echo "$w"; done | tail -1)
  if [ -n "$window" ] && xdotool windowfocus "$window"; then break; fi
  sleep .3
done
test -n "$window"
tap(){ xdotool keydown "$1"; sleep .14; xdotool keyup "$1"; sleep .55; }
snap(){ scrot -o "/capture/$1.png"; }
sleep 10
snap title; tap Down; snap replay-menu; tap z; sleep .9; snap pilot-rank
tap z; sleep .7; snap stage-selector
tap z; sleep .7; snap normal-speed
stages="1 2 3"
last_stage=3
if [ "${TH08_FIRST_STAGE_ONLY:-0}" = 1 ]; then
  stages=1
  last_stage=1
fi
for stage in $stages; do
  set -- "$stage" 10000 "stage-$stage"
  if [ -n "${TH08_MOTION_PATCH_LIMIT:-}" ]; then
    set -- "$@" "$TH08_MOTION_PATCH_LIMIT" "${TH08_MOTION_PATCH_STRIDE:-4}"
  fi
  wine explorer /desktop=study,800x600 Z:/capture/reader.exe "$@" >/dev/null 2>&1 &
done
for n in $(seq 1 150); do
  ready=1
  for stage in $stages; do grep -qs '^pid' "/capture/stage-$stage/observer.tsv" || ready=0; done
  if [ "$ready" = 1 ]; then break; fi
  sleep .2
done
for stage in $stages; do grep -q '^pid' "/capture/stage-$stage/observer.tsv"; done
tap z; sleep 3; snap physical-first-play
for n in $(seq 1 380); do
  if grep -q '^sealed' "/capture/stage-$last_stage/observer.tsv"; then break; fi
  sleep 5
done
for stage in $stages; do grep -q '^sealed' "/capture/stage-$stage/observer.tsv"; done
snap replay-finish
date -Ins > /capture/recording-complete
