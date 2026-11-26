#!/bin/sh
set -eu
export DISPLAY=:66 WINEPREFIX=/tmp/th12-wine WINEARCH=win32 WINEDEBUG=-all
export WINEDLLOVERRIDES=mscoree,mshtml= LP_NUM_THREADS=2
export PULSE_SERVER=unix:/tmp/th12-pulse.sock XDG_RUNTIME_DIR=/tmp/runtime
# The original ranking screen has no active life. Relaunch only after explicit recognition.
test "$(/input/th12-full-actor status)" = score_ranking
wineserver -k || true
sleep 1
cd /tmp/th12-game
wine th12.exe >> /capture/full-relaunch.log 2>&1 </dev/null &
configured=0
for attempt in $(seq 1 60); do
  w=$(xdotool search --onlyvisible --name 'Undefined Fantastic Object.*0\.02a' 2>/dev/null | tail -1)
  if test -n "$w"; then
    xdotool windowmove "$w" 3 22
    if test "$(/input/th12-full-actor status)" = title; then exit 0; fi
  elif test "$configured" = 0 && xwininfo -tree -root | grep -qF '("th12.exe" "th12.exe")  208x154+408+316'; then
    xdotool mousemove 427 364 click 1 sleep .25 mousemove 510 449 click 1
    configured=1
  fi
  sleep .5
done
exit 1
