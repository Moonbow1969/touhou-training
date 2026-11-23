#!/bin/sh
set -eu
export DISPLAY=:66 WINEDLLOVERRIDES=mscoree,mshtml=
Xvfb "$DISPLAY" -screen 0 1024x768x24 -nolisten tcp &
display_pid=$!
trap 'wineserver -k 2>/dev/null || true; kill "$display_pid" 2>/dev/null || true' EXIT
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    if xdotool getdisplaygeometry >/dev/null 2>&1; then break; fi
    sleep .2
done
xdotool getdisplaygeometry >/dev/null
timeout 30 wineboot --init
wine probe.exe &
probe_pid=$!
window=$(timeout 12 xdotool search --sync --onlyvisible --name '^TH14 environment probe$')
xdotool windowfocus --sync "$window"
xdotool keydown z
sleep .15
xdotool keyup z
wait "$probe_pid"
