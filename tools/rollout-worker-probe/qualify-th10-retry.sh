#!/bin/bash
# Substitute for the serving hook on a dedicated private qualification worker.
set -eu
window=$1
xdotool windowfocus --sync "$window"
tap() { xdotool keydown "$1"; sleep .15; xdotool keyup "$1"; }
trap 'xdotool keyup Left Right z Escape r || true' EXIT
now() { date +%s%3N; }
printf 'trial,reset_start_ms,reset_read_ms\n' >/capture/reset-times.csv
for trial in 1 2 3; do
    xdotool keydown z
    sleep 3
    xdotool keydown Right; sleep .50; xdotool keyup Right
    sleep 1.5; xdotool keyup z
    wine /tmp/th10-observe.exe --check >"/capture/prior-$trial.csv"
    scrot --overwrite "/capture/prior-$trial.png"
    start=$(now)
    tap Escape; sleep .65
    scrot --overwrite "/capture/pause-$trial.png"
    tap r; sleep 2
    scrot --overwrite "/capture/retry-$trial.png"
    if ! wine /tmp/th10-observe.exe --check >"/capture/reset-$trial.csv"; then
        sleep 1.8
        wine /tmp/th10-observe.exe --check >"/capture/reset-$trial.csv" || exit 5
    fi
    done_ms=$(now)
    printf '%s,%s,%s\n' "$trial" "$start" "$done_ms" >>/capture/reset-times.csv
    # Check reset state before spending another full trial on the same menu path.
    awk -F, 'NR==2 {ok=($2==1 && $3==1 && $6==2 && $7>=-.1 && $7<=.1 && $8>=390 && $8<=410 && $9==0)} END {exit !ok}' "/capture/reset-$trial.csv" || exit 6
done
