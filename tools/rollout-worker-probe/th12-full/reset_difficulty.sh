#!/bin/sh
set -eu
export DISPLAY=:66 WINEPREFIX=/tmp/th12-wine WINEARCH=win32 WINEDEBUG=-all
label=$1
case "$label" in ""|*[!a-zA-Z0-9_.-]*) exit 2;; esac
case "${TH12_DIFFICULTY:-}" in 2|3) ;; *) exit 2;; esac
dir="/capture/full/$label"
mkdir -p "$dir"
test ! -e "$dir/decisions.tsv"
state=$(/input/th12-full-actor status)
printf '1 reset=0 %s\n' "$state" > "$dir/credit.reset.log"
if [ "$state" != title ]; then
    TH12_ALLOW_TERMINAL_RELAUNCH=1 sh /input/th12-rank-return-title.sh
fi
window=$(xdotool search --onlyvisible --name 'Undefined Fantastic Object.*0\.02a' | tail -1)
test -n "$window"
xdotool windowfocus --sync "$window"
if command -v xset >/dev/null; then xset r off; fi
tap() { xdotool keydown "$1"; sleep .16; xdotool keyup "$1"; sleep .62; }
test "$(/input/th12-full-actor status)" = title
tap z; sleep .45
scrot -o -a 3,22,640,480 "$dir/difficulty-menu-initial.png"
for i in $(seq 2 "$TH12_DIFFICULTY"); do tap Down; done
scrot -o -a 3,22,640,480 "$dir/difficulty-menu-selected.png"
for item in rank character shot; do tap z; sleep .6; done
for i in $(seq 1 90); do
    screen=$(/input/th12-rank-camera-actor status)
    /input/th12-rank-camera-actor hud > "$dir/credit.initial-hud.txt"
    printf '%s screen=%s ' "$i" "$screen" >> "$dir/credit.reset.log"
    tr '\n' ' ' < "$dir/credit.initial-hud.txt" >> "$dir/credit.reset.log"; echo >> "$dir/credit.reset.log"
    if [ "$screen" = play ] && grep -qx 'lives 2' "$dir/credit.initial-hud.txt" && grep -qx 'bombs 2' "$dir/credit.initial-hud.txt"; then
        sleep .25
        [ "$(/input/th12-rank-camera-actor status)" = play ] || continue
        wine /input/th12-difficulty.exe | tr -d '\r' > "$dir/difficulty-at-reset.tsv"
        if awk -v expected="$TH12_DIFFICULTY" 'NR==2{ok=$2==expected && $3==expected && $4==1 && $5==0 && $6==0 && $7==0 && $8>=0 && $9==2} END{exit !ok}' "$dir/difficulty-at-reset.tsv"; then
            scrot -o -a 3,22,640,480 "$dir/difficulty-start-hud.png"
            scrot -o "$dir/credit.start.png"
            exit 0
        fi
        echo "Unexpected native recording difficulty or shot" >&2; cat "$dir/difficulty-at-reset.tsv" >&2
        exit 5
    fi
    sleep .15
done
exit 6
