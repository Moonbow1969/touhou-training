#!/bin/sh
set -eu
export DISPLAY=:66
label=$1
case "$label" in ""|*[!a-zA-Z0-9_.-]*) exit 2;; esac
window=$(xdotool search --onlyvisible --name 'Undefined Fantastic Object.*0\.02a' | tail -1)
test -n "$window"
xdotool windowfocus --sync "$window"
if command -v xset >/dev/null; then xset r off; fi
directory="/capture/full/$label"
mkdir -p "$directory"
prefix="$directory/credit"
test ! -e "$prefix.tsv"
df -Pk /capture | awk 'NR==2 { exit $4 >= 1048576 ? 0 : 1 }'

tap() {
    xdotool keydown "$1"
    sleep .18
    xdotool keyup "$1"
    sleep .52
}

reset=0
for attempt in $(seq 1 95); do
    observed=$(/input/th12-full-actor status)
    status=${observed%% *}
    printf '%s reset=%s %s\n' "$attempt" "$reset" "$observed" >> "$prefix.reset.log"
    case "$status" in
        score_ranking)
            sh /input/th12-full-return-title.sh
            window=$(xdotool search --onlyvisible --name 'Undefined Fantastic Object.*0\.02a' | tail -1)
            xdotool windowfocus --sync "$window"
            reset=0
            ;;
        play)
            if [ "$reset" = 1 ]; then
                sleep .65
                if /input/th12-full-actor hud | grep -qx 'lives 2'; then break; fi
            fi
            tap Escape
            reset=0
            ;;
        pause|gameover) tap Down ;;
        gameover_retry) tap z; reset=1; sleep 1.2 ;;
        pause_retry) tap z; sleep .65 ;;
        confirm) tap Up; tap z; reset=1; sleep 1.2 ;;
        title) for selection in Start Normal Reimu TypeA; do tap z; sleep .65; done; reset=1 ;;
        loading) sleep .22 ;;
        *) exit 3 ;;
    esac
done
test "$reset" = 1
test "$attempt" -lt 95
/input/th12-full-actor hud > "$prefix.initial-hud.txt"
grep -qx 'lives 2' "$prefix.initial-hud.txt"
grep -qx 'bombs 2' "$prefix.initial-hud.txt"
scrot --overwrite "$prefix.start.png"
