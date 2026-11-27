#!/bin/sh
set -eu
export DISPLAY=:66
op=$1
game_identity() {
    window=$(xdotool search --onlyvisible --name 'Undefined Fantastic Object.*0\.02a' | tail -1)
    pid=$(xprop -id "$window" _NET_WM_PID | awk '{print $NF}')
    test "$(cat "/proc/$pid/comm")" = th12.exe
    awk '{print $1,$22}' "/proc/$pid/stat"
}
if [ "$op" = process ]; then game_identity; exit; fi
label=$2
case "$label" in ''|*[!a-zA-Z0-9_.-]*) exit 2;; esac
dir=/capture/full/$label
marker=/capture/last-complete-reset-source.tsv
case "$op" in
finish)
    grep -q '^game_over ' "$dir/outcome.txt"
    test -s "$dir/terminal-at-control.ppm"
    awk -F '\t' 'NR>1 && $5==0 && $2==$3 && $2>=1 && $4>=0 {stock=$6} END {exit stock!=-1}' "$dir/engineering-native.tsv"
    game_identity > "$dir/terminal-process.txt"
    cmp "$dir/new-story-process.txt" "$dir/terminal-process.txt"
    # The marker belongs to the currently locked lane and is atomically replaced.
    { printf '%s\n' "$label"; cat "$dir/terminal-process.txt"; sha256sum "$dir/outcome.txt" "$dir/terminal-at-control.ppm" "$dir/engineering-native.tsv" "$dir/difficulty-native.tsv"; } > "$marker.$$"
    mv "$marker.$$" "$marker"
    ;;
check)
    test -s "$marker" || { echo 'no signed preceding terminal on this lane' >&2; exit 3; }
    prior=$(head -n 1 "$marker")
    case "$prior" in ''|*[!a-zA-Z0-9_.-]*) exit 3;; esac
    test "$prior" != "$label"
    cp "$marker" "$dir/preceding-terminal.tsv"
    if [ -n "$(find /capture/full -mindepth 2 -maxdepth 2 -name terminal-at-control.ppm -newer "$marker" -print -quit)" ]; then
        echo 'intervening original terminal after last verified reset marker' >&2; exit 3
    fi
    sed '1,2d' "$marker" | sha256sum -c >&2
    test "$(sed -n 2p "$marker")" = "$(game_identity)" || { echo 'game process changed after the previous terminal' >&2; exit 3; }
    ;;
*) exit 2;;
esac
