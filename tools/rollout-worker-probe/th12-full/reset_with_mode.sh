#!/bin/sh
set -eu
export DISPLAY=:66 WINEPREFIX=/tmp/th12-wine WINEARCH=win32 WINEDEBUG=-all
label=$1
case "$label" in ''|*[!a-zA-Z0-9_.-]*) exit 2;; esac
dir=/capture/full/$label
mkdir -p "$dir"
test ! -e "$dir/decisions.tsv"
identity=/input/th12-reset-provenance.sh
base_reset=/input/th12-mode-rank-reset.sh
mode=${TH12_RESET_MODE:-cold}
case "$mode" in cold|warm) ;; *) exit 2;; esac
photo() { scrot -o -a 3,22,640,480 "$dir/$1.png"; }
tap() { xdotool keydown "$1"; sleep .15; xdotool keyup "$1"; sleep .38; }
warm_attempt() {
    sh "$identity" check "$label"
    state=$(/input/th12-full-actor status)
    if [ "$state" = score_ranking ]; then
        photo original-ranking-before-navigation; tap Up; tap Left
        photo original-score-name-end; tap z; sleep .6
        state=$(/input/th12-full-actor status)
    fi
    case "$state" in gameover|gameover_retry) ;; *) echo "unverified Game Over menu: $state" >&2; exit 4;; esac
    gameover_cursor() {
        scrot -o -a 3,22,640,480 "$dir/current-gameover.ppm"
        /input/th12-warm-menu "$dir/current-gameover.ppm" --gameover
    }
    photo original-game-over-menu; selected=$(gameover_cursor)
    printf '%s\n' "$selected" > "$dir/gameover-cursor-before.txt"
    case "$selected" in 0) tap Down;; 1) ;; 2) tap Up;; 3) tap Up; tap Up;; *) exit 4;; esac
    test "$(gameover_cursor)" = 1 || { echo 'Return to Title is not selected' >&2; exit 4; }
    photo original-return-to-title; tap z
    for i in $(seq 1 40); do test "$(/input/th12-full-actor status)" != title || break; sleep .15; done
    test "$(/input/th12-full-actor status)" = title
    photo original-title-verified
    TH12_READ_RETAINED_RANK=1 sh "$base_reset" "$label"
    sh "$identity" process > "$dir/new-story-process.txt"
    cmp "$dir/start-process.txt" "$dir/new-story-process.txt"
}
if [ "${TH12_WARM_CHILD:-0}" = 1 ]; then warm_attempt; exit; fi
sh "$identity" process > "$dir/start-process.txt" || echo unknown > "$dir/start-process.txt"
if [ "$mode" = warm ]; then
    # Separate shell is required: conditional shell functions suppress set -e.
    if TH12_WARM_CHILD=1 sh "$0" "$label" > "$dir/warm-navigation.log" 2>&1; then
        printf 'requested=warm\neffective=warm\n' > "$dir/reset-mode.txt"; exit 0
    else
        rc=$?; printf 'rc=%s\n' "$rc" > "$dir/warm-failure.txt"
        scrot -o -a 3,22,640,480 "$dir/failed-warm-screen.png" || true
        mkdir -p "$dir/failed-warm-entry"
        for f in "$dir"/difficulty-* "$dir"/credit.*; do
            if [ -f "$f" ]; then mv "$f" "$dir/failed-warm-entry/"; fi
        done
        TH12_ALLOW_TERMINAL_RELAUNCH=1 TH12_FAILED_RESET_LABEL="$label" sh /input/th12-mode-return-title.sh
        printf 'requested=warm\neffective=cold_fallback\nwarm_exit=%s\n' "$rc" > "$dir/reset-mode.txt"
    fi
else
    printf 'requested=cold\neffective=cold\n' > "$dir/reset-mode.txt"
fi
sh "$base_reset" "$label"
sh "$identity" process > "$dir/new-story-process.txt"
