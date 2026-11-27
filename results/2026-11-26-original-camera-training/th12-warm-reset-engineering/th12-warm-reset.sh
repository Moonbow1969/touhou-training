#!/bin/sh
set -eu
export DISPLAY=:66 WINEPREFIX=/tmp/th12-wine WINEARCH=win32 WINEDEBUG=-all
label=$1
case "$label" in ''|*[!a-zA-Z0-9_.-]*) exit 2;; esac
dir=/capture/full/$label
mkdir -p "$dir"
game_identity() {
    w=$(xdotool search --onlyvisible --name 'Undefined Fantastic Object.*0\.02a' | tail -1)
    pid=$(xprop -id "$w" _NET_WM_PID | awk '{print $NF}')
    test "$(cat "/proc/$pid/comm")" = th12.exe
    awk '{print $1,$22}' "/proc/$pid/stat"
}
tap() { xdotool keydown "$1"; sleep .15; xdotool keyup "$1"; sleep .38; }
photo() { scrot -o -a 3,22,640,480 "$dir/$1.png"; }
rank_read() { scrot -o -a 3,22,640,480 "$dir/current-rank.ppm"; /input/th12-warm-menu "$dir/current-rank.ppm" /input/th12-warm-menu.mask; }
test ! -e "$dir/decisions.tsv"
game_identity > "$dir/start-process.txt"
state=$(/input/th12-full-actor status)
printf '%s\n' "$state" > "$dir/initial-screen.txt"
case "$TH12_ENGINEERING_RESET" in
cold)
    TH12_ALLOW_TERMINAL_RELAUNCH=1 sh /input/th12-rank-return-title.sh
    ;;
warm)
    case "$state" in
    score_ranking)
        photo original-ranking-before-navigation
        tap Up; tap Left; photo original-name-end; tap z; sleep .6
        state=$(/input/th12-full-actor status)
        ;;
    esac
    case "$state" in
    gameover)
        photo original-game-over-menu;tap Down;photo original-return-to-title;tap z
        ;;
    title) test "${TH12_WARM_ALREADY_TITLE:-0}" = 1;;
    *) echo "Unexpected original menu: $state" >&2;exit 4;;
    esac
    ;;
*) exit 2;;
esac
for attempt in $(seq 1 40); do
    test "$(/input/th12-full-actor status)" != title || break
    sleep .15
done
test "$(/input/th12-full-actor status)" = title
game_identity > "$dir/title-process.txt"
if [ "$TH12_ENGINEERING_RESET" = warm ]; then cmp "$dir/title-process.txt" "$dir/start-process.txt"; else ! cmp -s "$dir/title-process.txt" "$dir/start-process.txt"; fi
photo original-title-verified
tap z; sleep .55
photo rank-menu-as-entered
actual=$(rank_read); printf '%s\n' "$actual" > "$dir/rank-menu-as-entered.txt"
while [ "$actual" -gt 1 ]; do tap Up; actual=$((actual-1)); done
test "$(rank_read)" = 1
photo difficulty-menu-initial
for i in $(seq 2 "$TH12_DIFFICULTY"); do tap Down; done
test "$(rank_read)" = "$TH12_DIFFICULTY"
photo difficulty-menu-selected
game_identity > "$dir/new-story-process.txt"
cmp "$dir/new-story-process.txt" "$dir/title-process.txt"
printf 'engineering=%s title; selected original rank %s\n' "$TH12_ENGINEERING_RESET" "$TH12_DIFFICULTY" > "$dir/credit.reset.log"

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
