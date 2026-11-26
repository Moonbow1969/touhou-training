#!/bin/sh
# Run on the Linux host. Inputs are the original official trial already extracted by the user.
set -eu
if [ "$#" != 4 ]; then
    echo "usage: $0 /absolute/path/to/trial-app /absolute/path/to/capture docker-image container-prefix" >&2
    exit 2
fi
game=$(realpath "$1")
capture=$(realpath -m "$2")
image=$3
prefix=$4
lanes=${TH13_LANES:-"a b"}
for lane in $lanes; do case "$lane" in a|b) ;; *) exit 2 ;; esac; done
src=$(CDPATH= cd "$(dirname "$0")" && pwd)
case "$prefix" in ""|*[!a-z0-9-]*) exit 2 ;; esac
for w in $lanes; do
    if docker container inspect "$prefix$w" >/dev/null 2>&1; then
        echo "$prefix$w already exists" >&2
        exit 1
    fi
done
printf '%s  %s\n' 09818929cceea76ff4c65630e649bbccdb85d92b2e28bb4dfe6526749e6f4fb6 "$game/th13.exe" | sha256sum -c -
test -f "$game/th13tr.dat" && test -f "$game/thbgm_tr.dat"
mkdir -p "$capture/a" "$capture/b"
gcc -Wall -Wextra -Werror -O2 "$src/pixel-capture.c" -o "$capture/pixel-capture" -l:libX11.so.6 -l:libXtst.so.6
for w in $lanes; do
    docker run -d --name "$prefix$w" --network none --cpus 2 --memory 1100m --pids-limit 192 \
        -v "$game:/input:ro" -v "$capture/$w:/capture" -v "$src/run-th13.sh:/run-th13.sh:ro" \
        -v "$capture/pixel-capture:/tmp/pixel-capture:ro" --entrypoint /bin/sh "$image" /run-th13.sh
done
for w in $lanes; do
    docker exec -e DISPLAY=:66 "$prefix$w" sh -eu -c '
        timeout 60 xdotool search --sync --onlyvisible --class th13 >/dev/null
        xdotool mousemove 427 364 click 1 mousemove 507 449 click 1
        window=$(timeout 35 xdotool search --sync --onlyvisible --name "Ten Desires. ver 0.01a" | tail -1)
        xdotool windowfocus --sync "$window"
        sleep 5
        for screen in title difficulty character; do
            scrot --overwrite "/capture/$screen.png"
            xdotool keydown z; sleep .15; xdotool keyup z; sleep 1.4
        done
        sleep 7
        scrot --overwrite /capture/first-stage.png
        xdotool keydown Escape; sleep .18; xdotool keyup Escape; sleep .8
    ' &
done
wait
