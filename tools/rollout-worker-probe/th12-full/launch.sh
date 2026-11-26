#!/bin/sh
set -eu

root=${TH12_ROOT:-"$HOME/toho-th12-261125"}
batch="$root/capture/full-policy-live-261125"
image=${TH12_CONTAINER_IMAGE:-localhost/toho-trial:lab-5b67741}
executable=c6d15044394a33d6116c09eac44b33fdec814541c78518e29f60d54f49bdbd3a

test "$(nproc)" -ge 8
df -Pk "$root/capture" | awk 'NR==2 { exit $4 >= 1048576 ? 0 : 1 }'

mkdir "$batch"
for lane in alpha beta; do
    mkdir "$batch/$lane"
    podman run -d --rm --name "toho-th12-full-$lane-261125" --cpus 3 --memory 2g --network none \
        --pids-limit 192 -v "$batch/$lane:/capture:rw" -v "$root/input:/input:ro" -v "$root/game:/game:ro" \
        --entrypoint /bin/sh "$image" /input/th12-full-desktop.sh play > "$batch/$lane/container-id"
done

for lane in alpha beta; do
    scope="toho-th12-full-$lane-261125"
    window=
    configured=0
    for attempt in $(seq 1 70); do
        if test -f "$batch/$lane/executable.sha256"; then
            window=$(podman exec -e DISPLAY=:66 "$scope" xdotool search --onlyvisible --name 'Undefined Fantastic Object.*0\.02a' 2>/dev/null | tail -1)
            if test -n "$window"; then break; fi
            if test "$configured" -eq 0 && podman exec -e DISPLAY=:66 "$scope" xwininfo -tree -root \
                    | grep -F '("th12.exe" "th12.exe")  208x154+408+316'; then
                podman exec -e DISPLAY=:66 "$scope" scrot --overwrite /capture/resolution-dialog.png
                podman exec -e DISPLAY=:66 "$scope" xdotool mousemove 427 364 click 1 sleep .25 mousemove 510 449 click 1
                configured=1
            fi
        fi
        sleep 1
    done
    test -n "$window"
    podman exec -e DISPLAY=:66 "$scope" xdotool windowmove "$window" 3 22
    placed=0
    for attempt in $(seq 1 24); do
        status=$(podman exec -e DISPLAY=:66 "$scope" /input/th12-full-actor status)
        printf '%s\n' "$status" >> "$batch/$lane/initial-title-probe.log"
        if [ "${status%% *}" = title ]; then placed=1; break; fi
        sleep .5
    done
    test "$placed" -eq 1
    grep -qxF "$executable  ./th12.exe" "$batch/$lane/executable.sha256"
    podman exec "$scope" cat /sys/fs/cgroup/cpu.max /sys/fs/cgroup/memory.max > "$batch/$lane/resource-guard.log"
    grep -qxF '300000 100000' "$batch/$lane/resource-guard.log"
    grep -qxF '2147483648' "$batch/$lane/resource-guard.log"
done

touch "$batch/ready"
