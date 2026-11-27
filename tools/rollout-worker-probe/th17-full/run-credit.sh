#!/usr/bin/env bash
# Collect one complete unassisted TH17 0.01a credit in an isolated Wine display.
set -euo pipefail
usage() {
    echo 'run-credit.sh --game <local trial dir> --model <frozen policy.ts> --output <fresh episode dir> --seed N --rank hard|lunatic --arm NAME --split train|held|engineering [--lane NAME] [--cpus N] [--wine-image NAME] [--camera-image NAME] [--runtime-dir PATH] [--socket <shared actor.sock>]'
}
wine_image=toho-worker-trial
camera_image=toho-camera-cpu
lane=local
cpus=2
runtime_parent=${TMPDIR:-/tmp}
shared_socket=
host_label=$(hostname)
while (($#)); do
    case "$1" in
        --game) game=$2;; --model) model=$2;; --output) output=$2;; --seed) seed=$2;;
        --rank) difficulty=$2;; --arm) arm=$2;; --split) split=$2;; --lane) lane=$2;;
        --cpus) cpus=$2;; --wine-image) wine_image=$2;; --camera-image) camera_image=$2;;
        --runtime-dir) runtime_parent=$2;; --socket) shared_socket=$2;; --host-label) host_label=$2;;
        --help|-h) usage; exit 0;; *) usage >&2; exit 2;;
    esac
    shift 2
done
: "${game:?--game is required}" "${model:?--model is required}" "${output:?--output is required}" "${seed:?--seed is required}"
: "${difficulty:?--rank is required}" "${arm:?--arm is required}" "${split:?--split is required}"
case "$difficulty" in hard) rank=2;; lunatic) rank=3;; *) usage >&2;exit 2;; esac
case "$split" in train|held|engineering) ;; *) usage >&2;exit 2;; esac
[[ $seed =~ ^[1-9][0-9]*$ && $lane =~ ^[a-zA-Z0-9_.-]+$ && $arm =~ ^[a-zA-Z0-9_.-]+$ ]]
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
game=$(realpath "$game")
model=$(realpath "$model")
for file in th17.exe th17tr.dat thbgm_tr.dat; do test -s "$game/$file"; done
game_sha=$(sha256sum "$game/th17.exe" | cut -d' ' -f1)
test "$game_sha" = 3ffa3963003cce6dc02b1dae5adeb4411138772834efdb195a05b9081027aaba
model_sha=$(sha256sum "$model" | cut -d' ' -f1)
mkdir -p "$(dirname "$output")"
mkdir "$output"
output=$(realpath "$output")
runtime=$(mktemp -d "$runtime_parent/th17-rollout.XXXXXXXX")
uid=$(basename "$runtime")
game_container="th17-game-${uid,,}"
camera_container="th17-camera-${uid,,}"
die() { echo "$*" >&2;exit 1; }
clean() {
    docker rm -f "$game_container" >/dev/null 2>&1 || true
    docker rm -f "$camera_container" >/dev/null 2>&1 || true
    # This unique private runtime contains only freshly built tools and service socket.
    python3 - "$runtime" <<'PY'
import pathlib, sys, shutil
p=pathlib.Path(sys.argv[1]).resolve()
if p.is_dir() and p.name.startswith("th17-rollout."):shutil.rmtree(p)
PY
}
trap clean EXIT
mkdir "$runtime/socket"
cc -std=c11 -O2 -I"$repo/tools/rollout-worker-probe/th12-full" -I"$repo/tools/full_action" "$here/worker.c" -o "$runtime/worker" -ldl -lm
# Pin the COFF timestamp. The registered Debian build reproduces its observer byte for byte.
cp "$here/sensor.c" "$runtime/sensor.c"
docker run --rm --network none -v "$runtime:/input" --entrypoint sh "$wine_image" -c \
    'SOURCE_DATE_EPOCH=1795755599 i686-w64-mingw32-gcc-posix -O2 /input/sensor.c -o /input/sensor.exe'
sensor_sha=$(sha256sum "$runtime/sensor.exe" | cut -d' ' -f1)
actor_sha=$(sha256sum "$runtime/worker" | cut -d' ' -f1)
if [[ -n $shared_socket ]]; then
    test -S "$shared_socket"
    socket_dir=$(dirname "$(realpath "$shared_socket")")
    socket_file=$(basename "$shared_socket")
    # The small helper shares the image used for array processing and score verification.
    docker run -d --network none --name "$camera_container" -v "$repo:/repo:ro" -v "$output:/capture" \
        --entrypoint sleep "$camera_image" infinity >/dev/null
else
    socket_dir=$runtime/socket
    socket_file=actor.sock
    docker run -d --network none --cpus 1 --memory 1g --name "$camera_container" \
        -v "$repo:/repo:ro" -v "$output:/capture" -v "$model:/model.ts:ro" -v "$socket_dir:/camera" "$camera_image" \
        /repo/tools/full_action/camera_service.py --model /model.ts --sha "$model_sha" --directory /camera >/dev/null
fi
docker run -d --network none --cpus="$cpus" --memory=1800m --pids-limit=224 --name "$game_container" \
    -e DISPLAY=:66 -e WINEPREFIX=/tmp/th17-wine -e WINEARCH=win32 -e WINEDEBUG=-all \
    -v "$here:/source:ro" -v "$game:/game:ro" -v "$runtime:/input:ro" -v "$socket_dir:/camera:ro" \
    -v "$output:/capture" --entrypoint sh "$wine_image" /source/desktop.sh >/dev/null
act() { docker exec "$game_container" "$@"; }
helper() { docker exec "$camera_container" python "/repo/tools/rollout-worker-probe/th17-full/$1.py" "${@:2}"; }
photo() { act scrot -o -a 3,22,640,480 "/capture/$1"; }
tap() { act xdotool keydown "$1" sleep .16 keyup "$1" sleep .55; }
recognize() {
    local expected=$1 filename=$2
    for attempt in $(seq 1 12); do
        photo "$filename"
        if helper original_ui "/capture/$filename" "$expected" >/dev/null; then return; fi
        sleep .25
    done
    die "Original $expected UI could not be confirmed"
}
for i in $(seq 1 100); do
    if act xwininfo -root -tree 2>/dev/null | grep -F '208x166+408+310' >/dev/null; then break; fi
    sleep .5
done
act xdotool mousemove 427 345 click 1 sleep .12 mousemove 510 454 click 1
win=
for i in $(seq 1 60); do
    win=$(act xwininfo -root -tree | awk '/\("th17.exe" "th17.exe"\).*640x480/ {print $1;exit}')
    [[ -z $win ]] || break
    sleep .4
done
[[ -n $win ]] || die 'Original game client unavailable'
act xdotool windowmove "$win" 3 22 windowfocus "$win"
recognize title title.png
sleep .6
tap z;recognize normal rank-initial.png
for ((i=1;i<rank;i++)); do tap Down; done
recognize "$difficulty" rank-selected.png
tap z;recognize character character.png
tap z;recognize wolf beast.png
docker exec -d -w /capture "$game_container" sh -c 'wine /input/sensor.exe >sensor.log 2>&1; echo $? >sensor.rc'
for i in $(seq 1 60); do [[ -s $output/live.state ]] && break; sleep .15; done
test -s "$output/live.state"
docker exec -d -w /capture -e "TOHO_CAMERA_SHA=$model_sha" -e "TOHO_CAMERA_SOCKET=/camera/$socket_file" \
    "$game_container" sh -c '/input/worker "$1" "$2" >worker.log 2>&1; echo $? >worker.rc' sh "$seed" "$rank"
tap z;sleep 4;photo start-hud.png
for i in $(seq 1 1800); do [[ -s $output/worker.rc ]] && break; sleep 2; done
[[ -s $output/worker.rc && $(cat "$output/worker.rc") == 0 ]] || die 'Incomplete native credit; retained raw evidence'
sleep 2;photo terminal.png
python3 - "$output" "$seed" "$rank" "$arm" "$split" "$lane" "$host_label" "$model_sha" "$sensor_sha" "$actor_sha" "$cpus" <<'PY'
import json,pathlib,sys
path,seed,rank,arm,split,lane,host,sha,sensor,actor,cpus=sys.argv[1:]
identity=dict(seed=int(seed),difficulty=int(rank),arm=arm,split=split,lane=lane,host=host,policy_sha256=sha,native_observer_sha256=sensor,actor_sha256=actor,cpu_quota=float(cpus))
pathlib.Path(path,"identity.json").write_text(json.dumps(identity,indent=2)+"\n")
PY
helper receipt /capture --identity /capture/identity.json

