#!/bin/sh
set -eu
study=$(cd "$(dirname "$0")/.." && pwd)
app="$study/app"
seed="$1"; rank="$2"; team="$3"; role="$4"
out="$study/games/$seed"
mkdir -p "$out"
test ! -e "$out/receipt.json"
test "$(df -Pk "$study" | tail -1 | awk '{print $4}')" -gt 8000000
mkfifo "$out/actions.fifo" "$out/frames.fifo"
name="toho-th08-multiteam-$seed"
end(){
  docker stop --time 2 "$name" >/dev/null 2>&1 || true
  rm -f "$out/actions.fifo" "$out/frames.fifo"
}
trap end EXIT
date -Ins > "$out/setup-start.txt"
docker run --rm --network none --name "$name" --cpus=1.70 --memory=1350m --pids-limit=240 --log-driver=none \
  -v "$app/assets/game:/input/game:ro" -v "$app/assets/stream:/input/stream:ro" \
  -v "$app/game-entry.sh:/input/run.sh:ro" -v "$app/postcredit-ranking.sh:/input/postcredit-ranking.sh:ro" \
  -v "$out:/capture" --entrypoint sh toho-trial:lab-5b67741 \
  -c 'sh /input/run.sh "$1" "$2" </capture/actions.fifo >/capture/frames.fifo' sh "$rank" "$team" >"$out/game-container.log" 2>&1 &
game=$!
docker run --rm --network none --name "$name-cpu" --cpus=1.00 --memory=1500m --pids-limit=100 \
  -e PYTHONPATH=/deps:/app -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -v "$app:/app:ro" -v "$study/deps:/deps:ro" -v "$out:/capture" -w /app \
  --entrypoint python3 toho-ppo-pilot:cpu-v1 -u -m scripts.run_th08_trial_camera \
  --assets /app/assets --policy /app/policy.ts --out /capture --rank "$rank" --seed "$seed" --role "$role" --verified-menu
wait "$game"
date -Ins > "$out/setup-end.txt"
