"""Evaluate a frozen TH12/TH13 pixel actor in an untouched-by-this-actor TH10 trial."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

import numpy as np

PROBE = Path(__file__).resolve().parents[1]/"tools"/"rollout-worker-probe"
sys.path.insert(0,str(PROBE/"th13-pixels"))
from policy_credit import features
from update_policy import categorical

BINARY = "5430fac601d645798ccf794362ec7d4cd753814f09d160247dc628b7c6943fd8"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def execute(task, workers, output):
    host, arm, seed, lane, model = task
    client = workers[host]
    weights = np.load(model,allow_pickle=False)["weights"].astype("<f8")
    files = {name:(PROBE/name).read_bytes() for name in
             ["th10-trial-state.h","th10-trial-observe.c","run-th10-trial-probe.sh"]}
    files.update({p:(PROBE/"th13-pixels"/p).read_bytes() for p in ("pixel-policy.c","x11_pixels.h")})
    files["weights.bin"]=weights.tobytes()
    label = f"{host}-{arm}-{seed}-{lane}"
    folder = output/label
    folder.mkdir()
    input_tar = io.BytesIO()
    with tarfile.open(fileobj=input_tar,mode="w:gz") as tape:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tape.addfile(info,io.BytesIO(data))
    ssh = client["command"][:-1]
    # The remote shell allocates a new isolated directory for each frozen native credit.
    command = r"""
set -eu
umask 077
root="$HOME/.local/share/toho-ml/th10-official-trial-002a"
test "$(df -Pk "$root" | awk 'NR==2 {print $4}')" -gt 8388608
test "$(sha256sum "$root/installed-utf8/game/th10tr.exe" | cut -d' ' -f1)" = BINARY
run=$(mktemp -d "$root/joint-pixel.XXXXXXXX")
mkdir "$run/tools" "$run/credit"
tar -xzf - -C "$run/tools"
gcc -std=c11 -O2 -DPLAYFIELD_LEFT=32 -DPLAYFIELD_TOP=16 "$run/tools/pixel-policy.c" -o "$run/tools/pixel-actor" -l:libX11.so.6 -l:libXtst.so.6 -lm
docker run --rm --init --network=none --cpus=4 --memory=1536m --log-driver=local --log-opt=max-size=2m --log-opt=max-file=1 --log-opt=compress=false --name="toho-th10-px-${run##*.}" --mount "type=bind,source=$root/installed-utf8/game,target=/input/game,readonly" --mount "type=bind,source=$run/tools,target=/tools,readonly" --mount "type=bind,source=$run/credit,target=/capture" --entrypoint=/bin/sh IMAGE /tools/run-th10-trial-probe.sh pixel-credit SEED >&2
cd "$run/credit"
tar -czf - pixels.u8 wall.ns behavior.tsv actor-start.utc-ms native.csv opening.csv window-title.txt endpoint.png
""".replace("BINARY",BINARY).replace("IMAGE",shlex.quote(client["runtime"])).replace("SEED",str(seed))
    reply = subprocess.run([*ssh,command],input=input_tar.getvalue(),capture_output=True,timeout=155)
    (folder/"worker.log").write_bytes(reply.stderr)
    reply.check_returncode()
    if len(reply.stdout)>12_000_000:
        raise ValueError("Excessive archive size")
    with tarfile.open(fileobj=io.BytesIO(reply.stdout),mode="r:gz") as tape:
        def read(name):
            return tape.extractfile(name).read()
        native_raw=read("native.csv")
        native=list(csv.DictReader(io.StringIO(native_raw.decode())))
        start=int(read("actor-start.utc-ms"))
        opening=list(csv.DictReader(io.StringIO(read("opening.csv").decode())))[0]
        if (int(opening["stage"]),int(opening["difficulty"]),int(opening["lives"]),int(opening["score"]))!=(1,1,2,0):
            raise ValueError("Credit did not start at the original stage and fresh stock.")
        obs=np.frombuffer(read("pixels.u8"),dtype=np.uint8).reshape(-1,112,96)
        clocks=np.loadtxt(io.BytesIO(read("wall.ns")),dtype=np.int64)
        acts=list(csv.DictReader(io.StringIO(read("behavior.tsv").decode()),delimiter="\t"))
        action=np.array([int(row["action"]) for row in acts])
        probability=np.array([[float(row[k]) for k in ("p_left","p_neutral","p_right")] for row in acts])
        if len(action)!=len(obs)-1 or not np.allclose(categorical(features(obs[:-1]),weights),probability,atol=2e-9,rtol=2e-9):
            raise ValueError("Original pixel actions do not reproduce from the given weights.")
        if not 39.5 <= (clocks[-1]-clocks[0])/1e9 <= 42 or np.any(np.diff(clocks)<=0):
            raise ValueError("Actor clock does not cover the requested 40 seconds.")
        miss=next((int(s["utc_ms"]) for s in native if int(s["utc_ms"])>=start and int(s["lives"])<2),None)
        if not native or int(native[0]["utc_ms"])-start>750 or int(native[-1]["utc_ms"])<start+39800:
            raise ValueError("Native stock observations did not cover the credit.")
        (folder/"endpoint.png").write_bytes(read("endpoint.png"))
        (folder/"native.csv").write_bytes(native_raw)
        np.savez_compressed(folder/"behavior.npz",obs=obs,action=action,prob=probability,clock_ns=clocks)
        seconds=min(40.,(miss-start)/1000) if miss is not None else 40.
        r=dict(host=host,title=10,difficulty="Normal",shot="Reimu A",arm=arm,seed=seed,lane=lane,
               game_sha256=BINARY,model_sha256=sha(weights.tobytes()),crop=[32,16,416,464],
               decisions=len(action),native_samples=len(native),start_utc_ms=start,
               first_loss_within_limit=miss is not None and miss<=start+40000,
               first_loss_restricted_s=seconds,spares_last_read=int(native[-1]["lives"]),
               horizontal_counts=np.bincount(action,minlength=3).tolist(),
               position_range=[min(float(s["x"]) for s in native),max(float(s["x"]) for s in native)],
               source_sha256={k:sha(v) for k,v in files.items()},
               behavior_sha256=sha((folder/"behavior.npz").read_bytes()))
    (folder/"report.json").write_text(json.dumps(r,indent=2)+"\n",encoding="utf-8")
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("workers","parent","out"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--candidate",type=Path)
    p.add_argument("--seeds",type=int,nargs="+",required=True)
    args=p.parse_args()
    workers=json.loads(args.workers.read_bytes())
    args.out.mkdir(exist_ok=False)
    credits=[]
    for i,seed in enumerate(args.seeds):
        tasks=[]
        for host in ("momentum","accelerator"):
            swap=(i+int(host=="accelerator"))%2
            tasks.append((host,"parent",seed,swap,args.parent))
            if args.candidate:tasks.append((host,"candidate",seed,1-swap,args.candidate))
        with ThreadPoolExecutor(len(tasks)) as pool:
            credits.extend(pool.map(lambda t:execute(t,workers,args.out),tasks))
        (args.out/"report.json").write_text(json.dumps(dict(credits=credits,status="complete" if seed==args.seeds[-1] else "running"),indent=2)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
