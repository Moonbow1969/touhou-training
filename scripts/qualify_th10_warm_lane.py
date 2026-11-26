"""Qualify three original TH10 credits in one private Wine/Xvfb instance and measure warm starts."""
import argparse
import csv
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
from .collect_th10_full_action import GAME_SHA,install_weights,pin


def run(args):
    client=json.loads(args.workers.read_bytes())[args.host];endpoint=client["command"][:-1]
    build=pin(client,warm_reset=True);sha=install_weights(client,args.model,args.expected_sha)
    script=r'''
set -eu
base="$HOME/.local/share/toho-ml/th10-official-trial-002a"
test "$(df -Pk "$base"|awk 'NR==2 {print $4}')" -gt 8388608
test "$(sha256sum "$base/installed-utf8/game/th10tr.exe" | cut -d' ' -f1)" = GAME
dir=$(mktemp -d "$base/warm-qual.XXXXXXXX");mkdir "$dir/capture"
printf 'warm-job %s\n' "$dir" >&2
python3 -c 'import time;print(time.time_ns()//1000000)' > "$dir/capture/cold-start.ms"
rc=0
docker run --rm --init --network=none --cpus=3 --memory=1536m --name="toho-th10-warm-reset-${dir##*.}" --mount "type=bind,source=$base/installed-utf8/game,target=/input/game,readonly" --mount "type=bind,source=$base/full-code-BUNDLE,target=/tools,readonly" --mount "type=bind,source=$base/full-models/SHA.bin,target=/input/actor.bin,readonly" --mount "type=bind,source=$dir/capture,target=/capture" --entrypoint /bin/sh IMAGE /tools/run-th10-trial-probe.sh warm-reset-qualification SEED >"$dir/runner.log" 2>&1 || rc=$?
cd "$dir"
tar -czf - --exclude=pixels.u8 --exclude=features.f32 capture runner.log
exit "$rc"
'''.replace("GAME",GAME_SHA).replace("BUNDLE",build["bundle"]).replace("SHA",sha).replace("IMAGE",shlex.quote(client["runtime"])).replace("SEED",str(args.seed))
    args.out.mkdir(exist_ok=True,parents=True)
    reply=subprocess.run([*endpoint,script],capture_output=True,timeout=430)
    (args.out/'transport.log').write_bytes(reply.stderr)
    with tarfile.open(fileobj=io.BytesIO(reply.stdout),mode="r:gz") as archive:
        for member in archive:
            name=Path(member.name)
            if member.isfile() and not name.is_absolute() and '..' not in name.parts:
                file=args.out/name;file.parent.mkdir(parents=True,exist_ok=True)
                file.write_bytes(archive.extractfile(member).read())
    if reply.returncode:raise RuntimeError("Warm runner stopped; see the archived original-game evidence.")
    rows=[]
    for i in range(3):
        d=args.out/'capture'/f'cycle-{i}'
        def contents(name):return (d/name).read_text(encoding='utf-8').strip()
        first=next(csv.DictReader(io.StringIO(contents('opening.csv'))))
        game_states=list(csv.DictReader(io.StringIO(contents('native.csv'))));last=game_states[-1]
        stock=contents('opening-inventory.csv').split(',')
        settled=contents('settled-inventory.csv').split(',')
        assert [int(first[k]) for k in ['stage','difficulty','character','shot','lives','score','held_mask']]==[1,1,0,0,2,0,0]
        assert list(map(int,stock[1:]))==[1,2,0] and int(last['lives'])==-1 and int(settled[2])==-1
        first_motion=None
        for prev,now in zip(game_states,game_states[1:]):
            if int(now['lives'])!=2:break
            keys=int(now['native_input']);dx=float(now['x'])-float(prev['x']);dy=float(now['y'])-float(prev['y'])
            if ((keys&64 and dx<-.2) or (keys&128 and dx>.2) or (keys&16 and dy<-.2) or (keys&32 and dy>.2)):
                first_motion=(int(now['utc_ms'])-int(contents('reset-start.ms')))/1000
                break
        assert first_motion is not None
        rows.append(dict(seed=args.seed+i,start='cold' if i==0 else 'warm',reset_to_verified_play_s=(int(contents('first-play.ms'))-int(contents('reset-start.ms')))/1000,
                         reset_to_confirmed_player_motion_s=first_motion,
                         opening={k:first[k] for k in ['stage','difficulty','character','shot','lives','score','x','y','held_mask']},
                         opening_power=int(stock[3])/20,terminal=contents('native-stop'),terminal_power=int(settled[3])/20,
                         decisions=len(list(csv.DictReader(io.StringIO(contents('action.csv')))))))
    result=dict(host=args.host,game_sha256=GAME_SHA,policy_sha256=sha,build=build,cycles=rows,
                design="One isolated container on one physical host. Each run uses a new game process and a clean writable game directory. Warm restarts reuse Xvfb, PulseAudio, Wineprefix and the verified executable/data, then wait for the actual story state. No Continue or save state used.")
    (args.out/'report.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workers',required=True,type=Path)
    p.add_argument('--model',required=True,type=Path);p.add_argument('--expected-sha',required=True);p.add_argument('--out',required=True,type=Path)
    p.add_argument('--seed',type=int,default=719801);p.add_argument('--host',default='accelerator');run(p.parse_args())
