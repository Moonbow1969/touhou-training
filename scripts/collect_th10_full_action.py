"""Collect genuine full-life, no-Continue TH10 credits with the shared replay/PPO pixel actor."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

import numpy as np


PROBE = Path(__file__).resolve().parents[1] / "tools" / "rollout-worker-probe"
GAME_SHA = "5430fac601d645798ccf794362ec7d4cd753814f09d160247dc628b7c6943fd8"
FLEET = ("accelerator","momentum")
FULL_KEYS = ("move","focus","shoot","bomb")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)


def pin(client, paired=False, actor_header=None, locator_params=None, warm_reset=False):
    files = {name:(PROBE/name).read_bytes() for name in
             ("run-th10-trial-probe.sh","th10-trial-state.h","th10-trial-observe.c","th10-full-stock.c","th10-full-actor.c")}
    if paired:
        for n in ("th10-paired-targets.c","th10-trial-step.c"):files[n]=(PROBE/n).read_bytes()
    if warm_reset:
        files["th10-warm-reset.sh"]=(PROBE/"th10-warm-reset.sh").read_bytes()
    files["full_actor.h"] = (actor_header or PROBE.parent/"full_action/full_actor.h").read_bytes()
    if locator_params:
        files["native_locator.h"] = (PROBE.parent/"full_action/native_locator.h").read_bytes()
        files["locator_params.h"] = locator_params.read_bytes()
    files["x11_pixels.h"] = (PROBE/"th13-pixels/x11_pixels.h").read_bytes()
    sources={k:digest(b) for k,b in files.items()}
    bundle=digest(json.dumps(sources,sort_keys=True).encode())[:16]
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode="w:gz") as z:
        for name,data in files.items():
            info=tarfile.TarInfo(name);info.size=len(data);z.addfile(info,io.BytesIO(data))
    script=r"""
set -eu
umask 077
base="$HOME/.local/share/toho-ml/th10-official-trial-002a"
src="$base/full-code-BUNDLE"
if test ! -x "$src/full-actor"; then
  tmp=$(mktemp -d "$base/build-full.XXXXXXXX")
  tar -xz -C "$tmp"
  gcc -std=c11 ACTOR_FLAGS -DPLAYFIELD_LEFT=32 -DPLAYFIELD_TOP=16 "$tmp/th10-full-actor.c" -o "$tmp/full-actor" -l:libX11.so.6 -l:libXtst.so.6 -lm
  mv -Tn "$tmp" "$src"
fi
test -x "$src/full-actor"
sha256sum "$src/full-actor" | cut -d' ' -f1
""".replace("BUNDLE",bundle).replace("ACTOR_FLAGS","-O3 -DFA_LEARNED_LOCATOR=1" if locator_params else "-O2")
    reply=subprocess.run([*client["command"][:-1],script],input=payload.getvalue(),capture_output=True,timeout=30)
    if reply.returncode:raise RuntimeError(reply.stderr.decode(errors="replace")[-700:])
    return dict(bundle=bundle,sources=sources,executable=digest(files["th10-full-actor.c"]),linux_binary=reply.stdout.decode().strip())


def install_weights(client,path,expected_sha=None):
    data=path.read_bytes();sha=digest(data)
    if expected_sha and sha!=expected_sha:raise ValueError("Deployment SHA does not match the registered arm.")
    if data[:8]!=b"THFULL01" and not expected_sha:raise ValueError("An alternate actor requires an exact deployment SHA.")
    script=r"""
set -eu
umask 077
base="$HOME/.local/share/toho-ml/th10-official-trial-002a/full-models"; mkdir -p "$base"
tmp=$(mktemp "$base/upload.XXXXXXXX"); cat >"$tmp"
test "$(sha256sum "$tmp" | cut -d' ' -f1)" = SHA
if test -e "$base/SHA.bin"; then test "$(sha256sum "$base/SHA.bin"|cut -d' ' -f1)" = SHA; rm -f "$tmp"; else mv "$tmp" "$base/SHA.bin"; fi
""".replace("SHA",sha)
    subprocess.run([*client["command"][:-1],script],input=data,capture_output=True,check=True,timeout=30)
    return sha


def native_mask(action):
    return ((action&64>0)*1+(action&128>0)*2+(action&16>0)*4+(action&32>0)*8+
            (action&4>0)*16+(action&1>0)*32+(action&2>0)*64)


def unpack(raw,folder,host,seed,arm,sha,pinned):
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as z:
        def read(k):return z.extractfile(k).read()
        stop=read("native-stop").decode().strip()
        if stop!="native_final_life_lost":
            native=list(csv.DictReader(io.StringIO(read("native.csv").decode())))
            if any(int(row["stage"])==3 for row in native):
                (folder/"review-endpoint.png").write_bytes(read("endpoint.png"))
                (folder/"review-credit.tar.gz").write_bytes(raw)
                save(folder/"review-credit.json",dict(seed=seed,arm=arm,policy_sha256=sha,
                     native_stop=stop,last_native=native[-1],archive_sha256=digest(raw)))
                raise EOFError("Stage 3 endpoint needs review. Full original credit and endpoint were preserved.")
            raise RuntimeError("Technical censor: "+stop)
        opening=list(csv.DictReader(io.StringIO(read("opening.csv").decode())))[0]
        if [int(opening[k]) for k in ("stage","difficulty","lives","score")]!=[1,1,2,0]:
            raise ValueError("A fresh Normal credit was not verified.")
        data=read("native.csv")
        states=list(csv.DictReader(io.StringIO(data.decode())))
        logs=list(csv.DictReader(io.StringIO(read("action.csv").decode())))
        feature_data=np.frombuffer(read("features.f32"),dtype="<f4")
        if not logs or len(feature_data)%len(logs):raise ValueError("Invalid full-actor feature stream")
        feat=feature_data.reshape(len(logs),-1).copy()
        obs=np.frombuffer(read("pixels.u8"),dtype=np.uint8).reshape(-1,112,96)
        mono=np.loadtxt(io.BytesIO(read("frame.ns")),dtype=np.int64)
        screen=read("endpoint.png")
        settled=list(map(int,read("settled-inventory.csv").decode().strip().split(","))) if "settled-inventory.csv" in z.getnames() else None
    if not logs or len(feat)!=len(logs) or len(obs)!=len(logs)+1 or len(mono)!=len(obs) or not np.isfinite(feat).all():
        raise ValueError("Unaligned or nonfinite native actor trace.")
    act=np.array([[int(r[k]) for k in FULL_KEYS] for r in logs],dtype=np.int16)
    lp=np.array([[float(r['lp_'+k]) for k in FULL_KEYS] for r in logs],dtype=np.float32)
    val=np.array([float(r['value']) for r in logs],dtype=np.float32)
    if not np.isfinite(lp).all() or not np.isfinite(val).all():raise ValueError("Nonfinite replay of native actions.")
    utc=np.array([int(r['utc_ms']) for r in logs],dtype=np.int64)
    n=len(logs); at=lambda t: int(np.clip(np.searchsorted(utc,t,side="right")-1,0,n-1))
    costs=np.zeros(n,np.float32);life=np.zeros(n,np.float32);stage=np.zeros(n,np.float32)
    misses=[];casts=[];entries=[dict(stage=1,step=0,lives=2,power=0.)];epochs=[]
    last_miss=-10**15;last_x_on=-10**15;prev=states[0]
    correct=np.zeros(n,np.int32);reads=np.zeros(n,np.int32)
    request=np.array([int(r['mask']) for r in logs])
    if int(states[-1]['lives'])!=-1 or int(states[0]['lives'])!=2:raise ValueError("Original stock endpoints are inconsistent.")
    for s in states:
        tick=int(s['utc_ms']);i=at(tick)
        consumed=int(s['native_input'])
        if tick>=utc[0]:
            correct[i]+=int(native_mask(consumed)==request[i]);reads[i]+=1
        lpv=int(prev['lives']);lv=int(s['lives']);power=int(s['power_units']);prior_power=int(prev['power_units'])
        if not 0<=power<=100 or not -1<=lv<=8:raise ValueError("TH10 stock beyond qualified bounds.")
        if lv!=lpv:
            life[i]+=lv-lpv
            if lv<lpv:
                last_miss=tick;misses.append(dict(step=i,before=lpv,after=lv,power=power/20))
        if consumed&2 and not int(prev['native_input'])&2:last_x_on=tick
        if prior_power-power==20 and lv==lpv and (consumed&2) and tick-last_x_on<=320 and tick-last_miss>=500:
            casts.append(dict(step=i,power_before=prior_power/20,power_after=power/20,lives=lv));costs[i]-=.03
        if s['stage']!=prev['stage']:
            delta=int(s['stage'])-int(prev['stage'])
            if delta!=1:raise ValueError("The original stage did not progress monotonically.")
            stage[i]+=2;entries.append(dict(stage=int(s['stage']),step=i,lives=lv,power=power/20))
        if power!=prior_power:epochs.append(dict(step=i,before=prior_power/20,after=power/20))
        prev=s
    last_ms=int(states[-1]['utc_ms'])
    elapsed=np.maximum(0,np.minimum(np.r_[utc[1:],last_ms],last_ms)-utc)/1000
    survival=elapsed.astype(np.float32)*.001
    done=np.zeros(n,np.bool_);done[-1]=True
    event_stocks=[s for s in states if int(s['stage'])>=2]
    np.savez_compressed(folder/'behavior.npz',features=feat,action=act,logp=lp,value=val,
                        reward=survival+costs+life+stage,terminated=done,
                        reward_survival=survival,reward_bomb=costs,reward_life=life,reward_stage=stage,
                        action_utc_ms=utc,clock_ns=mono[:-1],native_input_matches=correct,native_samples=reads)
    np.savez_compressed(folder/'original-pixels.npz',obs=obs,monotonic_ns=mono)
    with gzip.open(folder/'native.csv.gz','wb') as f:f.write(data)
    (folder/'endpoint.png').write_bytes(screen)
    last=states[-1]
    result=dict(title=10,host=host,seed=seed,arm=arm,policy_sha256=sha,source=pinned,
        steps=n,behavior_sha256=digest((folder/'behavior.npz').read_bytes()),sha256=digest((folder/'behavior.npz').read_bytes()),
        behavior="behavior.npz",game_sha256=GAME_SHA,
        native_outcome=stop,game_over=True,terminal=True,stage2_reached=bool(event_stocks),stage_entries=entries,
        stage2_lives=entries[1]['lives'] if len(entries)>1 else None,stage2_power=entries[1]['power'] if len(entries)>1 else None,
        final_stage=int(last['stage']),final_lives=int(last['lives']),
        power_at_final_life_loss=int(last['power_units'])/20,
        final_power=settled[3]/20 if settled else None,
        final_power_source="settled native inventory at the end screen" if settled else "requires endpoint HUD reading",
        bomb_stock=None,bomb_rule="TH10 consumes 1.00 Power per successful activation; there is no separate Bomb inventory.",
        successful_bombs=casts,power_changes=epochs,misses=misses,raw_score=int(last['raw_score']),
        native_inputs_matching=int((correct>0).sum()),native_inputs_with_sample=int((reads>0).sum()),native_samples=len(states),
        duration_s=float(elapsed.sum()),median_control_ms=float(np.median(np.diff(mono))/1e6),
        moves=np.bincount(act[:,0],minlength=9).tolist(),focus_on=int(act[:,1].sum()),
        shoot_on=int(act[:,2].sum()),bomb_on=int(act[:,3].sum()),continues=0,
        terminal_screenshot_sha256=digest(screen),game_rng_controlled=False)
    save(folder/'report.json',result)
    return result


def one(task):
    client,pinned,sha,host,seed,arm,dest,hold,paired=task
    folder=dest/f"{host}-{seed}"
    if (folder/'report.json').is_file():
        cached=json.loads((folder/'report.json').read_bytes())
        if cached['policy_sha256']!=sha:raise ValueError("Cannot resume with different weights.")
        if not paired or ((folder/'privileged-targets.json').is_file() and (folder/'privileged-targets.npz').is_file()):
            return cached
    folder.mkdir(exist_ok=True)
    script=r"""
set -eu
umask 077
base="$HOME/.local/share/toho-ml/th10-official-trial-002a"
test "$(df -Pk "$base"|awk 'NR==2{print $4}')" -gt 8388608
test "$(sha256sum "$base/installed-utf8/game/th10tr.exe"|cut -d' ' -f1)" = GAME
cd "$dir"
docker run --rm --network=none --init --cpus=3 --memory=1536m --log-driver=local --log-opt=max-size=2m --log-opt=max-file=1 --log-opt=compress=false --name="toho-th10-fa-JOB" --mount "type=bind,source=$base/installed-utf8/game,target=/input/game,readonly" --mount "type=bind,source=$base/full-code-BUNDLE,target=/tools,readonly" --mount "type=bind,source=$base/full-models/SHA.bin,target=/input/actor.bin,readonly" --mount "type=bind,source=$dir,target=/capture" --entrypoint /bin/sh IMAGE /tools/run-th10-trial-probe.sh MODE SEED HOLD > "$dir/worker.log" 2>&1
cd "$dir"; test ! -f actor-failed
tar -czf behavior.tar.gz.tmp action.csv features.f32 frame.ns pixels.u8 native.csv native-stop opening.csv endpoint.png settled-inventory.csv RICH
mv behavior.tar.gz.tmp behavior.tar.gz
tar -tzf behavior.tar.gz >/dev/null
rm -f -- pixels.u8 features.f32 rich.bin
""".replace('BUNDLE',pinned['bundle']).replace('GAME',GAME_SHA).replace('SHA',sha).replace('IMAGE',shlex.quote(client['runtime'])).replace('SEED',str(seed)).replace('HOLD',str(hold))
    script=script.replace("MODE","full-paired" if paired else "full-credit").replace(" RICH"," rich.bin" if paired else "")
    for attempt in range(1,4):
        label=f"{seed}-{sha[:12]}-{attempt}"
        directory=f'"$HOME/.local/share/toho-ml/th10-official-trial-002a/full-story/{label}"'
        endpoint=client["command"][:-1]
        body='dir='+directory+'\n'+script.replace("JOB",label)
        launch=('set -eu; dir='+directory+'; mkdir -p "$dir"; '
                'if mkdir "$dir/claimed" 2>/dev/null; then '
                "nohup sh -c "+shlex.quote("("+body+") ; rc=$?; echo $rc > "+directory+'/status')+
                ' </dev/null >"$dir/job.log" 2>&1 & fi')
        for connection_attempt in range(4):
            try:
                subprocess.run([*endpoint,launch],capture_output=True,check=True,timeout=20)
                break
            except (subprocess.SubprocessError,OSError):
                if connection_attempt==3:raise
                time.sleep(connection_attempt+1)
        deadline=time.monotonic()+48*60
        archive=b""
        try:
            while time.monotonic()<deadline:
                try:
                    qry=subprocess.run([*endpoint,'dir='+directory+'; test ! -s "$dir/status" || cat "$dir/status"'],
                                       capture_output=True,timeout=20)
                except (subprocess.TimeoutExpired,OSError):
                    time.sleep(4)
                    continue
                if qry.returncode==0 and qry.stdout.strip():break
                time.sleep(4)
            else:raise RuntimeError("Original-game credit reached infrastructure watchdog.")
            if qry.stdout.strip()!=b'0':
                err=subprocess.run([*endpoint,'dir='+directory+'; tail -20 "$dir/worker.log" "$dir/job.log"'],capture_output=True,timeout=12)
                (folder/f"worker-{attempt}.log").write_bytes(err.stdout)
                raise RuntimeError("Native worker exit "+qry.stdout.strip().decode())
            for transport_attempt in range(3):
                try:
                    archive=subprocess.run([*endpoint,'cat '+directory+'/behavior.tar.gz'],capture_output=True,check=True,timeout=90).stdout
                    break
                except (subprocess.SubprocessError,OSError):
                    if transport_attempt==2:raise
            row=unpack(archive,folder,host,seed,arm,sha,pinned)
            if paired:
                from scripts.pair_th10_full_state import export
                export(archive,folder)
            return row
        except (ValueError,RuntimeError,KeyError) as e:
            save(folder/f'rejected-{attempt}.json',dict(reason=str(e),archive_sha256=digest(archive),seed=seed,arm=arm))
            if attempt==3:raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers",type=Path,required=True);p.add_argument("--model",type=Path)
    p.add_argument("--paired",action="store_true")
    p.add_argument("--actor-header",type=Path)
    p.add_argument("--locator-params",type=Path,help="Compile the image-only learned player locator using these frozen parameters.")
    p.add_argument("--expected-sha")
    p.add_argument("--prepare-only",action="store_true");p.add_argument("--arm",default="sft")
    p.add_argument("--out",type=Path,required=True);p.add_argument("--seeds",nargs="+",type=int)
    p.add_argument("--concurrency",type=int,default=4);p.add_argument("--hold",type=int,default=120)
    args=p.parse_args()
    workers=json.loads(args.workers.read_bytes())
    args.out.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(2) as pool:
        if (args.actor_header or args.locator_params) and not args.expected_sha:p.error("Custom actors require --expected-sha")
        pins=dict(zip(FLEET,pool.map(lambda h:pin(workers[h],args.paired,args.actor_header,args.locator_params),FLEET)))
        if args.prepare_only:save(args.out/'builds.json',pins);return
        shas=list(pool.map(lambda h:install_weights(workers[h],args.model,args.expected_sha),FLEET))
    if shas[0]!=shas[1] or len(args.seeds)!=len(set(args.seeds)):raise ValueError("Invalid seed assignment or model distribution.")
    plan=dict(arm=args.arm,policy_sha256=shas[0],holds_ms=args.hold,seeds=args.seeds,
              assignment={str(s):FLEET[(s-1)%2] for s in args.seeds},
              complete_credits=0,technical_failures=[],receipts=[])
    save(args.out/'collection.json',plan)
    tasks=[(workers[h],pins[h],shas[0],h,int(s),args.arm,args.out,args.hold,args.paired) for s,h in plan['assignment'].items()]
    with ThreadPoolExecutor(args.concurrency) as pool:
        fs={pool.submit(one,x):x[4] for x in tasks}
        for f in as_completed(fs):
            try:r=f.result();plan['receipts'].append(r);plan['complete_credits']+=1
            except Exception as e:plan['technical_failures'].append(dict(seed=fs[f],reason=str(e)))
            save(args.out/'collection.json',plan)
    if plan['technical_failures']:raise RuntimeError("Technical censored credits remain; see collection.json")


if __name__=="__main__":
    main()
