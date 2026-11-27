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


def pin(client, paired=False, actor_header=None, locator_params=None, warm_reset=False, camera=False, camera_history=False, difficulty=1, stage_sensor='qualified-v2'):
    files = {name:(PROBE/name).read_bytes() for name in
             ("run-th10-trial-probe.sh","th10-trial-state.h","th10-trial-observe.c","th10-full-stock.c","th10-full-actor.c")}
    if stage_sensor not in ('legacy','qualified-v2'):raise ValueError('Unknown TH10 stage sensor.')
    if stage_sensor=='legacy':
        # Resume historical preregistered arms with their exact native source SHA.
        for name in ('th10-trial-state.h','th10-full-stock.c'):
            files[name]=(PROBE/'th10-legacy-stage'/name).read_bytes()
    if difficulty not in (0,1):raise ValueError('Full-credit capture supports Easy (0) and Normal (1).')
    if difficulty==0:
        # Build a distinct Easy-only launcher from the same verified difficulty-menu detector.
        # The existing Normal source bundle and cache identity stay byte-for-byte reproducible.
        launcher=files['run-th10-trial-probe.sh']
        for old,new in [(b'difficulty=1\n',b'difficulty=0\n'),
                        (b'if test "$difficulty" -eq 2; then\n    rank=',b'if test "$difficulty" -eq 0; then\n    rank='),
                        (b'xdotool keydown Down; sleep .18; xdotool keyup Down',b'xdotool keydown Up; sleep .18; xdotool keyup Up'),
                        (b'/capture/hard-rank.png',b'/capture/easy-rank.png')]:
            if launcher.count(old)!=1:raise ValueError('Original difficulty-menu launcher changed.')
            launcher=launcher.replace(old,new)
        files['run-th10-trial-probe.sh']=launcher
        files['th10-rank-screen.pl']=(PROBE/'th10-rank-screen.pl').read_bytes()
    if paired:
        for n in ("th10-paired-targets.c","th10-trial-step.c"):files[n]=(PROBE/n).read_bytes()
    if warm_reset:
        files["th10-warm-reset.sh"]=(PROBE/"th10-warm-reset.sh").read_bytes()
    files["full_actor.h"] = (actor_header or PROBE.parent/"full_action"/("camera_actor.h" if camera else "full_actor.h")).read_bytes()
    if camera:
        files["camera_service.py"]=(PROBE.parent/"full_action/camera_service.py").read_bytes()
    if camera_history:
        files["full_actor.h"]=b"#define FA_CAMERA_EXPLICIT_PREVIOUS_MOVE 1\n"+files["full_actor.h"]
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
    result=dict(bundle=bundle,sources=sources,executable=digest(files["th10-full-actor.c"]),linux_binary=reply.stdout.decode().strip(),camera=camera,camera_history=camera_history)
    if difficulty!=1:result['difficulty']=difficulty
    if stage_sensor!='legacy':result['stage_sensor']=stage_sensor
    return result


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

def start_camera_service(client,pinned,sha):
    script=r"""
set -eu
umask 077
base="$HOME/.local/share/toho-ml/th10-official-trial-002a"
dir="$base/camera-services/SHA-BUNDLE"
mkdir -p "$dir"; cd "$dir"
exec 9>start.lock; flock -x 9
probe(){ python3 -c 'import socket;s=socket.socket(socket.AF_UNIX);s.settimeout(3);s.connect("actor.sock");assert s.recv(64,socket.MSG_WAITALL)==b"SHA"'; }
if test -S actor.sock && probe 2>/dev/null; then exit 0; fi
if ! test -s pid || ! kill -0 "$(cat pid)" 2>/dev/null; then
  if test -S actor.sock; then rm -- actor.sock; fi
  py=python3
  if ! python3 -c 'import torch' 2>/dev/null; then
    py="$HOME/.local/share/toho-ml/distributed-ppo/venv/bin/python3"
    "$py" -c 'import torch'
  fi
  nohup "$py" "$base/full-code-BUNDLE/camera_service.py" --model "$base/full-models/SHA.bin" --sha SHA --directory "$dir" HIST 9>&- </dev/null >service.log 2>&1 &
  echo $! >pid
fi
for i in $(seq 1 75); do
  if test -S actor.sock && probe 2>/dev/null; then exit 0; fi
  if ! kill -0 "$(cat pid)" 2>/dev/null;then tail -8 service.log >&2;exit 1;fi
  sleep .2
done
exit 1
""".replace("BUNDLE",pinned["bundle"]).replace("SHA",sha).replace("HIST","--history-byte" if pinned.get("camera_history") else "")
    proc=subprocess.run([*client['command'][:-1],script],stdin=subprocess.DEVNULL,capture_output=True,timeout=35)
    if proc.returncode:raise RuntimeError(proc.stderr.decode(errors="replace")[-900:])


def unpack(raw,folder,host,seed,arm,sha,pinned,hold=120):
    with tarfile.open(fileobj=io.BytesIO(raw),mode="r:gz") as z:
        def read(k):return z.extractfile(k).read()
        if 'actor-identity.json' in z.getnames():
            identity=json.loads(read('actor-identity.json'))
            if identity['seed']!=seed or identity['policy_sha256']!=sha or identity['hold_ms']!=hold:
                raise ValueError('The immutable native worker ran a different policy, seed or cadence.')
            for flag in ('camera','camera_history'):
                if bool(identity['source'].get(flag))!=bool(pinned.get(flag)):
                    raise ValueError('The original worker used a different observation contract.')
            if identity['source'].get('difficulty',1)!=pinned.get('difficulty',1):
                raise ValueError('The original worker used another difficulty selection.')
            if identity['source'].get('stage_sensor','legacy')!=pinned.get('stage_sensor','legacy'):
                raise ValueError('The original worker used a different stage register.')
            pinned=identity['source']
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
        difficulty=pinned.get('difficulty',1)
        if [int(opening[k]) for k in ("stage","difficulty","lives","score")]!=[1,difficulty,2,0]:
            raise ValueError("A fresh original credit at the specified difficulty was not verified.")
        data=read("native.csv")
        states=list(csv.DictReader(io.StringIO(data.decode())))
        qualified_stage=pinned.get('stage_sensor')=='qualified-v2'
        if qualified_stage:
            if not states or 'legacy_start_field' not in states[0] or int(states[0]['stage'])!=1:
                raise ValueError('Missing qualified progression telemetry or fresh Stage 1 opening.')
            (folder/'opening.csv').write_bytes(read('opening.csv'))
        if difficulty==0:
            if any(int(row['difficulty'])!=0 for row in states):raise ValueError('Native Easy difficulty changed during the credit.')
            (folder/'easy-rank.png').write_bytes(read('easy-rank.png'))
            (folder/'opening.csv').write_bytes(read('opening.csv'))
        logs=list(csv.DictReader(io.StringIO(read("action.csv").decode())))
        feature_data=np.frombuffer(read("features.f32"),dtype="<f4")
        if not logs or len(feature_data)%len(logs):raise ValueError("Invalid full-actor feature stream")
        feat=feature_data.reshape(len(logs),-1).copy()
        obs=np.frombuffer(read("pixels.u8"),dtype=np.uint8).reshape(-1,112,96)
        mono=np.loadtxt(io.BytesIO(read("frame.ns")),dtype=np.int64)
        screen=read("endpoint.png")
        settled=list(map(int,read("settled-inventory.csv").decode().strip().split(","))) if "settled-inventory.csv" in z.getnames() else None
        camera_logits=np.frombuffer(read("camera-logits.f32"),dtype='<f4').reshape(-1,15).copy() if pinned.get("camera") else None
        previous_move=np.frombuffer(read('previous-move.u8'),dtype=np.uint8).copy() if pinned.get('camera_history') else None
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
    extra={}
    if camera_logits is not None:
        if camera_logits.shape!=(n,15) or not np.isfinite(camera_logits).all():raise ValueError("Invalid camera model logits")
        for head,(left,right) in enumerate(((0,9),(9,11),(11,13),(13,15))):
            logits=camera_logits[:,left:right].astype(np.float64);logits-=logits.max(1,keepdims=True)
            exact=logits[np.arange(n),act[:,head]]-np.log(np.exp(logits).sum(1))
            if not np.allclose(exact,lp[:,head],atol=2e-5,rtol=1e-6):raise ValueError("Acted camera logits differ from the logged behavior.")
        extra['camera_logits']=camera_logits
        if previous_move is not None:
            if not np.array_equal(previous_move,np.r_[4,act[:-1,0]]):raise ValueError("Transmitted camera history differs from the previous sampled movement.")
            extra['previous_move']=previous_move
    np.savez_compressed(folder/'behavior.npz',features=feat,action=act,logp=lp,value=val,**extra,
                        reward=survival+costs+life+stage,terminated=done,
                        reward_survival=survival,reward_bomb=costs,reward_life=life,reward_stage=stage,
                        action_utc_ms=utc,clock_ns=mono[:-1],native_input_matches=correct,native_samples=reads)
    np.savez_compressed(folder/'original-pixels.npz',obs=obs,monotonic_ns=mono)
    with gzip.open(folder/'native.csv.gz','wb') as f:f.write(data)
    (folder/'endpoint.png').write_bytes(screen)
    last=states[-1]
    result=dict(title=10,host=host,seed=seed,arm=arm,requested_hold_ms=hold,policy_sha256=sha,source=pinned,
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
    if camera_logits is not None:
        result.update(camera_policy=True,camera_model_inputs="original-pixels.npz obs[:-1]; [gray,abs(gray-previous)] / 255; first difference zero",features_role="Portable diagnostic features; raw camera is the model input.")
        if pinned.get('camera_history'):result['additional_policy_input']='Previous sampled movement; 4 (neutral) at the start of each credit.'
    if result['median_control_ms']<hold-3:raise ValueError('Observed camera cadence is faster than the registered native hold.')
    result.update(stage_sensor='qualified-v2' if qualified_stage else 'legacy-unvalidated',
                  stage_register='0x472b40' if qualified_stage else '0x472b44',
                  stage_sensor_validated=qualified_stage)
    if qualified_stage:
        result['highest_observed_native_stage']=max(int(s['stage']) for s in states)
        result['legacy_start_field_values']=sorted({int(s['legacy_start_field']) for s in states})
        first_stage2=next((j for j,row in enumerate(states) if int(row['stage'])==2),None)
        entry=None
        if first_stage2 is not None:
            row=states[first_stage2]
            assert first_stage2>0 and int(states[first_stage2-1]['stage'])==1
            entry=dict(native_row=first_stage2,utc_ms=int(row['utc_ms']),action_step=at(int(row['utc_ms'])),
                       spares=int(row['lives']),power_units=int(row['power_units']),power=int(row['power_units'])/20,
                       legacy_start_field=int(row['legacy_start_field']))
        result['stage2_evidence']=dict(schema='th10-native-stage-v2',native_trace_file='native.csv.gz',
            native_trace_sha256=digest((folder/'native.csv.gz').read_bytes()),native_rows=len(states),
            stage_column='stage',stage_register='0x472b40',legacy_column='legacy_start_field',legacy_register='0x472b44',
            state_header_sha256=pinned['sources']['th10-trial-state.h'],stock_source_sha256=pinned['sources']['th10-full-stock.c'],
            source_bundle=pinned['bundle'],first_stage2=entry,highest_native_stage=result['highest_observed_native_stage'])
    if difficulty==0:result.update(difficulty=0,difficulty_name='Easy',difficulty_verified_native_samples=len(states),
                                   difficulty_menu_sha256=digest((folder/'easy-rank.png').read_bytes()))
    save(folder/'report.json',result)
    return result


def one(task):
    client,pinned,sha,host,seed,arm,dest,hold,paired=task
    folder=dest/f"{host}-{seed}"
    if (folder/'report.json').is_file():
        cached=json.loads((folder/'report.json').read_bytes())
        if cached['policy_sha256']!=sha:raise ValueError("Cannot resume with different weights.")
        if cached['source'].get('difficulty',1)!=pinned.get('difficulty',1):raise ValueError("Cannot resume another difficulty.")
        if cached['source'].get('stage_sensor','legacy')!=pinned.get('stage_sensor','legacy'):
            raise ValueError('Cannot reuse an original credit from a different stage sensor.')
        if cached.get('requested_hold_ms',120)!=hold:raise ValueError("Cannot resume a different control cadence.")
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
printf '%s\n' IDENTITY > actor-identity.json
docker run --rm --network=none --init --cpus=3 --memory=1536m --log-driver=local --log-opt=max-size=2m --log-opt=max-file=1 --log-opt=compress=false --name="toho-th10-fa-JOB" --mount "type=bind,source=$base/installed-utf8/game,target=/input/game,readonly" --mount "type=bind,source=$base/full-code-BUNDLE,target=/tools,readonly" --mount "type=bind,source=$base/full-models/SHA.bin,target=/input/actor.bin,readonly" --mount "type=bind,source=$dir,target=/capture" --entrypoint /bin/sh IMAGE /tools/run-th10-trial-probe.sh MODE SEED HOLD > "$dir/worker.log" 2>&1
cd "$dir"; test ! -f actor-failed
tar -czf behavior.tar.gz.tmp actor-identity.json action.csv features.f32 frame.ns pixels.u8 native.csv native-stop opening.csv endpoint.png settled-inventory.csv RICH
mv behavior.tar.gz.tmp behavior.tar.gz
tar -tzf behavior.tar.gz >/dev/null
rm -f -- pixels.u8 features.f32 rich.bin
""".replace('BUNDLE',pinned['bundle']).replace('GAME',GAME_SHA).replace('SHA',sha).replace('IMAGE',shlex.quote(client['runtime'])).replace('SEED',str(seed)).replace('HOLD',str(hold))
    script=script.replace("MODE","full-paired" if paired else "full-credit").replace(" RICH",(" rich.bin" if paired else "")+(" camera-logits.f32" if pinned.get("camera") else "")+(" previous-move.u8" if pinned.get("camera_history") else ""))
    if pinned.get('difficulty',1)==0:script=script.replace('endpoint.png settled-inventory.csv','endpoint.png easy-rank.png settled-inventory.csv')
    script=script.replace('IDENTITY',shlex.quote(json.dumps(dict(source=pinned,seed=seed,hold_ms=hold,policy_sha256=sha),separators=(',',':'))))
    if pinned.get("camera"):
        mount='--env TOHO_CAMERA_SHA='+sha+' --mount "type=bind,source=$base/camera-services/'+sha+'-'+pinned['bundle']+',target=/camera,readonly" '
        script=script.replace("--entrypoint /bin/sh",mount+"--entrypoint /bin/sh")
    for attempt in range(1,4):
        cadence_suffix="" if hold==120 else f"-h{hold}"
        if pinned.get('difficulty',1)==0:cadence_suffix+='-easy'
        if pinned.get('stage_sensor')=='qualified-v2':cadence_suffix+='-stagev2'
        label=f"{seed}-{sha[:12]}{cadence_suffix}-{attempt}"
        directory=f'"$HOME/.local/share/toho-ml/th10-official-trial-002a/full-story/{label}"'
        endpoint=client["command"][:-1]
        body='dir='+directory+'\n'+script.replace("JOB",label)
        launch=('set -eu; dir='+directory+'; mkdir -p "$dir"; '
                'if mkdir "$dir/claimed" 2>/dev/null; then '
                "nohup sh -c "+shlex.quote("("+body+") ; rc=$?; echo $rc > "+directory+'/status')+
                ' </dev/null >"$dir/job.log" 2>&1 & fi')
        for connection_attempt in range(4):
            try:
                claimed=subprocess.run([*endpoint,'test -d '+directory+'/claimed'],stdin=subprocess.DEVNULL,capture_output=True,timeout=20)
                if claimed.returncode==0:break
                subprocess.run([*endpoint,launch],stdin=subprocess.DEVNULL,capture_output=True,check=True,timeout=25)
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
                                       stdin=subprocess.DEVNULL,capture_output=True,timeout=20)
                except (subprocess.TimeoutExpired,OSError):
                    time.sleep(4)
                    continue
                if qry.returncode==0 and qry.stdout.strip():break
                time.sleep(4)
            else:raise RuntimeError("Original-game credit reached infrastructure watchdog.")
            if qry.stdout.strip()!=b'0':
                err=subprocess.run([*endpoint,'dir='+directory+'; tail -n 20 -- "$dir/worker.log" "$dir/job.log"'],stdin=subprocess.DEVNULL,capture_output=True,timeout=12)
                (folder/f"worker-{attempt}.log").write_bytes(err.stdout+err.stderr)
                raise RuntimeError("Native worker exit "+qry.stdout.strip().decode())
            for transport_attempt in range(3):
                try:
                    archive=subprocess.run([*endpoint,'cat '+directory+'/behavior.tar.gz'],stdin=subprocess.DEVNULL,capture_output=True,check=True,timeout=90).stdout
                    break
                except (subprocess.SubprocessError,OSError):
                    if transport_attempt==2:raise
            row=unpack(archive,folder,host,seed,arm,sha,pinned,hold)
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
    p.add_argument("--camera",action="store_true",help="Deploy a hash-pinned TorchScript camera policy through the host's private Unix socket.")
    p.add_argument("--camera-history",action="store_true",help="Include the preceding sampled movement byte with each camera request.")
    p.add_argument("--difficulty",type=int,choices=(0,1),default=1,help="Select Easy (0) or Normal (1) in the original menu.")
    p.add_argument("--stage-sensor",choices=('legacy','qualified-v2'),
                   help='New runs use qualified-v2. Existing arms preserve their frozen sensor. Legacy Stage 2 endpoints are unvalidated.')
    p.add_argument("--only-host",choices=FLEET)
    p.add_argument("--prepare-only",action="store_true");p.add_argument("--arm",default="sft")
    p.add_argument("--out",type=Path,required=True);p.add_argument("--seeds",nargs="+",type=int)
    p.add_argument("--concurrency",type=int,default=4);p.add_argument("--hold",type=int,default=120)
    args=p.parse_args()
    workers=json.loads(args.workers.read_bytes())
    fleet=(args.only_host,) if args.only_host else FLEET
    args.out.mkdir(parents=True,exist_ok=True)
    prior_sensors={json.loads(p.read_bytes())['source'].get('stage_sensor','legacy') for p in args.out.glob('*/report.json')}
    lock=args.out/'stage-sensor-lock.json'
    if lock.is_file():prior_sensors.add(json.loads(lock.read_bytes())['stage_sensor'])
    if len(prior_sensors)>1:p.error('A frozen arm contains conflicting native stage sources.')
    if args.stage_sensor and prior_sensors and args.stage_sensor not in prior_sensors:
        p.error('Use a separate output directory for an independent stage-sensor re-evaluation.')
    args.stage_sensor=args.stage_sensor or next(iter(prior_sensors),'qualified-v2')
    if args.stage_sensor not in ('legacy','qualified-v2'):p.error('Invalid frozen stage-source lock.')
    with ThreadPoolExecutor(2) as pool:
        if (args.actor_header or args.locator_params or args.camera) and not args.expected_sha:p.error("Custom actors require --expected-sha")
        if args.camera and (args.actor_header or args.locator_params):p.error("The camera actor manages its own header.")
        if args.camera_history and not args.camera:p.error('--camera-history requires --camera')
        pins=dict(zip(fleet,pool.map(lambda h:pin(workers[h],args.paired,args.actor_header,args.locator_params,camera=args.camera,camera_history=args.camera_history,difficulty=args.difficulty,stage_sensor=args.stage_sensor),fleet)))
        if args.prepare_only:save(args.out/'builds.json',pins);return
        shas=list(pool.map(lambda h:install_weights(workers[h],args.model,args.expected_sha),fleet))
        if args.camera:list(pool.map(lambda h:start_camera_service(workers[h],pins[h],shas[0]),fleet))
    if len(set(shas))!=1 or len(args.seeds)!=len(set(args.seeds)):raise ValueError("Invalid seed assignment or model distribution.")
    plan=dict(arm=args.arm,policy_sha256=shas[0],holds_ms=args.hold,seeds=args.seeds,stage_sensor=args.stage_sensor,
              assignment={str(s):fleet[(s-1)%len(fleet)] for s in args.seeds},
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
