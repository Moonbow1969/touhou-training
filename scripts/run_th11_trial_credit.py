"""Run a frozen four-head camera policy on a complete, unmodified TH11 0.02a trial credit."""
import argparse
import csv
import gzip
import io
import json
from pathlib import Path
import shlex
import tarfile
import time
import numpy as np
from scripts.th11_trial_lane import Lane, SRC
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest

def start_actor(lane,sha,seed,rank):
    lane.remote("cd "+lane.dir+" && gcc -O2 -std=c99 camera-credit.c -o camera-credit -l:libX11.so.6 -l:libXtst.so.6 -lm")
    lane.inside("""cd /capture
test -x camera-credit
i686-w64-mingw32-gcc -O2 native-stock.c -o native-stock.exe
for i in $(seq 1 60); do xwininfo -root -tree | grep -q 208x154 && break; sleep .5; done
xdotool mousemove 427 364 click 1; sleep .12; xdotool mousemove 507 449 click 1
sleep 12
scrot -o /capture/title.png
wine /capture/native-stock.exe >native.log 2>&1 &
xdotool keydown z; sleep .18; xdotool keyup z; sleep 1.6
for i in $(seq 2 RANK); do xdotool keydown Down; sleep .14; xdotool keyup Down; sleep .6; done
scrot -o /capture/selected-rank.png
for screen in character weapon; do
 xdotool keydown z; sleep .18; xdotool keyup z; sleep 1.5; scrot -o /capture/$screen.png
done
xdotool keydown z; sleep .16; xdotool keyup z
for i in $(seq 1 50); do test -S actor.sock && test -s native.bin && break; sleep .1; done
(sleep 3; scrot -o /capture/opening.png) &
TOHO_CAMERA_SHA=POLICY_HASH TOHO_CAMERA_SOCKET=/capture/actor.sock ./camera-credit SEED RANK >actor.log 2>&1
sleep 4
touch /capture/stop-stock
sleep .08
scrot -o /capture/endpoint.png
gzip -c pixels.u8 >pixels.u8.gz
echo done >done
""".replace("POLICY_HASH",sha).replace("RANK",str(rank)).replace("SEED",str(seed)),detach=True)


def run(root,connection,seed,rank,*,role="engineering",arm="merged-sft",model=None,out=None,expected_sha=None):
    ident=f"camera-{seed}" if role=="engineering" else f"{role}-{arm}-{seed}"
    lane=Lane(root,connection,ident,out)
    lane.start()
    repo=Path(__file__).resolve().parents[1]
    camera=repo/"tools/full_action"
    model=model or root/"specialist-vs-shared/merged-equal/policy.ts"
    sha=digest(model)
    assert sha==(expected_sha or "1f7f1a6b73fee215c0142a9a27be8d1523d38baa9ad99fba3d37d4839a792b3a")
    sources=[SRC/"camera-credit.c",SRC/"native-stock.c",SRC/"launch-game.sh",camera/"camera_actor.h",camera/"camera_service.py",
             SRC.parent/"th13-pixels/x11_pixels.h"]
    for file in [*sources,model]:
        lane.remote("cat >"+shlex.quote(lane.dir+"/"+file.name),file.read_bytes() if file==model else file.read_bytes().replace(b"\r\n",b"\n"))
    boot="""cd {directory}
test ! -e actor.sock
nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 camera_service.py --model policy.ts --sha {sha} --directory {directory} > inference.log 2>&1 </dev/null & echo $! > inference.pid
""".format(directory=lane.dir,sha=sha)
    lane.remote(boot)
    start_actor(lane,sha,seed,rank)
    for i in range(600):
        if lane.remote("test ! -f "+lane.dir+"/done || echo done").strip():break
        time.sleep(2)
    else:raise TimeoutError(lane.name)
    return finish(lane,sha,seed,rank,sources,role,arm)


def finish(lane,sha,seed,rank,sources,role,arm):
    """Recover a finished original from its private lane without opening another credit."""
    if lane.remote("cd "+shlex.quote(lane.dir)+" && test -s done && sha256sum policy.ts").split()[0].decode()!=sha:
        raise ValueError('Completed native actor differs from the requested policy')
    names=["title.png","selected-rank.png","character.png","weapon.png","opening.png","endpoint.png","native.bin","pixels.u8.gz",
           "camera.ns","behavior.csv","inference.log","actor.log","actual-executable.sha256"]
    data=lane.remote("tar -czf - -C "+lane.dir+" "+" ".join(names),timeout=180)
    with tarfile.open(fileobj=io.BytesIO(data),mode="r:gz") as archive:
        for name in names:(lane.out/name).write_bytes(archive.extractfile(name).read())
    receipt=score(lane,sha,seed,rank,sources,names,role,arm)
    close(lane)
    return receipt


def score(lane,sha,seed,rank,sources,names,role="engineering",arm="merged-sft"):
    native=np.fromfile(lane.out/"native.bin","<i4").reshape(-1,12)
    rows=list(csv.DictReader((lane.out/"behavior.csv").open()))
    frames=np.frombuffer(gzip.decompress((lane.out/"pixels.u8.gz").read_bytes()),np.uint8).reshape(-1,112,96)
    assert len(rows)==len(frames)>10
    acts=np.array([[int(r[k]) for k in ("dir","focus","shoot","bomb")] for r in rows])
    logp=np.array([[float(r[k]) for k in ("logp_dir","logp_focus","logp_shoot","logp_bomb")] for r in rows],np.float32)
    assert np.isfinite(logp).all() and np.all((acts[:,0]>=0)&(acts[:,0]<=8))
    ok=np.array([r["mask"]==r["observed_mask"] for r in rows])
    t0=int(rows[0]["before_ms"]);t1=int(rows[-1]["after_ms"])
    elapsed=(native[:,0].view(np.uint32).astype(np.uint64)-t0)&0xffffffff
    original=native[elapsed<=((t1-t0)&0xffffffff)].astype(np.int64)
    original[:,0]&=0xffffffff
    assert len(original)>len(rows) and list(original[0,[3,4,5,6,7,8,9]])==[0,0,2,rank,1,0,20]
    assert np.all(original[:,6]==rank) and np.all(original[:,8]==0) and original[-1,5]==-1
    loss=original[1:][np.diff(original[:,5])<0]
    assert len(loss)>=3 and loss[-1,5]==-1 and (np.abs(np.diff(original[:,5]))<=1).all()
    saves=[dict(original_ms=int(a[0]-t0),new_spares=int(a[5]),power_units=int(a[2]),native_score=int(a[1])*10) for a in loss]
    changes=[]
    for before,after in zip(original[:-1],original[1:]):
        if after[5]==before[5] and after[2]<before[2]:
            changes.append(dict(time_ms=int(after[0]-t0),power_before=int(before[2]),power_after=int(after[2]),key=int(after[11])&251))
    # Native Power can change near a death; a same-life whole debit and a real X input
    # are required here. Leave ambiguous debits separately visible.
    casts=[a for a in changes if a["power_before"]-a["power_after"]==20 and a["key"]&2]
    clocks=np.loadtxt(lane.out/"camera.ns",dtype=np.int64)
    assert ok.mean()>.9 and len(clocks)==len(rows)
    settled=native[-1]
    assert list(settled[[5,6,8]])==[-1,rank,0] and (int(settled[0])%2**32-t1)%2**32>=3500
    receipt=dict(title=11,trial="0.02a",official_limit="Stages 1 through 3",host="momentum",role=role,arm=arm,terminal=True,outcome="game_over",
        seed=seed,difficulty=rank,character="Reimu A (Yukari)",policy_sha256=sha,full_credit=True,continues=0,
        terminal_evidence="Native stock exhausted and all controls released before the original Game Over screen.",
        opening_spares=2,terminal_spares=0,native_terminal_sentinel=-1,terminal_power_units=int(settled[2]),power_at_last_loss_units=int(original[-1,2]),
        power_units_per_level=20,score=int(original[-1,1])*10,opening_score=int(original[0,1])*10,
        original_graze=int(settled[10]),graze_at_last_loss=int(original[-1,10]),effective_confirmed_casts=len(casts),cast_events=casts,power_debits=changes,
        native_life_losses=saves,steps=len(rows),native_samples=len(original),decisions_per_second=(len(rows)-1)/(np.ptp(clocks)/1e9),
        real_wall_seconds=float(np.ptp(clocks)/1e9),unique_directions=np.unique(acts[:,0]).tolist(),
        directions=np.bincount(acts[:,0],minlength=9).tolist(),focus_shoot_bomb_held=acts[:,1:].sum(0).tolist(),
        native_input_matches=int(ok.sum()),native_input_total=len(ok),stage_values=np.unique(original[:,7]).tolist(),
        stage2_semantics="The study's stage2-qualification.json identifies the exact-build original Stage 1 to Stage 2 positive witness. qualified_th11_stage2.read_credit verifies this credit against its native stage stream.",
        source_sha256={f.name:digest(f) for f in sources},assets=json.loads((lane.out/"build.json").read_bytes()),
        evidence_sha256={name:digest(lane.out/name) for name in names},native_format="int32 little-endian [T,12]: ms score_div10 power_raw character shot spares rank stage_candidate continues raw_units_per_level graze consumed_input",
        camera="96x112 grayscale, original field crop 384x448 at X11 (35,38); absolute change from prior image. Ordinary XTest; four model heads [9,2,2,2]. Native telemetry used for evaluation and endpoint only.",
        reproducibility="No control over the native game's hidden RNG. Engineering seed controls action sampling only.")
    after_times=np.array([int(r["after_ms"]) for r in rows],np.int64)
    delta=np.diff(original[:,5])
    change=np.flatnonzero(delta)
    step=np.searchsorted(after_times,original[change+1,0],side="left")
    assert (step<len(rows)).all()
    reward=np.zeros(len(rows),np.float32)
    np.add.at(reward,step,delta[change])
    assert reward.sum()==-3 and reward[-1]==-1
    terminated=np.zeros(len(rows),bool);terminated[-1]=True
    receipt["reward_components"]={"gained_spares":int(delta[delta>0].sum()),"spent_lives":int(-delta[delta<0].sum()),"return":int(reward.sum())}
    np.savez_compressed(lane.out/"trajectory.npz",frames=frames,action=acts.astype("u1"),logp=logp,value=np.array([r["value"] for r in rows],np.float32),reward=reward,terminated=terminated,
                       wall_ns=clocks,native=original)
    receipt["trajectory_sha256"]=digest(lane.out/"trajectory.npz")
    save(lane.out/"receipt.json",receipt)
    print(json.dumps({k:receipt[k] for k in ("steps","native_input_matches","score","native_life_losses","effective_confirmed_casts","real_wall_seconds")}),flush=True)
    return receipt


def close(lane):
    # Each service starts in its own seed directory, including simultaneous rank lanes.
    lane.remote("cd "+shlex.quote(lane.dir)+""" && {
        p=$(cat inference.pid)
        case "$p" in ''|*[!0-9]*) exit 1;; esac
        if test -e "/proc/$p"; then
            test "$(readlink "/proc/$p/cwd")" = "$(pwd -P)" &&
            grep -aqF camera_service.py "/proc/$p/cmdline" && kill "$p"
        fi
    }""")
    lane.remote("docker stop "+lane.name)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("connection",type=Path);p.add_argument("seed",type=int);p.add_argument("rank",type=int,choices=(1,2,3))
    a=p.parse_args();run(a.root.resolve(),a.connection,a.seed,a.rank)
