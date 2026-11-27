"""Drive genuine TH13 human input to an original tick, then collect a pure visual-policy suffix."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import numpy as np
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
from scripts.run_th13_accelerator_camera import CameraPool, NATIVE


def run(root,seed,target,training=None):
    assert 500 <= target <= 9000
    root=root.resolve();study=root/"engineering/th13-live-midstage-handoff"
    words=study/"taraka2-hard-reimu-s1-input-u16le.bin"
    assert digest(words)=="d8e38081c8fd646dff3bc95f99478a3b90a27ee27947d6c4d2883a73d27486b9"
    policy=root/"four-game-online/policy/policy.ts"
    assert digest(policy)=="3b26d1b7f8a8a27c5e271bc1e1eccf8e9c58d9f4c3186cc168efaacb3f07b82c"
    role="train" if training is not None else "engineering"
    tag=f"{role}-humanprefix-visual-{seed}"
    output=training.resolve() if training is not None else study/f"live-{seed}"
    folder=output/tag; folder.mkdir(parents=True,exist_ok=True)
    metadata=folder if training is not None else output
    opts=SimpleNamespace(host="momentum",host_label="momentum",known_hosts=None,
        remote_root=f"/tmp/toho-v26-midstage-prefix-{seed}",prior_bridge="/tmp/toho-th13-cadence-261125/bridge",
        assets="/tmp/toho-th13-261125/assets/app",launcher="/tmp/toho-v26-high-camera/src/launch.sh",
        image="toho-trial:endeavor-native-v6",python_executable="python3",checkpoint=policy,
        output=output,role=role,arm="humanprefix-visual",difficulty=2)
    class Pilot(CameraPool):
        def boot(self,ident,ticks):
            lib=self.o.remote_root+"/"+ident+"/a/lib"
            self.remote("mkdir -p "+shlex.quote(lib)+"; touch "+shlex.quote(lib+"/record-original-frames")+
                "; cat >"+shlex.quote(lib+"/human-prefix.u16"),data=words.read_bytes())
            name,folder=super().boot(ident,ticks)
            self.remote("docker update --cpus=1.75 --cpu-shares=256 "+shlex.quote(name)+" >/dev/null")
            return name,folder
    pilot=Pilot(opts);pilot.prepare()
    source=(NATIVE/"full-trial-actor.c").read_bytes().replace(b"\r\n",b"\n")
    header=(NATIVE/"human-prefix.h").read_bytes().replace(b"\r\n",b"\n")
    include=b"int main(int argc,char**argv){";guard=b" int64_t beginning=now_ns();"
    assert source.count(include)==source.count(guard)==1
    original=source
    source=source.replace(include,b'#include "human-prefix.h"\n'+include)
    ready=b" if(!native(fd,pre)||pre[4]!=2||pre[5]!=2||pre[3]!=1||pre[6]!=0||pre[7]!=0||pre[2]!=2)return 7;"
    assert source.count(ready)==1
    # The ordinary harness waits 3.9 wall seconds after Restart. An input
    # replay must attach at the native opening, including on faster hosts.
    source=source.replace(ready,b""" int opening_ready=0;
 for(int i=0;i<3000;i++){
   if(native(fd,pre)&&pre[4]==2&&pre[5]==2&&pre[3]==1&&pre[6]==0&&pre[7]==0&&pre[2]==2&&pre[8]==2&&pre[9]==0&&pre[11]>=0&&pre[11]<60){
     opening_ready=1;break;
   }
   hold_ms(3);
 }
 if(!opening_ready){fprintf(stderr,"missed native opening: stage=%d menu=%d rank=%d char=%d tick=%d spares=%d bombs=%d score=%d\\n",pre[3],pre[2],pre[8],pre[9],pre[11],pre[4],pre[5],pre[6]);return 7;}""")
    source=source.replace(guard,b" if(!engineering_prefix(d,fd,pre))return 19;\n"+guard)
    bridge=opts.remote_root+"/bridge"
    for name,body in (("human-prefix.h",header),("engineering-actor.c",source)):
        pilot.remote("cat >"+shlex.quote(bridge+"/"+name),data=body)
    pilot.remote("set -eu; cd "+shlex.quote(bridge)+f"; gcc -O2 -Wall -std=c11 -DTOHO_CAMERA_POLICY -DTOHO_PREFIX_TARGET={target} engineering-actor.c -o engineering-actor -l:libX11.so.6 -l:libXtst.so.6 -lm; cp engineering-actor actor")
    wrapper=(NATIVE/"collect-credit.sh").read_bytes().replace(b"\r\n",b"\n")
    old=b"sleep 3.9\nscrot -o initial.png"
    assert wrapper.count(old)==1
    wrapper=wrapper.replace(old,b"scrot -o initial.png &")
    pilot.remote("cat >"+shlex.quote(bridge+"/collect-credit.sh"),data=wrapper)
    ident=f"prefix-study-{seed}"
    path=opts.remote_root+"/"+ident+"/a/"+tag
    assert not (folder/"receipt.json").exists()
    save(metadata/"protocol.json",dict(role=("PPO training: post-handoff actions only" if training is not None else "engineering only; excluded from gradient updates and scored full-credit studies"),
        title=13,rank=2,registered_start=dict(stage=1,character=0,spares=2,bombs=2,continues=0),
        human_source="TRAIN Taraka2, original stage-1 Reimu Hard 0.01a replay",source_input_sha256=digest(words),
        target_original_stage1_tick=target,policy_sha256=digest(policy),controller_seed=seed,
        native_actor_source_sha256=hashlib.sha256(original).hexdigest(),engineering_actor_sha256=hashlib.sha256(source).hexdigest(),
        prefix_source_sha256=hashlib.sha256(header).hexdigest(),native_rng="Fresh original RNG, not set from replay"))
    failure=None
    try:
        pilot.attempt(ident,tag,folder,seed)
    except Exception as exc:
        failure=exc
    finally:
        for filename in ("prefix.csv","prefix-opening.png","handoff-original.png","observation-labels.bin","actor.log"):
            command="if test -f "+shlex.quote(path+"/"+filename)+"; then cat "+shlex.quote(path+"/"+filename)+"; fi"
            body=pilot.remote(command,timeout=90)
            if body:(folder/filename).write_bytes(body)
    if failure is not None:
        end=pilot.state(opts.remote_root+"/"+ident+"/a")
        raw=folder/"prefix.csv"
        lines=list(csv.DictReader(raw.open())) if raw.is_file() else []
        prefix_death=bool(end and lines and int(lines[-1]["stage_tick"])<target and end[4]==-1 and end[7]==0)
        qualified=dict(seed=seed,validated_handoff=False,technical_failure=not prefix_death,
               outcome="game_over_before_handoff" if prefix_death else "technical_failure",
               original_terminal_stock=list(end) if end else None,observed_prefix_ticks=len(lines),
               last_prefix=lines[-1] if lines else None,
               human_trace_sha256=digest(raw) if raw.exists() else None,error=str(failure))
        save(metadata/"qualification.json",qualified)
        if prefix_death:return qualified
        raise failure
    r=json.loads((folder/"receipt.json").read_bytes())
    with (folder/"prefix.csv").open() as f:prefix=list(csv.DictReader(f))
    data=np.fromfile(folder/"steps.bin","<f4").reshape(-1,143)
    with np.load(folder/"rollout.npz") as b:
        assert len(b["frames"])==len(data) and not b["terminated"][:-1].any() and b["terminated"][-1]
    initial,hand=prefix[0],prefix[-1]
    assert int(initial["spares"])==int(initial["bombs"])==2 and int(hand["stage_tick"])>=target
    assert {r["rank"] for r in prefix}=={"2"} and {r["character"] for r in prefix}=={"0"}
    assert (data[:,141]==2).all() and (data[:,132]==0).all()
    live=int(data[0,134]);assert target<=live<=int(hand["stage_tick"])+40
    directions=data[:,110].astype(int)
    result=dict(engineering_seed=seed,validated_handoff=True,actual_human_source_sha256=digest(words),
        prefix_opening=initial,handoff=hand,distinct_observed_prefix_ticks=len(prefix),
        observed_tick_coverage=len({int(x["stage_tick"]) for x in prefix})/(int(hand["stage_tick"])-min(int(x["stage_tick"]) for x in prefix)+1),
        native_human_input_agreement=sum(x["native_key"]==x["human_key_after_tick"] for x in prefix)/len(prefix),
        requested_key_x11_agreement=sum(x["requested_next_key"]==x["x11_key_after_request"] for x in prefix)/len(prefix),
        first_policy_state=dict(tick=live,stage=int(data[0,125]),spares=int(data[0,127]),bombs=int(data[0,129])),
        last_policy_state=dict(tick=int(data[-1,135]),stage=int(data[-1,126]),spares=int(data[-1,128]),bombs=int(data[-1,130])),
        stage2=bool(r["stage2_reached"]),continues=int(r["continues"]),post_handoff_decisions=len(data),
        native_action_matches=r["native_key_match"],post_handoff_directions=np.bincount(directions,minlength=9).tolist(),
        terminal_reason=r["native_outcome"] if "native_outcome" in r else r.get("outcome",r.get("terminal_reason")),
        native_trace_sha256=digest(folder/"steps.bin"),human_prefix_trace_sha256=digest(folder/"prefix.csv"),
        scope="One live original, using its own RNG. Exact human replay trajectory is not assumed. Neural actor begins with zero previous-image difference at handoff; four heads are sampled without assistance.")
    save(metadata/"qualification.json",result)
    print(json.dumps(result,indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--root",type=Path,required=True);p.add_argument("--seed",type=int,required=True)
    p.add_argument("--target-tick",type=int,default=5000)
    a=p.parse_args();run(a.root,a.seed,a.target_tick)
