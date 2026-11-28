"""Sample the registered frozen camera model in a private, active TH07 practice game."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import numpy as np
import torch

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("policy",type=Path)
    p.add_argument("registration",type=Path)
    p.add_argument("seed",type=int)
    args=p.parse_args()
    registration=json.loads(args.registration.read_bytes())
    plan=next(row for row in registration["rows"] if row["seed"]==args.seed)
    assert digest(args.policy)==registration["policy_sha256"]
    assert plan["scenario"]==os.environ["TH07_SCENARIO"]
    spares,bombs=map(int,os.environ["TH07_STOCK"].split(","))
    out=Path(os.environ["TH07_EVIDENCE"])
    docker=(["wsl","-d",os.environ["TH07_WSL"],"--exec"] if os.environ["TH07_WSL"] else [])+["docker","exec"]
    name=os.environ["TH07_PRACTICE_CONTAINER"]
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model=torch.jit.load(str(args.policy),map_location="cpu").eval()
    rng=torch.Generator().manual_seed(args.seed)
    with torch.inference_mode():
        for _ in range(4):
            assert [v.numel() for v in model(torch.zeros(1,2,112,96))]==[9,2,2,2,1]
    err=(out/"actor-stderr.log").open("wb")
    proc=subprocess.Popen([*docker,"-i",name,"/practice-actor",str(spares),str(plan["decision_cap"])],
                          stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
    def exact(n):
        data=proc.stdout.read(n)
        if len(data)!=n:raise RuntimeError(f"Native worker sent {len(data)}/{n} bytes; status {proc.poll()}")
        return data
    frames=[];actions=[];logps=[];values=[];ticks=[];times=[]
    prev=None
    start=time.monotonic()
    (out/"worker-ready").touch()
    with (out/"controls.csv").open("x",newline="",buffering=1) as f,torch.inference_mode():
        wr=csv.writer(f)
        wr.writerow(("seq","tick","requested","spares_hud","bombs_hud","camera_ns","wall_s"))
        try:
            while True:
                kind,i,tick,word,shown_lives,shown_bombs,ns=struct.unpack("<6IQ",exact(32))
                assert i==len(actions)
                if kind in (2,5):break
                assert kind in (1,4) and tick%4==0
                image=np.frombuffer(exact(112*96),"u1").copy().reshape(112,96)
                x=torch.from_numpy(image).float()/255
                heads=model(torch.stack((x,torch.zeros_like(x) if prev is None else (x-prev).abs()))[None])
                if kind==4:
                    np.save(out/"boundary-frame.npy",image,allow_pickle=False)
                    boundary_value=float(heads[4].item())
                    break
                action=[int(torch.multinomial(head.softmax(-1),1,generator=rng)) for head in heads[:4]]
                move,focus,shoot,bomb=action
                mask=(16,0,32)[move//3]|(64,0,128)[move%3]|(4*focus)|shoot|(2*bomb)
                proc.stdin.write(struct.pack("<I",mask))
                proc.stdin.flush()
                wr.writerow((i,tick,mask,shown_lives,shown_bombs,ns,time.monotonic()-start))
                frames.append(image);actions.append(action);ticks.append(tick);times.append(ns)
                logps.append([float(h.log_softmax(-1)[0,a]) for h,a in zip(heads[:4],action)])
                values.append(float(heads[4].item()))
                prev=x
                if i in (32,64,120,240,360,540):
                    subprocess.Popen([*docker,name,"scrot","-o",f"/capture/original-spell-{i:04d}.png"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                if i==0:
                    subprocess.Popen([*docker,name,"scrot","-o","/capture/full-opening.png"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            proc.stdin.close()
            if proc.wait(timeout=12)!=0:raise RuntimeError("Original worker did not seal its native records")
        finally:
            if proc.poll() is None:proc.stdin.close()
            subprocess.run([*docker,name,"xdotool","keyup","Left","Right","Up","Down","Shift_L","z","x"],check=False)
            np.savez_compressed(out/"policy-trace.npz",frames=np.asarray(frames),action=np.asarray(actions),logp=np.asarray(logps),value=np.asarray(values),native_tick=np.asarray(ticks),camera_ns=np.asarray(times))
            err.close()
    endpoint=dict(**plan,sha256=registration["policy_sha256"],policy_sha256=registration["policy_sha256"],
                  registration_sha256=digest(args.registration),actions=len(actions),source_kind="practice",
                  native_rank=1,initial_spares=spares,initial_bombs=bombs,continues=0,last_tick=ticks[-1],
                  terminal_observed_tick=tick,wall_seconds=time.monotonic()-start)
    if kind==4:
        boundary=next(csv.DictReader((out/"native-cutoff.csv").open()))
        assert int(boundary["tick"])==tick and int(boundary["camera_ns"])==ns
        endpoint.update(outcome="native_tick_cutoff",boundary_tick=tick,boundary_stage=int(boundary["stage"]),
                        boundary_camera_ns=ns,boundary_value=boundary_value,boundary_frame_sha256=digest(out/"boundary-frame.npy"),
                        boundary_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),native_boundary=boundary)
    elif kind==5:
        source=next(csv.DictReader((out/"practice-result-state.csv").open()))
        assert int(source["supervisor_state"])==6 and int(source["last_tick"])==tick and int(source["retry"])==0 and int(source["continues"])==0
        endpoint.update(outcome="practice_stage_clear",practice_result=source,boundary_value=0.)
        subprocess.run([*docker,name,"scrot","-o","/capture/native-practice-result.png"],check=True)
    else:
        endpoint.update(outcome="game-over")
        subprocess.run([*docker,name,"scrot","-o","/capture/native-continue.png"],check=True)
    (out/"finished-game.json").write_text(json.dumps(endpoint,indent=2)+"\n",encoding="utf-8")

if __name__=="__main__":main()
