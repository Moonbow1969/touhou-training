"""Replay TRAIN sources in the unmodified TH13 trial and bracket sampled special input with native state and screens."""
import argparse
import csv
import json
import os
import shlex
import sys
import time
from pathlib import Path
import numpy as np


def run(workspace,name,stage,host,assets,image,rpy_folder=None,output=None,stop_tick=None):
    workspace=workspace.resolve()
    from scripts.th13_trance_replay_lane import ReplayLane
    from replays.inspect_rpy import action_streams
    from scripts.run_th10_camera_cohort import digest
    from scripts.collect_th10_full_action import save
    d=workspace/"outputs/touhou-vision-261126";study=d/"engineering/th13-trance";study.mkdir(exist_ok=True,parents=True)
    rows=json.loads((d/"human-trial13/frozen-source-splits.json").read_bytes())["sources"]
    source=next(r for r in rows if r["name"]==name);assert source["split"]=="train"
    rpy=(rpy_folder or workspace/"work/th13-human-vision/trial-archive")/name;assert digest(rpy)==source["sha256"]
    stream=dict(action_streams(rpy,13,version="0.01a"))[stage];word=np.frombuffer(stream,"<u2").reshape(-1,3)[:,0]
    special=word&0xa00;events=np.flatnonzero((special!=0)&np.r_[True,special[:-1]==0])
    assert len(events)
    targets=sorted({int(t+n) for t in events for n in (-18,5,30,100,350,650)})
    target_end=min(max(targets)+8,len(word)-180)
    if stop_tick is not None:target_end=min(target_end,stop_tick)
    target=output or study/f"replay-{rpy.stem}-s{stage}";target.mkdir(parents=True,exist_ok=True)
    code=workspace/"work/th14-rl/tools/rollout-worker-probe/th13-full/observe-trance-trial.c"
    meta=dict(replay_sha256=source["sha256"],source=source["source"],player_group=source["player_group"],split="train",
              stage=stage,shot=source["shot"],difficulty=source["difficulty"],event_source_ticks=events.tolist(),
              input_alignment="source index t equals consumed word after native stage tick t+1",observer_sha256=digest(code),
              source_candidates="0x800,0x200, and 0xA00 cases measured separately",raw_fields="Image base +0xb9094 (candidate meter), +0xb9098 (candidate active), +0xdf430 (native input)",
              witness_boot="Original replay menu and stage selector. No gameplay input is sent during the witness.")
    if (target/"witness.json").exists():raise ValueError("Choose a fresh output directory to preserve the original witness")
    save(target/"protocol.json",meta)
    lane=ReplayLane("trance-"+rpy.stem.replace("_","-")+f"-s{stage}",stage,host,assets,image)
    remote=lane.command
    lane.boot(rpy,source["sha256"])
    try:
        lane.select()
        remote("cat >"+shlex.quote(lane.remote+"/trance-sensor.c"),code.read_bytes())
        lane.run("i686-w64-mingw32-gcc -Os -Wall -Werror /capture/trance-sensor.c -o /capture/trance-sensor.exe")
        lane.run("wine /capture/trance-sensor.exe >/capture/trance-engineering.log 2>&1",detach=True)
        lane.tap("z")
        deadline=time.monotonic()+1150;next_target=0;screens=[]
        while time.monotonic()<deadline:
            latest=remote("tail -1 "+shlex.quote(lane.remote+"/trance-observations.csv")+" 2>/dev/null || true").decode().strip().split(",")
            if len(latest)!=11 or not latest[0].isdigit() or int(latest[1])!=stage:time.sleep(.12);continue
            t=int(latest[0])
            if next_target<len(targets) and t>=targets[next_target]:
                event=targets[next_target];file=f"original-{event}.png"
                lane.run("scrot -o /capture/"+file)
                end=remote("tail -1 "+shlex.quote(lane.remote+"/trance-observations.csv")).decode().strip().split(",")
                (target/file).write_bytes(lane.get(file))
                screens.append(dict(target=event,native_before=t,native_after=int(end[0]),file=file,sha256=digest(target/file)))
                next_target+=1
            if t>=target_end:break
            time.sleep(.16)
        else:raise TimeoutError("Original replay trace did not reach expected native tick")
        lane.run("touch /capture/trance-stop; sleep 1")
        p=target/"native.csv";p.write_bytes(lane.get("trance-observations.csv"))
        raw={int(r["native_tick"]):r for r in csv.DictReader(p.open()) if int(r["stage"])==stage and 0<int(r["native_tick"])<=len(word)}
        ev=[]
        for index in events:
            t=int(index)+1
            window=[raw[k] for k in sorted(raw) if t-10<=k<=t+25]
            if not window:continue
            ev.append(dict(source_index=int(index),source_word=int(word[index]),native_tick=t,rows=window))
        matches=sum(int(r["native_input"])==int(word[t-1]) for t,r in raw.items())
        result=dict(**meta,raw_sha256=digest(p),observed_ticks=len(raw),exact_full_word_matches=matches,
                    screens=screens,events=ev,game_sha256=lane.get("executable.sha256").decode().split()[0])
        save(target/"witness.json",result)
        print("WITNESS",target,len(raw),matches,[(e["native_tick"],e["source_word"]) for e in ev],flush=True)
    finally:lane.stop()


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("workspace",type=Path);p.add_argument("name");p.add_argument("stage",type=int)
    p.add_argument("--host",default="momentum");p.add_argument("--assets",default="/tmp/toho-th13-261125/assets/app");p.add_argument("--image",default="toho-trial:endeavor-native-v6")
    p.add_argument("--rpy-folder",type=Path);p.add_argument("--output",type=Path);p.add_argument("--stop-tick",type=int)
    a=p.parse_args();run(a.workspace,a.name,a.stage,a.host,a.assets,a.image,a.rpy_folder,a.output,a.stop_tick)
