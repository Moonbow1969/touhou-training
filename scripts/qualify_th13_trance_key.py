"""Qualify TH13 native C separately from ordinary inputs in a private running game."""
import argparse
import csv
import io
import json
from pathlib import Path
import shlex
import time
from types import SimpleNamespace
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
from scripts.run_th13_accelerator_camera import CameraPool,NATIVE


def run(root):
    root=root.resolve();out=root/"engineering/th13-trance";out.mkdir(parents=True,exist_ok=True)
    opts=SimpleNamespace(host="momentum",host_label="momentum",known_hosts=None,remote_root="/tmp/toho-v26-trance-e1",
        prior_bridge="/tmp/toho-th13-cadence-261125/bridge",assets="/tmp/toho-th13-261125/assets/app",launcher="/tmp/toho-v26-high-camera/src/launch.sh",
        image="toho-trial:endeavor-native-v6",python_executable="python3",checkpoint=root/"four-game-online/policy/policy.ts",
        output=out,role="engineering",arm="native-c",difficulty=2)
    pool=CameraPool(opts);pool.prepare();ident="trance-control-1311301";name,folder=pool.boot(ident,4)
    phases=[]
    try:
        pool.remote("cat >"+shlex.quote(folder+"/trance-sensor.c"),data=(NATIVE/"observe-trance-trial.c").read_bytes())
        pool.exec(name,"i686-w64-mingw32-gcc -Os -Wall -Werror /capture/trance-sensor.c -o /capture/trance-sensor.exe")
        pool.exec(name,"wine /capture/trance-sensor.exe",True)
        pool.exec(name,"xdotool keydown r; sleep .18; xdotool keyup r; sleep 4")
        def current():
            return list(csv.DictReader(io.StringIO(pool.remote("cat "+shlex.quote(folder+"/trance-observations.csv")).decode())))[-1]
        for i,key in enumerate((None,"c",None,"c",None,"Shift_L",None,"z",None,"c",None)):
            before=current()
            if key:pool.exec(name,"xdotool keydown "+key)
            time.sleep(.45)
            after=current()
            if key:pool.exec(name,"xdotool keyup "+key)
            phases.append(dict(key=key or "none",begin_tick=int(before["native_tick"]),end_tick=int(after["native_tick"])))
            if i in (0,1,3,9):pool.exec(name,f"scrot -o /capture/trance-control-{i:02}.png")
            time.sleep(.15)
        pool.exec(name,"touch /capture/trance-stop; sleep 1")
        (out/"c-key-original.csv").write_bytes(pool.remote("cat "+shlex.quote(folder+"/trance-observations.csv")))
        for i in (0,1,3,9):(out/f"c-key-control-{i:02}.png").write_bytes(pool.remote("cat "+shlex.quote(folder+f"/trance-control-{i:02}.png")))
        for source,filename in (("telemetry.bin","physical-probe-terminal-stock.bin"),("trance-sensor.c","physical-probe-sensor-source.c"),("difficulty.png","physical-probe-native-selected-rank.png")):
            (out/filename).write_bytes(pool.remote("cat "+shlex.quote(folder+"/"+source)))
        rows=list(csv.DictReader((out/"c-key-original.csv").open()))
        for phase in phases:
            subset=[r for r in rows if phase["begin_tick"]+4<=int(r["native_tick"])<=phase["end_tick"]-3]
            # A Restart can repeat the preflight tick. The phase target is the last running attempt.
            valid=[]
            for r in subset:
                if valid and int(r["native_tick"])<int(valid[-1]["native_tick"]):valid=[]
                valid.append(r)
            phase.update(steady_native_rows=len(valid),observed_words=sorted({int(r["native_input"]) for r in valid}),
                        candidate_gauge=sorted({int(r["gauge_candidate"]) for r in valid}),candidate_active=sorted({int(r["active_candidate"]) for r in valid}))
        save(out/"c-key-engineering.json",dict(official_trial_sha256="09818929cceea76ff4c65630e649bbccdb85d92b2e28bb4dfe6526749e6f4fb6",
            target_difficulty=2,phases=phases,observer_source_sha256=digest(NATIVE/"observe-trance-trial.c"),raw_sha256=digest(out/"c-key-original.csv"),
            scope="Controlled physical X11 key pulses in original Hard Reimu. Candidate resources require a separate successful activation.",engine_seed=1311301))
        print(phases)
    finally:pool.release(name)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);run(p.parse_args().root)
