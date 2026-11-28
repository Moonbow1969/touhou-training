"""Serial, collision-safe driver for a registered TH07 practice roster."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import time

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser()
    for key in ("registration","game-dir","thprac","probe","actor","policy","root"):
        p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--host",required=True)
    p.add_argument("--lane",required=True)
    p.add_argument("--seed-allow",nargs="+",type=int)
    p.add_argument("--intake-index",required=True,type=Path)
    p.add_argument("--image",required=True)
    p.add_argument("--wsl")
    args=p.parse_args()
    out=args.root.resolve()
    reg=json.loads(args.registration.read_bytes())
    assert sha(args.actor)==reg["actor_sha256"] and sha(args.policy)==reg["policy_sha256"]
    files=Path(__file__).resolve().parent
    for name in ("practice_reset.py","learned-worker.py","scenario-glyphs.npz","start.sh"):
        assert sha(files/name)==reg["driver_sha256"][name]
    started=time.time()
    pinned=reg["pinned_lanes"][args.lane]
    os.environ["TH07_PRACTICE_CPUSET"]=pinned
    os.environ["TH07_PHYSICAL_LANE"]=args.lane
    for row in reg["rows"]:
        if args.seed_allow is not None and row["seed"] not in args.seed_allow:continue
        if row["host"]!=args.host or row["lane"]!=args.lane:continue
        if shutil.disk_usage(out).free < 2*1024**3:raise RuntimeError("Live disk guard: less than 2 GiB free")
        committed=sum(x.stat().st_size for x in (out/"raw").rglob("*") if x.is_file())
        if committed>1000*1024**2:raise RuntimeError("Study exceeds its 1 GiB raw-source budget")
        path=out/"raw"/str(row["seed"])
        if (path/"qualification.json").exists() or (path/"attempt-failed.json").exists():continue
        if path.exists() and args.seed_allow is not None:
            # A preceding dispatcher launched this original. Wait for the same owner to close.
            until=time.monotonic()+900
            while not (path/"qualification.json").exists() and not (path/"attempt-failed.json").exists():
                if time.monotonic()>until:raise RuntimeError(f"Unsealed previously dispatched original {row['seed']}")
                time.sleep(2)
            continue
        if path.exists():raise RuntimeError(f"Unsealed original {row['seed']} already owns this directory")
        path.parent.mkdir(parents=True,exist_ok=True)
        logfile=out/"logs"/(str(row["seed"])+".log")
        logfile.parent.mkdir(exist_ok=True)
        command=[sys.executable,str(files/"practice_reset.py"),"--game-dir",str(args.game_dir),"--thprac",str(args.thprac),
                 "--probe",str(args.probe),"--actor",str(args.actor),"--out",str(path),"--image",args.image,
                 "--scenario",row["scenario"]]
        if args.wsl:command+=["--wsl",args.wsl]
        command+=["--worker",sys.executable,str(files/"learned-worker.py"),str(args.policy),str(args.registration),str(row["seed"])]
        launched=time.time()
        (out/("dispatch-"+args.lane+".ndjson")).open("a").write(json.dumps(dict(seed=row["seed"],event="launch",time=launched))+"\n")
        with logfile.open("xb") as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            path.mkdir(exist_ok=True)
            failure=dict(seed=row["seed"],role=row["role"],scenario=row["scenario"],returncode=result.returncode,
                         start_epoch=launched,end_epoch=time.time(),log_sha256=sha(logfile),lossless_evidence_files={f.name:sha(f) for f in path.iterdir() if f.is_file()})
            (path/"attempt-failed.json").write_text(json.dumps(failure,indent=2)+"\n")
        (out/("dispatch-"+args.lane+".ndjson")).open("a").write(json.dumps(dict(seed=row["seed"],event="sealed" if result.returncode==0 else "failed",time=time.time()))+"\n")
        if row["seed"]==21749211:
            until=time.monotonic()+180
            while time.monotonic()<until:
                try: audit=next(r for r in json.loads(args.intake_index.read_bytes())["rows"] if r["seed"]==row["seed"])
                except (OSError,ValueError,StopIteration):time.sleep(2);continue
                if not audit["qualified"]:raise RuntimeError(f"Second private lane paused after its engineering intake: {audit}")
                break
            else:raise RuntimeError("Independent second-lane intake is pending")
    (out/("dispatch-complete-"+args.lane+".json")).write_text(json.dumps(dict(host=args.host,registration_sha256=sha(args.registration),seconds=time.time()-started))+"\n")
if __name__=="__main__":main()
