"""Serial, collision-safe driver for a registered TH07 practice roster."""
import argparse
import hashlib
import json
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
    for row in reg["rows"]:
        if row["host"]!=args.host:continue
        path=out/"raw"/str(row["seed"])
        if (path/"qualification.json").exists() or (path/"attempt-failed.json").exists():continue
        if path.exists():raise RuntimeError(f"Existing incomplete original {row['seed']}; verify before any further start")
        path.parent.mkdir(parents=True,exist_ok=True)
        logfile=out/"logs"/(str(row["seed"])+".log")
        logfile.parent.mkdir(exist_ok=True)
        command=[sys.executable,str(files/"practice_reset.py"),"--game-dir",str(args.game_dir),"--thprac",str(args.thprac),
                 "--probe",str(args.probe),"--actor",str(args.actor),"--out",str(path),"--image",args.image,
                 "--scenario",row["scenario"]]
        if args.wsl:command+=["--wsl",args.wsl]
        command+=["--worker",sys.executable,str(files/"learned-worker.py"),str(args.policy),str(args.registration),str(row["seed"])]
        launched=time.time()
        (out/"dispatch.ndjson").open("a").write(json.dumps(dict(seed=row["seed"],event="launch",time=launched))+"\n")
        with logfile.open("xb") as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            path.mkdir(exist_ok=True)
            failure=dict(seed=row["seed"],role=row["role"],scenario=row["scenario"],returncode=result.returncode,
                         start_epoch=launched,end_epoch=time.time(),log_sha256=sha(logfile),lossless_evidence_files={f.name:sha(f) for f in path.iterdir() if f.is_file()})
            (path/"attempt-failed.json").write_text(json.dumps(failure,indent=2)+"\n")
        (out/"dispatch.ndjson").open("a").write(json.dumps(dict(seed=row["seed"],event="sealed" if result.returncode==0 else "failed",time=time.time()))+"\n")
    (out/"dispatch-complete.json").write_text(json.dumps(dict(host=args.host,registration_sha256=sha(args.registration),seconds=time.time()-started))+"\n")
if __name__=="__main__":main()
