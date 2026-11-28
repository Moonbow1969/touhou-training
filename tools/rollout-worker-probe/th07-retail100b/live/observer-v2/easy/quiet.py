from pathlib import Path
import json,os,subprocess,sys,time
from engineering import lane
root=Path(__file__).resolve().parents[2]/"record"
plan=json.loads((root/"protocol.json").read_bytes())
def stats():
    paths=["/proc/pressure/cpu","/proc/pressure/io","/proc/pressure/memory","/proc/diskstats","/proc/loadavg","/proc/stat"]
    return {k:Path(k).read_text().splitlines()[:1] if k=="/proc/stat" else Path(k).read_text() for k in paths}
def monitor():
    while True:
        print(json.dumps(dict(ns=time.monotonic_ns(),utc_ns=time.time_ns(),data=stats())),flush=True);time.sleep(1)
if "--monitor" in sys.argv:
    monitor()
else:
    with (root/"canary.csv").open("x") as fc,(root/"host.jsonl").open("x")as fh:
        canary=subprocess.Popen(["taskset","-c","20",str(root/"tool/canary")],stdout=fc)
        host=subprocess.Popen(["taskset","-c","21",sys.executable,"-u",__file__,"--monitor"],stdout=fh)
        results=[]
        try:
            for q in plan["lanes"]:
                try:results.append(lane(q))
                except Exception as exc:
                    import traceback
                    out=root/(str(q["seed"])+"-parent")
                    (out/"qualification-error.log").write_text(traceback.format_exc())
                    results.append(dict(seed=q["seed"],qualified=False,error=repr(exc)))
                    name=(out/"actual-container").read_text().strip() if (out/"actual-container").is_file() else ""
                    if name:
                        info=subprocess.run(["docker","inspect",name],capture_output=True)
                        if info.returncode==0:
                            m=json.loads(info.stdout)[0]["Mounts"]
                            if not any(x["Destination"]=="/capture" and Path(x["Source"])==out.resolve() for x in m):
                                raise RuntimeError("Unexpected active container mount")
                            subprocess.run(["docker","stop","-t","2",name],check=True)
                (root/"results-so-far.json").write_text(json.dumps(results,indent=2))
        finally:
            for p in [canary,host]:p.terminate();p.wait(timeout=4)
