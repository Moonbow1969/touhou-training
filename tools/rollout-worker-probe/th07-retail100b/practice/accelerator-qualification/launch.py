from pathlib import Path
import hashlib,json,os,subprocess,sys,time
r=Path(__file__).resolve().parent;base=r.parent
reg=r/"registration.json";p=json.loads(reg.read_bytes());tool=base/"tools";out=r/"2176503"
assert not out.exists()
for f,h in p["driver_sha256"].items():assert hashlib.sha256((r/f).read_bytes()).hexdigest()==h
policy=base.parent.parent/"moonbow-th07-trial-261127/policy/policy.ts"
assert hashlib.sha256(policy.read_bytes()).hexdigest()==p["policy_sha256"]
actor=tool/"practice-actor-v2";assert hashlib.sha256(actor.read_bytes()).hexdigest()==p["actor_sha256"]
env=dict(os.environ,TH07_PHYSICAL_LANE="practice-portability-only",TH07_PRACTICE_CPUSET="4,6,8,14")
args=[sys.executable,str(r/"practice_reset.py"),"--game-dir",str(base.parent/"input"),"--thprac",str(tool/"thprac.v2.3.1.1.exe"),"--probe",str(tool/"keyprobe"),"--actor",str(actor),"--out",str(out),"--image","toho-trial:accelerator-5b67741","--scenario","letty-2-3","--worker",sys.executable,str(r/"learned-worker.py"),str(policy),str(reg),"2176503"]
with (r/"run.log").open("x") as log,(r/"companion-cgroup.jsonl").open("x",buffering=1)as trace:
 job=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT);cg=None
 while job.poll() is None:
    if not cg and (out/"allocation.json").exists():
        name=json.loads((out/"allocation.json").read_bytes())["container"]
        pid=subprocess.run(["docker","inspect","--format","{{.State.Pid}}",name],capture_output=True,text=True)
        if pid.returncode==0 and pid.stdout.strip().isdigit() and int(pid.stdout)>0:
            path=Path("/proc")/pid.stdout.strip()/"cgroup"
            if path.exists():cg=Path("/sys/fs/cgroup")/path.read_text().strip().split(":")[-1].lstrip("/")
    if cg and cg.exists():
        try:
            print(json.dumps(dict(ns=time.monotonic_ns(),cpu=(cg/"cpu.stat").read_text(),pressure=(cg/"cpu.pressure").read_text(),io=Path("/proc/pressure/io").read_text())),file=trace)
        except FileNotFoundError:pass
    time.sleep(.25)
(r/"physical-status.json").write_text(json.dumps(dict(seed=2176503,role="engineering",returncode=job.returncode,attempted=1))+"\n")
