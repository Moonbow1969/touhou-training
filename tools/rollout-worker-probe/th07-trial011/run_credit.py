import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

base=Path("/var/tmp/moonbow-th07-trial-261127")
study=base/"locator-bc/native"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True)
    parser.add_argument("--arm",choices=["action","action-xy"],required=True)
    parser.add_argument("--role",choices=["engineering","held"],required=True)
    parser.add_argument("--lane",choices=["0","1"],required=True)
    parser.add_argument("--engineering-revision",default="v1")
    a=parser.parse_args()
    p=json.loads((study/"protocol.json").read_text())
    if a.role=="engineering":
        assert a.seed==p["qualification_seeds"][int(a.lane)]
        assert a.engineering_revision.isalnum()
    else:
        gate=json.loads((study/"qualification.json").read_text())
        assert gate["passed"] and gate["protocol_sha256"]==sha(study/"protocol.json")
        assert str(a.seed) in p["seed_lane"] and int(a.lane)==p["seed_lane"][str(a.seed)]
    policy=base/f"locator-bc/fit/71281-{a.arm}/policy.ts"
    assert sha(policy)==p["models"][a.arm]
    for file,expected in p["install"].items():assert sha(base/file)==expected
    image=subprocess.check_output(["docker","image","inspect","toho-trial:endeavor-native-v6","--format","{{.Id}}"],text=True).strip()
    assert image==p["docker_id"]
    role=a.role if a.role=="held" else f"engineering-{a.engineering_revision}"
    folder=study/f"{role}-{a.seed}-{a.arm}"
    folder.mkdir(exist_ok=False)
    name=f"moonbow-th07-loc-{role}-{a.seed}-{a.arm}"
    run=["docker","run","-d","--rm","--name",name,"--network","none","--cpus","2","--memory","1800m","--pids-limit","300","--cap-add","SYS_PTRACE",
         "--tmpfs","/tmp:rw,exec,size=1300m","-v",f"{base}/input/youmu:/input:ro","-v",f"{base}/input/launch.sh:/start.sh:ro","-v",f"{folder}:/capture",
         "-v",f"{study}/tool:/tool:ro","-e","DISPLAY=:73","--entrypoint","sh",image,"/start.sh"]
    with (folder/"runner.log").open("w") as log:
        subprocess.run(run,check=True,stdout=log,stderr=subprocess.STDOUT)
        env=dict(os.environ,TH07_OUT=str(folder),TH07_SEED=str(a.seed),TH07_MODEL=str(policy),TH07_SHA=sha(policy),TH07_CONTAINER=name,
                 TH07_VIDEO=str(int(a.seed==p["first_held_seed"])))
        try:
            subprocess.run(["python3","-u",str(study/"tool/native_locator_credit.py")],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1200,check=True)
        finally:
            subprocess.run(["docker","stop","-t","2",name],stdout=log,stderr=subprocess.STDOUT)
if __name__=="__main__":
    main()
