"""Launch registered original TH13 trials from merged human SFT and a fresh joint PPO update."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
from scripts.run_th13_accelerator_camera import CameraPool, run as accelerator
from scripts.run_th13_four_game_generation2 import archive, wait


def run(root,workers,ssh_target,known_hosts):
    root=root.resolve();work=root/"four-game-merged-online";native=work/"native13";native.mkdir(exist_ok=True)
    register=work/"camera-protocol.json";plan=json.loads(register.read_bytes());ref,alt=plan["reference_arm"],plan["candidate"]
    source=root/"specialist-vs-shared/merged-equal/policy.ts";sha=digest(source);assert sha==plan["behavior_sha256"]
    previous=root/"four-game-online";old_alloc=json.loads((previous/"th13-held-host-allocation.json").read_bytes())
    train=plan["train_seeds"]["13"];held=plan["eval_seeds"]["13"];ranks=plan["rank_by_seed"]["13"]
    train_host={str(k):json.loads((previous/"native13"/f"train-four-sft-{k}"/"receipt.json").read_bytes())["host"] for k in train}
    assert set(train_host.values())=={"endeavour"} and len(train)==8 and len(held)==24
    prev=json.loads((previous/"camera-protocol.json").read_bytes())["eval_seeds"]["13"]
    hosts={str(k):old_alloc["seed_host"][str(old)] for k,old in zip(held,prev)}
    enrollment=dict(study_protocol_sha256=digest(register),train_seed_host=train_host,seed_host=hosts,
        physical_pairing="Fresh original action trajectories; training reuses the original controller seeds on the very same physical host as G1. For new held seeds the corresponding ordinal reuses G1's physical assignment. Both frozen arms run on that host, order counterbalanced within physical host and difficulty.")
    allocation=work/"th13-held-host-allocation.json"
    if allocation.is_file():assert json.loads(allocation.read_bytes())==enrollment
    else:
        assert not any(native.rglob("receipt.json"));save(allocation,enrollment)
    def collect(host,role,arm,path,seeds,rank):
        missing=[k for k in seeds if not (native/f"{role}-{arm}-{k}/receipt.json").is_file()]
        if not missing:return
        if host=="accelerator":
            accelerator(SimpleNamespace(qualification=root/"engineering/th13-accelerator",connection_config=workers,
                checkpoint=path,output=native,role=role,arm=arm,seeds=missing,workers=3,difficulty=rank))
        else:
            pool=CameraPool(SimpleNamespace(host=ssh_target,host_label="endeavour",known_hosts=known_hosts,
                remote_root=f"/tmp/toho-v26-merged-online/rank{rank}",prior_bridge=f"/tmp/toho-v26-quartet-camera/rank{rank}/bridge",
                assets="/tmp/toho-th13-261125/assets/app",launcher="/tmp/toho-v26-quartet-camera/src/launch.sh",
                image="toho-trial:endeavor-native-v6",python_executable="/tmp/toho-v26-torch/bin/python",checkpoint=path,
                output=native,role=role,arm=arm,difficulty=rank))
            pool.prepare()
            with ThreadPoolExecutor(2) as lanes:list(lanes.map(pool.run_credit,missing))
    def train_rank(rank):
        selected=[k for k in train if ranks[str(k)]==rank]
        collect("endeavour","train",ref,source,selected,rank)
    candidate=work/"policy/policy.ts"
    def host_run(host):
        if host=="endeavour":
            with ThreadPoolExecutor(2) as pool:list(pool.map(train_rank,(2,3)))
            archive(work,"train",train,[ref],train_host,{ref:sha})
        assigned=[s for s in held if hosts[str(s)]==host]
        for arm,i in ((ref,0),(alt,1),(alt,0),(ref,1)):
            if arm==alt:
                wait(candidate.parent/"report.json")
                assert digest(candidate)==json.loads((candidate.parent/"report.json").read_bytes())["policy_sha256"]
            path=source if arm==ref else candidate
            for rank in (2,3):
                selected=[k for k in assigned if ranks[str(k)]==rank];assert len(selected)==6
                collect(host,"held",arm,path,selected[i*3:(i+1)*3],rank)
    with ThreadPoolExecutor(2) as pool:list(pool.map(host_run,("endeavour","accelerator")))
    archive(work,"held",held,[ref,alt],hosts,{ref:sha,alt:digest(candidate)})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--workers",type=Path,required=True)
    p.add_argument("--ssh-target",required=True);p.add_argument("--known-hosts",type=Path,required=True)
    a=p.parse_args();run(a.root,a.workers.resolve(),a.ssh_target,a.known_hosts.resolve())
