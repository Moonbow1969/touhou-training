"""TH13 Hard and Lunatic camera SFT/PPO originals on a preregistered physical host."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
from scripts.run_th13_accelerator_camera import CameraPool
from scripts.run_th13_four_game_generation2 import archive,wait


def run(root):
    root=root.resolve();study=root/"human78-online";out=study/"native13";out.mkdir(parents=True,exist_ok=True)
    p=study/"camera-protocol.json";j=json.loads(p.read_bytes());ref,alt=j["reference_arm"],j["candidate"]
    source=root/"scaled-human-sft/new-64/policy/policy.ts";sha=digest(source);assert sha==j["behavior_sha256"]
    train=j["train_seeds"]["13"];held=j["eval_seeds"]["13"];ranks=j["rank_by_seed"]["13"]
    assert len(train)==16 and len(held)==24 and not set(train)&set(held)
    reg=dict(study_protocol_sha256=digest(p),seed_host={str(k):"momentum" for k in held},train_seed_host={str(k):"momentum" for k in train},
        physical_pairing="Hard and Lunatic by registered seed, same physical host both arms, split first-half versus second-half order within each rank.")
    flag=study/"th13-held-host-allocation.json"
    if flag.exists():assert json.loads(flag.read_bytes())==reg
    else:assert not any(out.glob("*/receipt.json"));save(flag,reg)
    def group(rank,role,arm,model,seeds):
        missing=[k for k in seeds if not (out/f"{role}-{arm}-{k}/receipt.json").exists()]
        if not missing:return
        pool=CameraPool(SimpleNamespace(host="momentum",host_label="momentum",known_hosts=None,remote_root=f"/tmp/toho-v26-human78/rank{rank}",
            prior_bridge="/tmp/toho-th13-cadence-261125/bridge",assets="/tmp/toho-th13-261125/assets/app",launcher="/tmp/toho-v26-high-camera/src/launch.sh",
            image="toho-trial:endeavor-native-v6",python_executable="python3",checkpoint=model,output=out,role=role,arm=arm,difficulty=rank))
        pool.prepare()
        with ThreadPoolExecutor(2) as lanes:list(lanes.map(pool.run_credit,missing))
    def trained(rank):group(rank,"train",ref,source,[k for k in train if ranks[str(k)]==rank])
    with ThreadPoolExecutor(2) as lanes:list(lanes.map(trained,(2,3)))
    archive(study,"train",train,[ref],reg["train_seed_host"],{ref:sha})
    model=study/"policy/policy.ts"
    def tested(rank):
        chosen=[k for k in held if ranks[str(k)]==rank]
        for arm,block in ((ref,chosen[:6]),(alt,chosen[6:]),(alt,chosen[:6]),(ref,chosen[6:])):
            if arm==alt:
                wait(model.parent/"report.json");assert digest(model)==json.loads((model.parent/"report.json").read_bytes())["policy_sha256"]
            group(rank,"held",arm,model if arm==alt else source,block)
    with ThreadPoolExecutor(2) as lanes:list(lanes.map(tested,(2,3)))
    archive(study,"held",held,[ref,alt],reg["seed_host"],{ref:sha,alt:digest(model)})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);run(p.parse_args().root)
