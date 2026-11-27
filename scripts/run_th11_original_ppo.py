"""Schedule the first registered TH11 original PPO training and same-host held comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import digest
from scripts.run_th13_four_game_generation2 import wait
from scripts.run_th11_trial_credit import run as game

def run(root,workers):
    study=root/"th11-original-ppo";p=study/"camera-protocol.json"
    assert digest(p)=="22d31f0df059337d0a6b7f37f7a1d7cafb9724bcb49845f20d079a2ee27af6d2"
    plan=json.loads(p.read_bytes())
    ref,alt=plan["reference_arm"],plan["candidate"]
    train=plan["train_seeds"]["11"];held=plan["eval_seeds"]["11"];rankmap=plan["rank_by_seed"]["11"]
    assert len(train)==16 and len(held)==16 and not set(train)&set(held)
    native=study/"native11";native.mkdir(parents=True,exist_ok=True)
    reference=root/"specialist-vs-shared/merged-equal/policy.ts";candidate=study/"policy/policy.ts"
    allocation=dict(study_protocol_sha256=digest(p),seed_host={str(k):"momentum" for k in train+held},
        rank_lane={str(k):"th11-momentum-"+str(rankmap[str(k)]) for k in train+held},
        physical_pairing="One sequential workload per rank on the same physical host. Both arms use the same launcher, rank-specific queue and per-credit namespace. First four within-rank pairs reference-first, final four candidate-first.")
    dest=study/"th11-host-allocation.json"
    if dest.exists():assert json.loads(dest.read_bytes())==allocation
    else:assert not any(native.rglob("receipt.json"));save(dest,allocation)
    def jobs(rank):
        def fire(role,arm,model,seed):
            folder=native/f"{role}-{arm}-{seed}"
            sha=plan["behavior_sha256"] if arm==ref else json.loads((model.parent/"report.json").read_bytes())["policy_sha256"]
            if (folder/"receipt.json").exists():
                r=json.loads((folder/"receipt.json").read_bytes())
                assert r["policy_sha256"]==sha and r["seed"]==seed
                return
            r=game(root,workers,seed,rank,role=role,arm=arm,model=model,out=folder,expected_sha=sha)
            assert r["role"]==role and r["terminal"] and r["difficulty"]==rank
        for seed in train:
            if rankmap[str(seed)]==rank:fire("train",ref,reference,seed)
        selected=[k for k in held if rankmap[str(k)]==rank]
        for arm,block in [(ref,selected[:4]),(alt,selected[4:]),(alt,selected[:4]),(ref,selected[4:])]:
            if arm==alt:wait(candidate.parent/"report.json")
            for seed in block:fire("held",arm,candidate if arm==alt else reference,seed)
    with ThreadPoolExecutor(2) as pool:list(pool.map(jobs,(2,3)))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("root",type=Path);p.add_argument("workers",type=Path)
    a=p.parse_args();run(a.root.resolve(),a.workers.resolve())
