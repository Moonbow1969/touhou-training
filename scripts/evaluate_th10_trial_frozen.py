"""Evaluate frozen TH10 Normal policies on verified native Hard credits."""

import argparse
import gzip
import json
from pathlib import Path
from statistics import median

from stable_baselines3 import PPO
import torch

from scripts.checkpoint_storage import digest
from scripts.train_full_policy import publish_json
from scripts.train_th10_trial_live_ppo import ARMS, NativeTH10


def credit(worker, actor, seed, horizon, trace_path):
    with gzip.open(trace_path,"wt",encoding="utf-8") as trace:
        env = NativeTH10(worker,horizon,trace)
        torch.manual_seed(seed)
        try:
            observation,_ = env.reset()
            while True:
                action,_ = actor.predict(observation,deterministic=False)
                observation,_,terminated,truncated,info = env.step(action)
                if terminated or truncated:
                    return info
        finally:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker",type=Path,required=True)
    parser.add_argument("--normal-pilot",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--seeds",type=int,nargs="+",default=[8101,8102,8103])
    args=parser.parse_args()
    torch.set_num_threads(1)
    worker=json.loads(args.worker.read_text())
    src=args.normal_pilot/"report.json"
    original=json.loads(src.read_text())
    if (worker["difficulty"]!=2 or original["status"]!="complete" or len(args.seeds)<3
            or len(args.seeds)!=len(set(args.seeds)) or set(args.seeds)&set(original["eval_seeds"])):
        parser.error("Use the native Hard slot, a sealed Normal pilot, and fresh matched policy seeds.")
    args.out.mkdir(parents=True,exist_ok=False)
    actors, weights = {}, {}
    for arm in ARMS:
        path=args.normal_pilot/f"{arm}.zip"
        weights[arm]=digest(path)
        if weights[arm]!=original[f"{arm}_sha256"]:
            raise ValueError("A frozen Normal policy differs from the sealed benchmark.")
        actors[arm]=PPO.load(path,device="cpu")
    report=dict(status="evaluating",source_report_sha256=digest(src),actor_sha256=weights,
                title="TH10 Mountain of Faith",version="official trial 0.02a",host="accelerator",
                executable_sha256=worker["executable_sha256"],worker_sources=worker["sources"],
                expected_native_difficulty=2,setting="Hard Reimu A Stage 1",
                horizon=original["horizon"],frozen=True,shoot=1,bomb=0,
                input_evidence="Held is the XTest bridge acknowledgement. Native XY and lives are separately read from the game.",
                cadence="Nominal 85 ms resumed-wall action hold, not frame-locked.",assignments=args.seeds,
                runner_sha256=digest(Path(__file__)),trials=[],rejected=[])
    path=args.out/"report.json"
    publish_json(path,report)
    for block,seed in enumerate(args.seeds):
        for order in range(2):
            arm=ARMS[(block+order)%2]
            for attempt in range(1,4):
                shard=args.out/f"seed-{seed}-{arm}-attempt{attempt}.jsonl.gz"
                try:
                    result=credit(worker,actors[arm],seed,report["horizon"],shard)
                except (RuntimeError,OSError,ValueError) as error:
                    report["rejected"].append(dict(arm=arm,seed=seed,attempt=attempt,error=str(error),trace=shard.name))
                    publish_json(path,report)
                    if attempt==3:
                        raise
                else:
                    report["trials"].append(dict(arm=arm,seed=seed,order=order+1,trace=shard.name,trace_sha256=digest(shard),**result))
                    publish_json(path,report)
                    break
    report["summary"]={}
    for arm in ARMS:
        rows=[row for row in report["trials"] if row["arm"]==arm]
        report["summary"][arm]=dict(n=len(rows),first_misses=sum(row["first_miss"] for row in rows),
                                    median_decisions=median(row["decisions"] for row in rows),
                                    median_score=median(row["raw_score"] for row in rows),
                                    median_resumed_ms=median(row["resumed_wall_ms"] for row in rows))
    report["status"]="complete"
    publish_json(path,report)
    print(json.dumps(report["summary"],indent=2))


if __name__=="__main__":
    main()
