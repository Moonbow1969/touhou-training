"""Evaluate fixed Normal and mixed-rank PPO checkpoints against human SFT in native TH10."""
import argparse
import gzip
import json
from pathlib import Path

import torch
from stable_baselines3 import PPO

from scripts.checkpoint_storage import digest
from scripts.run_cross_title_th10 import SharedNative,actor
from scripts.train_full_policy import publish_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker",type=Path,required=True)
    p.add_argument("--sft",type=Path,required=True)
    p.add_argument("--normal",type=Path,required=True)
    p.add_argument("--mixed",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--seeds",type=int,nargs="+",required=True)
    p.add_argument("--horizon",type=int,default=220)
    p.add_argument("--arms",nargs="+",choices=["sft","normal_2048","mixed_2048","mixed_4096"],
                   default=["sft","normal_2048","mixed_2048","mixed_4096"])
    p.add_argument("--order-offset",type=int,default=0)
    args=p.parse_args()
    if not 128 <= args.horizon <= 1800 or len(set(args.seeds))!=len(args.seeds):
        p.error("Choose a supported decision horizon and distinct controller seeds.")
    if len(set(args.arms))!=len(args.arms) or len(args.arms)<2:
        p.error("Choose at least two distinct policy arms.")
    torch.set_num_threads(1)
    worker=json.loads(args.worker.read_bytes())
    args.out.mkdir(exist_ok=False)
    trials=[]
    args.out.joinpath("report.json").write_text("{}")
    report=dict(status="running",host=worker["host"],difficulty=worker["difficulty"],trials=trials,
                horizon=args.horizon,seed_blocks=args.seeds,checkpoints={},rejected=[],
                arms=args.arms,order_offset=args.order_offset)
    dest=args.out/"report.json"
    publish_json(dest,report)
    with gzip.open(args.out/"decisions.jsonl.gz","wt",encoding="utf-8") as trace:
        env=SharedNative(worker,args.horizon,trace)
        env.motion_sign=True
        arms=args.arms
        models={"sft":actor(env,613,args.sft)}
        report["checkpoints"]["sft"]=digest(args.sft)
        for name,folder,steps in [("normal_2048",args.normal,2048),("mixed_2048",args.mixed,2048),
                                  ("mixed_4096",args.mixed,4096)]:
            if name not in arms:
                continue
            j=json.loads((folder/"report.json").read_bytes())
            records=[r for r in j["updates"] if r["accepted_total"]==steps]
            if j["status"]!="complete" or len(records)!=1 or j["sft_sha256"]!=digest(args.sft):
                raise ValueError("Incomplete or unmatched PPO comparison.")
            rec=records[0]
            path=folder/rec["checkpoint"]
            if digest(path)!=rec["sha256"]:
                raise ValueError("PPO fingerprint changed.")
            report["checkpoints"][name]=rec["sha256"]
            models[name]=PPO.load(path,device="cpu")
        try:
            for block,seed in enumerate(args.seeds):
                for order in range(len(arms)):
                    arm=arms[(args.order_offset+block+order)%len(arms)]
                    for attempt in range(1,4):
                        try:
                            obs,initial=env.reset(seed=seed)
                            env.trace_tag=dict(arm=arm,seed=seed,order=order,attempt=attempt)
                            torch.manual_seed(seed)
                            while True:
                                act,_=models[arm].predict(obs,deterministic=False)
                                obs,_,term,trunc,info=env.step(act)
                                if term or trunc:
                                    break
                        except (RuntimeError,OSError,EOFError) as exc:
                            report["rejected"].append(dict(arm=arm,seed=seed,attempt=attempt,error=str(exc)))
                            publish_json(dest,report)
                            env.close()
                            if attempt==3:
                                raise
                            continue
                        trials.append(dict(arm=arm,seed=seed,order=order,initial=initial,**info))
                        publish_json(dest,report)
                        break
            report["status"]="complete"
        except Exception as exc:
            report.update(status="interrupted",error=str(exc))
            raise
        finally:
            env.close()
            publish_json(dest,report)


if __name__=="__main__":
    main()
