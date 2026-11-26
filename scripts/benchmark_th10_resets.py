"""Compare original-game Retry with a fresh TH10 Wine client on the same private machine."""
import argparse
import gzip
import json
from pathlib import Path
from statistics import median

from scripts.train_full_policy import publish_json
from scripts.train_th10_trial_live_ppo import NativeTH10


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    worker=json.loads(args.worker.read_bytes())
    args.out.mkdir(exist_ok=False)
    rows=[]
    report=dict(status="running",executable_sha256=worker["executable_sha256"],sources=worker["sources"],resets=rows)
    path=args.out/"report.json"
    publish_json(path,report)
    with gzip.open(args.out/"steps.jsonl.gz","wt",encoding="utf-8") as trace:
        env=NativeTH10(dict(worker,native_retry=True),230,trace)
        try:
            for idx in range(6):
                request_warm=bool(idx%2)
                if not request_warm:
                    env.close()
                _,initial=env.reset()
                if abs(env.state["position"][0])>.1 or abs(env.state["position"][1]-400)>1:
                    raise RuntimeError("Initial TH10 spawn disagrees with the original trial.")
                row=dict(index=idx+1,warm_requested=request_warm,**initial)
                before=env.state["position"][0]
                # Exercise a full first-life prefix in one pair to include the native death animation.
                count=220 if idx==2 else 36
                for i in range(count):
                    action=(5,0) if i<12 else (3,0) if i<24 else (4,0)
                    _,_,term,trunc,info=env.step(action)
                    if i==11:
                        row["right_displacement"]=env.state["position"][0]-before
                    if term or trunc:
                        break
                row.update(last_score=env.state["score"],last_x=env.state["position"][0],
                           last_lives=env.state["lives"],prefix_steps=env.count,outcome=info.get("outcome"))
                rows.append(row)
                publish_json(path,report)
        finally:
            env.close()
    report["status"]="complete"
    cold=[r["reset_seconds"] for r in rows if not r["warm_requested"]]
    warm=[r["reset_seconds"] for r in rows if r["native_retry"]]
    report.update(median_cold_seconds=median(cold),qualified_warm=len(warm),
                  median_qualified_warm_seconds=median(warm) if warm else None)
    publish_json(path,report)


if __name__=="__main__":
    main()
