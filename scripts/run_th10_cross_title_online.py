"""Evaluate genuine multi-title human SFT and its direct visual PPO continuation on original TH10."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch

from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_cohort import verify,collect,checkpoint,digest
from scripts.run_th10_camera_pair import run as paired
from scripts.package_th10_credit_logs import package


def run(args):
    root=args.root.resolve();out=root/'cross-title-online/native10';out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    seeds=list(range(861101,861109));held=list(range(861201,861209))
    sft=root/'camera-cross-sft/policy.ts';ppo=root/'camera-cross-ppo/policy.ts'
    save(out/'cross-protocol.json',dict(title=10,host='momentum',training_seeds=seeds,held_seeds=held,
        train_actor='camera-cross-sft/policy.ts',held_actors=['camera-cross-sft/policy.ts','camera-cross-ppo/policy.ts'],
        control_hold_ms=120,experiment='Full original TH10 Normal trial from stock lives to Game Over/ending, all four sampled controls. No geometrical avoidance, aim correction, or automatic spirit strikes. Held seeds use counterbalanced order on one physical host.'))
    report=sft.parent/'report.json'
    while not report.is_file() or not (root/'camera-round2/native10/paired.json').is_file():time.sleep(4)
    model=checkpoint(sft)
    receipt=json.loads(report.read_bytes())
    assert receipt['torchscript_sha256']==digest(sft)
    jobs=SimpleNamespace(host='momentum',workers=args.workers.resolve())
    collect(jobs,'cross-sft-train',sft,seeds,out,3,True)
    verify(out,sft,model,seeds,'train',host='momentum')
    paired(SimpleNamespace(out=out,workers=jobs.workers,host='momentum',reference_name='held-sft',candidate_name='held-ppo',
        reference=sft,candidate=ppo,candidate_history=False,concurrency=3,seeds=held))
    package(out.parent/'th10-cross-title-native-logs.zip',[out,out/'held-sft',out/'held-ppo'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--workers',type=Path,required=True)
    run(p.parse_args())
