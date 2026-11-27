"""Collect complete original Easy credits and test an Easy continuation of fourteen-player PPO."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch

from scripts.collect_th10_full_action import save
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_cohort import checkpoint,collect,digest,verify
from scripts.run_th10_camera_pair import run as paired


def run(root,workers):
    root=root.resolve();workers=workers.resolve();parent=root/'v2-online/policy/policy.ts'
    experiment=root/'easy-curriculum';out=experiment/'native10';out.mkdir(exist_ok=True,parents=True)
    candidate=experiment/'policy/policy.ts';torch.set_num_threads(2)
    jobs=SimpleNamespace(workers=workers,host='accelerator',difficulty=0)
    train=list(range(1101001,1101013));held=list(range(1101051,1101063))
    save(out/'native-protocol.json',dict(train_seeds=train,held_seeds=held,host='accelerator',difficulty=0,
        parent='v2-online/policy/policy.ts',candidate='easy-curriculum/policy/policy.ts',
        original='TH10 official trial Easy, full stocks, no Continue, 120ms control hold, original difficulty menu and native readbacks per credit.',
        actions='Frozen camera policy samples all nine directions, focus, shooting and Spirit Strike.'))
    model=checkpoint(parent)
    assert json.loads((parent.parent/'report.json').read_bytes())['policy_sha256']==digest(parent)
    collect(jobs,'v2-ppo',parent,train[:1],out,1,True)
    for start in range(1,len(train)):
        if (root/'v2-online/native10/paired.json').is_file():
            collect(jobs,'v2-ppo',parent,train[start:],out,2,True)
            break
        collect(jobs,'v2-ppo',parent,train[start:start+1],out,1,True)
    verify(out,parent,model,train,'train',host=jobs.host,difficulty=0)
    package(experiment/'th10-train-native-logs.zip',[out])
    while not (candidate.parent/'report.json').is_file():time.sleep(4)
    assert json.loads((candidate.parent/'report.json').read_bytes())['policy_sha256']==digest(candidate)
    paired(SimpleNamespace(out=out,workers=workers,host=jobs.host,difficulty=0,
        reference_name='held-v2-ppo',candidate_name='held-curriculum-ppo',
        reference=parent,candidate=candidate,seeds=held,concurrency=2,candidate_history=False))
    package(experiment/'th10-held-native-logs.zip',[out/'held-v2-ppo',out/'held-curriculum-ppo'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--workers',required=True,type=Path)
    args=p.parse_args();run(args.root,args.workers)
