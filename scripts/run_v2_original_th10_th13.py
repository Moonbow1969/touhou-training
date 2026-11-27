"""Run a preregistered native SFT/PPO full-credit comparison on original TH10 or TH13."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch

from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_cohort import checkpoint, collect, verify
from scripts.run_th10_camera_pair import run as paired
from scripts.run_th13_accelerator_camera import run as original_th13
from scripts.summarize_th13_accelerator_pairs import summarize as summarize_th13


def run(root,connection,title,active,experiment_name='v2-online'):
    root=root.resolve();connection=connection.resolve();torch.set_num_threads(2)
    experiment=root/experiment_name
    while not (experiment/'camera-protocol.json').is_file():time.sleep(3)
    protocol=json.loads((experiment/'camera-protocol.json').read_bytes())
    train=protocol['train_seeds'][str(title)];held=protocol['eval_seeds'][str(title)]
    sources={'v2-online':'v2-supervised/joint-v2-3000/policy.ts','attention-online':'spatial-policy-study/spatial-attention/policy.ts'}
    baseline=root/sources[experiment_name];candidate=experiment/'policy/policy.ts'
    ref_name=protocol['reference_arm'];new_name=protocol['candidate']
    out=experiment/f'native{title}';out.mkdir(exist_ok=True)
    if title==10:
        jobs=SimpleNamespace(host='momentum',workers=connection)
        model=checkpoint(baseline)
        collect(jobs,ref_name,baseline,train,out,active,True)
        verify(out,baseline,model,train,'train',host='momentum')
        package(experiment/'th10-train-native-logs.zip',[out])
        paired(SimpleNamespace(out=out,workers=connection,host='momentum',reference=baseline,
            candidate=candidate,reference_name='held-sft',candidate_name='held-ppo',
            seeds=held,concurrency=active,candidate_history=False))
        package(experiment/'th10-held-native-logs.zip',[out/'held-sft',out/'held-ppo'])
        return
    qualification=root/'engineering/th13-accelerator'
    while not (qualification/'qualification.json').is_file():time.sleep(4)
    middle=len(held)//2
    for role,arm,path,seeds in [('train',ref_name,baseline,train),('held',ref_name,baseline,held[:middle]),
                                ('held',new_name,candidate,held[middle:]),('held',new_name,candidate,held[:middle]),
                                ('held',ref_name,baseline,held[middle:])]:
        while not (path.parent/'report.json').is_file():time.sleep(4)
        original_th13(SimpleNamespace(qualification=qualification,connection_config=connection,checkpoint=path,
            output=out,role=role,arm=arm,seeds=seeds,workers=active))
        for seed in seeds:
            folder=out/f'{role}-{arm}-{seed}'
            if not (folder/'steps.bin').is_file() or not (folder/'receipt.json').is_file():
                raise ValueError(f'Original TH13 raw clock or endpoint missing for {seed}')
    summarize_th13(root,experiment_name)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--workers',required=True,type=Path)
    parser.add_argument('--title',type=int,choices=(10,13),required=True)
    parser.add_argument('--lanes',type=int,default=3)
    parser.add_argument('--experiment',choices=('v2-online','attention-online'),default='v2-online')
    args=parser.parse_args()
    run(args.root,args.workers,args.title,args.lanes,args.experiment)
