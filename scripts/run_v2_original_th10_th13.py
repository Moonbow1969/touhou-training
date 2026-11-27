"""Run a preregistered native SFT/PPO full-credit comparison on original TH10 or TH13."""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace
import torch

from scripts.collect_th10_full_action import save
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_cohort import checkpoint, collect, verify, digest
from scripts.run_th10_camera_pair import run as paired
from scripts.run_th13_accelerator_camera import run as original_th13
from scripts.summarize_th13_accelerator_pairs import summarize as summarize_th13
from scripts.score_original_camera_run import verified_th10_entry


def qualified_ledger(directories,host,seeds,output):
    records=[]
    for folder in directories:
        for seed in seeds:
            credit=folder/f'{host}-{seed}'
            row=json.loads((credit/'report.json').read_bytes())
            assert row['host']==host and row['seed']==seed and row['source']['bundle']=='418969c346d1f020'
            reached,trace,resources=verified_th10_entry(row,credit)
            if reached is None or not trace:
                raise ValueError(f'Unqualified native TH10 stage at {credit}')
            records.append(dict(arm=row['arm'],host=host,seed=seed,policy_sha256=row['policy_sha256'],
                source_bundle=row['source']['bundle'],stage_register=row['stage_register'],stage2=reached,
                stage2_spares=resources[0],stage2_power=resources[2],final_spares=row['final_lives'],
                terminal_power=row['final_power'],effective_strikes=len(row['successful_bombs']),
                native_rows=row['stage2_evidence']['native_rows'],steps=row['steps'],native_key_matches=row['native_inputs_matching'],
                behavior_sha256=row['behavior_sha256'],native_trace_sha256=trace,source=row['stage2_evidence']))
    save(output,records)


def run(root,connection,title,active,experiment_name='v2-online',reference=None,host='momentum'):
    root=root.resolve();connection=connection.resolve();torch.set_num_threads(2)
    experiment=root/experiment_name
    while not (experiment/'camera-protocol.json').is_file():time.sleep(3)
    protocol=json.loads((experiment/'camera-protocol.json').read_bytes())
    train=protocol['train_seeds'][str(title)];held=protocol['eval_seeds'][str(title)]
    sources={'v2-online':'v2-supervised/joint-v2-3000/policy.ts','attention-online':'spatial-policy-study/spatial-attention/policy.ts',
             'normal-large-batch':'v2-online/policy/policy.ts'}
    baseline=reference.resolve() if reference is not None else root/sources[experiment_name]
    candidate=experiment/'policy/policy.ts'
    ref_name=protocol['reference_arm'];new_name=protocol['candidate']
    out=experiment/f'native{title}';out.mkdir(exist_ok=True)
    if title==10:
        jobs=SimpleNamespace(host=host,workers=connection)
        qualified=experiment_name=='normal-large-batch' or reference is not None
        if qualified:
            jobs.stage_sensor='qualified-v2'
            if experiment_name=='normal-large-batch':
                assert train==list(range(1151001,1151033)) and held==list(range(1151051,1151075))
            report=json.loads((baseline.parent/'report.json').read_bytes())
            assert digest(baseline)==report.get('torchscript_sha256',report.get('policy_sha256'))
        model=checkpoint(baseline)
        collect(jobs,ref_name,baseline,train,out,active,True)
        if qualified:qualified_ledger([out],host,train,out/'train-qualified-stage-receipts.json')
        verify(out,baseline,model,train,'train',host=host)
        package(experiment/'th10-train-native-logs.zip',[out])
        left,right=('held-'+ref_name,'held-'+new_name) if qualified else ('held-sft','held-ppo')
        paired(SimpleNamespace(out=out,workers=connection,host=host,reference=baseline,
            stage_sensor='qualified-v2' if qualified else None,
            candidate=candidate,reference_name=left,candidate_name=right,
            seeds=held,concurrency=active,candidate_history=False))
        if qualified:
            assert digest(candidate)==json.loads((candidate.parent/'report.json').read_bytes())['policy_sha256']
            qualified_ledger([out/left,out/right],host,held,out/'qualified-stage-receipts.json')
        package(experiment/'th10-held-native-logs.zip',[out/left,out/right])
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
    parser.add_argument('--experiment',default='v2-online')
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--host',choices=('accelerator','momentum'),default='momentum')
    args=parser.parse_args()
    run(args.root,args.workers,args.title,args.lanes,args.experiment,args.reference,args.host)
