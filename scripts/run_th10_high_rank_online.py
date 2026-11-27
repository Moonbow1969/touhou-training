"""Collect a registered TH10 high-rank camera SFT→PPO cohort with original rank evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from scripts.collect_th10_full_action import save, DIFFICULTIES
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_cohort import collect, verify, checkpoint, digest
from scripts.run_th10_camera_pair import run as paired
from scripts.score_original_camera_run import verified_th10_entry


def native_ledger(folders,seeds,rank):
    records=[]
    menu=DIFFICULTIES[rank].lower()+'-rank.png'
    for folder in folders:
        for seed in seeds:
            credit=folder/f'momentum-{seed}';r=json.loads((credit/'report.json').read_bytes())
            assert r['difficulty']==rank and r['terminal'] and not r['continues']
            with (credit/'opening.csv').open() as f:opening=next(csv.DictReader(f))
            assert [int(opening[k]) for k in ('stage','difficulty','character','shot','lives','score')]==[1,rank,0,0,2,0]
            with gzip.open(credit/'native.csv.gz','rt') as f:native=list(csv.DictReader(f))
            assert len(native)==r['native_samples'] and {int(s['difficulty']) for s in native}=={rank}
            stage2,trace,stock=verified_th10_entry(r,credit)
            assert stage2 is not None and r['difficulty_menu_sha256']==digest(credit/menu)
            records.append(dict(seed=seed,arm=r['arm'],host=r['host'],difficulty=rank,native_rank_column='difficulty',
                rank_samples=len(native),opening_character=0,opening_shot=0,opening_spares=2,source_bundle=r['source']['bundle'],
                native_trace_sha256=trace,stage2=stage2,stage2_spares=stock[0],stage2_power=stock[2],
                terminal_spares=r['final_lives'],terminal_power=r['final_power'],effective_strikes=len(r['successful_bombs']),
                score=r['raw_score'],steps=r['steps'],native_input_matches=r['native_inputs_matching'],policy_sha256=r['policy_sha256'],
                original_menu_sha256=r['difficulty_menu_sha256'],behavior_sha256=r['behavior_sha256']))
    return records


def run(root,workers,lanes,experiment_name='high-rank-online',parallel_ranks=2):
    torch.set_num_threads(2)
    root=root.resolve();workers=workers.resolve();experiment=root/experiment_name;out=experiment/'native10'
    plan=json.loads((experiment/'camera-protocol.json').read_bytes());ref=plan['reference_arm'];alt=plan['candidate']
    models={'high-rank-online':('scaled-human-sft/cnn-new-24/policy/policy.ts','d6004d78f43c8b55262e24fd1d90cd99852d43826a788265a04d67a05f128573',16),
        'four-game-online':('ranked-multigame-sft/four-title/policy.ts','510132b2cd0cd04b728d066fe73e8703440c7e3cce1dcd165393da691ce76220',8)}
    source,expected_sha,n_train=models[experiment_name]
    path=root/source;candidate=experiment/'policy/policy.ts'
    intended=json.loads((path.parent/'report.json').read_bytes())['torchscript_sha256']
    assert digest(path)==intended==expected_sha
    train=plan['train_seeds']['10'];held=plan['eval_seeds']['10']
    assert len(set(train))==n_train and len(set(held))==24 and not set(train)&set(held)
    def group(rank,role):
        label=DIFFICULTIES[rank].lower();folder=out/label;folder.mkdir(exist_ok=True,parents=True)
        expected_parity=1 if rank==2 else 0
        seeds=[s for s in (train if role=='train' else held) if s%2==expected_parity]
        jobs=SimpleNamespace(workers=workers,host='momentum',difficulty=rank,stage_sensor='qualified-v2')
        if role=='train':
            assert len(seeds)==n_train//2
            collect(jobs,ref,path,seeds,folder,lanes,True)
            valid=native_ledger([folder],seeds,rank)
            receipts,_=verify(folder,path,checkpoint(path),seeds,'train',host='momentum',difficulty=rank)
        else:
            assert len(seeds)==12
            arms=('held-'+ref,'held-'+alt)
            paired(SimpleNamespace(out=folder,workers=workers,host='momentum',difficulty=rank,stage_sensor='qualified-v2',
                reference_name=arms[0],candidate_name=arms[1],reference=path,candidate=candidate,seeds=seeds,
                concurrency=lanes,candidate_history=False))
            assert digest(candidate)==json.loads((candidate.parent/'report.json').read_bytes())['policy_sha256']
            valid=native_ledger([folder/a for a in arms],seeds,rank);receipts=[]
        save(folder/(role+'-qualified-native-ranks.json'),valid)
        return valid,receipts
    with ThreadPoolExecutor(parallel_ranks) as pool:
        for role in ('train','held'):
            results=[f.result() for f in [pool.submit(group,rank,role) for rank in (2,3)]]
            save(out/(role+'-qualified-native-ranks.json'),[r for items,_ in results for r in items])
            if role=='train':
                save(out/'ppo-receipts.json',[r for _,items in results for r in items])
                package(experiment/'th10-train-native-logs.zip',[out/'hard',out/'lunatic'])
            else:
                for label in ('hard','lunatic'):
                    package(experiment/f'th10-{label}-held-native-logs.zip',[out/label/('held-'+ref),out/label/('held-'+alt)])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--workers',type=Path,required=True)
    parser.add_argument('--lanes-per-rank',type=int,default=1)
    parser.add_argument('--parallel-ranks',type=int,choices=(1,2),default=2)
    parser.add_argument('--experiment',choices=('high-rank-online','four-game-online'),default='high-rank-online')
    args=parser.parse_args();run(args.root,args.workers,args.lanes_per_rank,args.experiment,args.parallel_ranks)
