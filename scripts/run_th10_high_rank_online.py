"""Collect a registered TH10 high-rank camera SFT→PPO cohort with original rank evidence."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import json
from pathlib import Path
from threading import Event
import time
from types import SimpleNamespace

import torch
from scripts.collect_th10_full_action import save, DIFFICULTIES
from scripts.package_th10_credit_logs import package
from scripts.run_th10_camera_cohort import collect, verify, checkpoint, digest
from scripts.run_th10_camera_pair import run as paired
from scripts.score_original_camera_run import verified_th10_entry


def native_ledger(folders,seeds,rank,host="momentum"):
    records=[]
    menu=DIFFICULTIES[rank].lower()+'-rank.png'
    for folder in folders:
        for seed in seeds:
            credit=folder/f'{host}-{seed}';r=json.loads((credit/'report.json').read_bytes())
            assert r['difficulty']==rank and r['terminal'] and not r['continues']
            with (credit/'opening.csv').open() as f:opening=next(csv.DictReader(f))
            assert [int(opening[k]) for k in ('stage','difficulty','character','shot','lives','score')]==[1,rank,0,0,2,0]
            with gzip.open(credit/'native.csv.gz','rt') as f:native=list(csv.DictReader(f))
            assert len(native)==r['native_samples'] and {int(s['difficulty']) for s in native}=={rank}
            stage2,trace,stock=verified_th10_entry(r,credit)
            assert stage2 is not None and r['difficulty_menu_sha256']==digest(credit/menu)
            timing={}
            if r.get('native_stage_clock'):
                from scripts.th10_native_stage_clock import verify as verify_clock
                timing=dict(last_observed_stage1_tick=verify_clock(credit,r),first_loss_tick=r['native_stage_clock']['first_life_loss_tick'],
                            stage_clock_sha256=r['native_stage_clock']['data_sha256'])
            records.append(dict(seed=seed,arm=r['arm'],host=r['host'],difficulty=rank,native_rank_column='difficulty',
                rank_samples=len(native),opening_character=0,opening_shot=0,opening_spares=2,source_bundle=r['source']['bundle'],
                native_trace_sha256=trace,stage2=stage2,stage2_spares=stock[0],stage2_power=stock[2],
                terminal_spares=r['final_lives'],terminal_power=r['final_power'],effective_strikes=len(r['successful_bombs']),
                score=r['raw_score'],steps=r['steps'],native_input_matches=r['native_inputs_matching'],policy_sha256=r['policy_sha256'],
                original_menu_sha256=r['difficulty_menu_sha256'],behavior_sha256=r['behavior_sha256'],**timing))
    return records


def run(root,workers,lanes,experiment_name='high-rank-online',parallel_ranks=2,total_held_lanes=None,
        reserve_until=None,reserved_lanes=0,host="momentum"):
    torch.set_num_threads(2)
    root=root.resolve();workers=workers.resolve();experiment=root/experiment_name;out=experiment/'native10'
    plan=json.loads((experiment/'camera-protocol.json').read_bytes());ref=plan['reference_arm'];alt=plan['candidate']
    models={'human78-online':('scaled-human-sft/new-64/policy/policy.ts','5fa285072ead0cc62cae6c6f1e5051e12a40e1e96068388c9360119a29113bb2',16),
        'high-rank-online':('scaled-human-sft/cnn-new-24/policy/policy.ts','d6004d78f43c8b55262e24fd1d90cd99852d43826a788265a04d67a05f128573',16),
        'four-game-online':('ranked-multigame-sft/four-title/policy.ts','510132b2cd0cd04b728d066fe73e8703440c7e3cce1dcd165393da691ce76220',8),
        'four-game-merged-online':('specialist-vs-shared/merged-equal/policy.ts','1f7f1a6b73fee215c0142a9a27be8d1523d38baa9ad99fba3d37d4839a792b3a',8),
        'four-game-ppo-generation2':('four-game-online/policy/policy.ts','3b26d1b7f8a8a27c5e271bc1e1eccf8e9c58d9f4c3186cc168efaacb3f07b82c',16)}
    source,expected_sha,n_train=models[experiment_name]
    path=root/source;candidate=experiment/'policy/policy.ts'
    metadata=json.loads((path.parent/'report.json').read_bytes())
    intended=metadata.get('policy_sha256') or metadata['torchscript_sha256']
    sensor='qualified-v2-clock' if experiment_name in ('four-game-ppo-generation2','four-game-merged-online','human78-online') else 'qualified-v2'
    assert digest(path)==intended==expected_sha
    train=plan['train_seeds']['10'];held=plan['eval_seeds']['10']
    assert len(set(train))==n_train and len(set(held))==24 and not set(train)&set(held)
    finished={rank:Event() for rank in (2,3)}
    def frozen_candidate(path):
        report=path.parent/'report.json'
        while not report.is_file():time.sleep(4)
        if digest(path)!=json.loads(report.read_bytes())['policy_sha256']:
            raise ValueError('Candidate bytes differ from the frozen PPO receipt')
    if total_held_lanes is not None:
        assert parallel_ranks==2 and total_held_lanes>=2
        if reserve_until is not None:
            assert 0<reserved_lanes<=total_held_lanes-2
            reserve_until=reserve_until.resolve()
        if candidate.is_file():
            assert digest(candidate)==json.loads((candidate.parent/'report.json').read_bytes())['policy_sha256']
    def group(rank,role):
        label=DIFFICULTIES[rank].lower();folder=out/label;folder.mkdir(exist_ok=True,parents=True)
        expected_parity=1 if rank==2 else 0
        seeds=[s for s in (train if role=='train' else held) if s%2==expected_parity]
        jobs=SimpleNamespace(workers=workers,host=host,difficulty=rank,stage_sensor=sensor)
        if role=='train':
            assert len(seeds)==n_train//2
            collect(jobs,ref,path,seeds,folder,lanes,True)
            valid=native_ledger([folder],seeds,rank,host)
            receipts,_=verify(folder,path,checkpoint(path),seeds,'train',host=host,difficulty=rank)
        else:
            assert len(seeds)==12
            arms=('held-'+ref,'held-'+alt)
            options=SimpleNamespace(out=folder,workers=workers,host=host,difficulty=rank,stage_sensor=sensor,
                reference_name=arms[0],candidate_name=arms[1],reference=path,candidate=candidate,seeds=seeds,
                concurrency=lanes,candidate_history=False,validate_candidate=frozen_candidate)
            if total_held_lanes is not None:
                options.concurrency=(total_held_lanes+1)//2 if rank==2 else total_held_lanes//2
                def block_capacity():
                    available=total_held_lanes
                    if reserve_until is not None and not reserve_until.is_file():available-=reserved_lanes
                    if finished[5-rank].is_set():return available
                    return (available+1)//2 if rank==2 else available//2
                options.block_capacity=block_capacity
            paired(options)
            finished[rank].set()
            assert digest(candidate)==json.loads((candidate.parent/'report.json').read_bytes())['policy_sha256']
            valid=native_ledger([folder/a for a in arms],seeds,rank,host);receipts=[]
        save(folder/(role+'-qualified-native-ranks.json'),valid)
        return valid,receipts
    with ThreadPoolExecutor(parallel_ranks) as pool:
        for role in ('train','held'):
            if role=='held' and parallel_ranks==1 and not candidate.is_file():
                # Both strata have six registered reference-first pairs. Fill them
                # on the single reserved worker while the joint learner is pending.
                for rank in (2,3):
                    seeds=[s for s in held if 3-s%2==rank][:6]
                    target=out/DIFFICULTIES[rank].lower()/('held-'+ref)
                    jobs=SimpleNamespace(workers=workers,host=host,difficulty=rank,stage_sensor=sensor)
                    collect(jobs,'held-'+ref,path,seeds,target,lanes,True)
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
    parser.add_argument('--total-held-lanes',type=int)
    parser.add_argument('--reserve-until',type=Path);parser.add_argument('--reserved-lanes',type=int,default=0)
    parser.add_argument('--experiment',choices=('high-rank-online','four-game-online','four-game-ppo-generation2','four-game-merged-online','human78-online'),default='high-rank-online')
    parser.add_argument("--host",choices=("momentum","accelerator"),default="momentum")
    args=parser.parse_args();run(args.root,args.workers,args.lanes_per_rank,args.experiment,args.parallel_ranks,
                                args.total_held_lanes,args.reserve_until,args.reserved_lanes,args.host)
