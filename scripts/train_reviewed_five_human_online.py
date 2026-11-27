"""Register and fit ordinary five-title PPO from the 84-stage human camera policy."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

from scripts.checkpoint_storage import digest
from scripts.train_five_original_games import freeze
from scripts.score_original_camera_run import records


def read(path):return json.loads(path.read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    a=parser.parse_args();d=a.data.resolve();name='reviewed-five-human-online'
    root=d/name;root.mkdir(exist_ok=True)
    source=d/'first-five-title-retail-sft/five-reviewed'
    sft=read(source/'report.json');actor=digest(source/'policy.ts')
    if (actor!=sft['torchscript_sha256'] or digest(source/'camera-policy.pt')!=sft['sft'] or len(sft['train'])!=84 or
        sum(row['decisions'] for row in sft['train'])!=284222):raise ValueError('Unexpected SFT source')
    prior=read(d/'five-game-generation3/camera-protocol.json')
    games=(10,11,12,13,17)
    attempts={str(t):[1840000+t*100+i for i in range(1,25 if t==12 else 17)] for t in games}
    held={str(t):[1840000+t*100+i for i in range(51,67)] for t in games}
    plan=dict(name='Full-action five-title PPO from the reviewed 84-stage human SFT',reference_arm='reviewed-sft',candidate='reviewed-ppo',
              behavior_sha256=actor,source_sft_report=digest(source/'report.json'),games=prior['games'],train_seeds=attempts,eval_seeds=held,
              rank_by_seed={k:{str(v):2 if v%2 else 3 for v in values+held[k]} for k,values in attempts.items()},
              balanced_title_rank=True,native_executable_sha256=prior['native_executable_sha256'],verify_th10_playable_end=True,
              ppo=prior['ppo'],learning_rates=prior['learning_rates'],controls=prior['controls'],
              training='Freeze the initial 84-stage SFT and initialize a fresh zero critic. Use all 16 ordinary complete original starts per title for TH10/11/13/17. TH12 reserves 24 physical starts. Select the first eight strictly covered controller seeds numerically on each difficulty. If fewer qualify, use the same smaller number on each TH12 rank. Keep every physical endpoint and its coverage status. The other four games retain all original starts. Verify zero initial native behavior values and every received action log probability before optimization.',
              evaluation='Eight matched ordinary full-stock pairs per title and rank. Preserve all physical endpoints including incomplete TH12 control as a separate population. Same build, rank, host and controller seed per pair, counterbalanced actor order. In-game RNG is uncontrolled. Primary outcome Stage 2. Secondary measures include spare lives, Bomb stock or Power, first life loss and per-game progress. Original camera only, no assistance or Continue.')
    protocol=root/'camera-protocol.json';freeze(protocol,plan)
    qualification=root/'stage2-qualification.json';src=d/'th11-original-ppo/stage2-qualification.json'
    if not qualification.exists():shutil.copy2(src,qualification)
    if digest(qualification)!=digest(src):raise ValueError('Changed TH11 original qualifier')
    print('Registered five titles,',sum(map(len,attempts.values())),'training starts and',2*sum(map(len,held.values())),'held starts',actor,flush=True)
    grades_path=root/'th12-train-qualified-coverage.json'
    while not grades_path.exists():time.sleep(15)
    grades=read(grades_path)['credits']
    by_seed={int(v['seed']):v for v in grades}
    if len(grades)!=24 or set(by_seed)!=set(attempts['12']):raise ValueError('Incomplete TH12 attempt roster')
    qualified={2:[],3:[]};fitted_receipts={}
    for t,file in records(root,True,True):
        if t!=12:continue
        row=read(file);seed=row['seed']
        if seed not in by_seed or by_seed[seed]['coverage']!='qualified':continue
        if seed in fitted_receipts or row['policy_sha256']!=actor or digest(file)!=by_seed[seed]['receipt_sha256']:
            raise ValueError('Unbound TH12 training receipt')
        fitted_receipts[seed]=file
    for seed,v in sorted(by_seed.items()):
        if (v['coverage'] not in ('qualified','indeterminate') or v['rank']!=plan['rank_by_seed']['12'][str(seed)]
            or digest(d/v['evidence_path'])!=v['evidence_sha256'] or digest(d/v['receipt_path'])!=v['receipt_sha256']):
            raise ValueError('Unverified original TH12 grade')
        if v['coverage']=='qualified':
            if seed not in fitted_receipts:raise ValueError('Missing qualified TH12 provenance')
            qualified[plan['rank_by_seed']['12'][str(seed)]].append(seed)
    n=min(8,len(qualified[2]),len(qualified[3]))
    if not n:raise ValueError('TH12 supplied no matched qualifying rank pair')
    chosen=sorted(qualified[2][:n]+qualified[3][:n])
    if chosen!=sorted(seed for seed,v in by_seed.items() if v['selected_for_ppo']):
        raise ValueError('Independent TH12 input reviewers selected different seeds')
    effective=dict(plan,train_seeds={**attempts,'12':chosen},registered_protocol_sha256=digest(protocol),th12_grades_sha256=digest(grades_path))
    selected=root/'fit-protocol.json';freeze(selected,effective)
    subprocess.run([sys.executable,'-u','-m','scripts.run_registered_camera_update','--root',str(root),'--protocol',str(selected),'--parent',str(source),'--from-sft','--fit-only'],check=True)
    print('Completed five-title online fit',digest(root/'policy/policy.ts'),flush=True)


if __name__=='__main__':main()
