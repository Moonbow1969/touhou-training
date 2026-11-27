"""Verify original TH11 stock rewards, fit the registered camera PPO, and score complete held pairs."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts.checkpoint_storage import digest
from scripts.qualified_th11_stage2 import read_credit as verify_stage


def wait_file(path):
    while not path.is_file():
        time.sleep(8)


def audit(folder, role, seed, parent, plan):
    file = folder / 'receipt.json'
    wait_file(file)
    r = json.loads(file.read_bytes())
    rank = plan['rank_by_seed']['11'][str(seed)]
    if (r['seed'] != seed or r['difficulty'] != rank or r['role'] != role
            or r['policy_sha256'] != parent or r['continues'] or not r['full_credit']):
        raise ValueError(f'Unregistered original source {file}')
    if (folder/'actual-executable.sha256').read_text().split()[0] != plan['native_executable_sha256']:
        raise ValueError('Original trial executable differs')
    trace = folder/'trajectory.npz'
    if digest(trace) != r['trajectory_sha256']:
        raise ValueError('Original trajectory checksum differs')
    for name in ('behavior.csv','endpoint.png','native.bin','camera.ns'):
        if digest(folder/name) != r['evidence_sha256'][name]:
            raise ValueError(f'Changed original evidence {folder/name}')
    with (folder/'behavior.csv').open(encoding='utf8') as stream:
        decisions = list(csv.DictReader(stream))
    with np.load(trace,allow_pickle=False) as z:
        n=len(z['action'])
        if n!=len(decisions) or n!=r['steps'] or z['frames'].shape!=(n,112,96):
            raise ValueError('Action/camera mismatch')
        original=z['native']
        if original.shape[1]!=12 or np.any(original[:,6]!=rank) or np.any(original[:,8]!=0):
            raise ValueError('Invalid rank or continues from native stream')
        if original[0,5]!=2 or original[-1,5]!=-1:
            raise ValueError('Missing original full-life start or game over')
        after=np.asarray([int(v['after_ms']) for v in decisions],dtype=np.int64)
        before=int(decisions[0]['before_ms'])
        ends=(after-before) % (2**32)
        event_time=(original[:,0]-before) % (2**32)
        if np.any(np.diff(ends)<0) or np.any(np.diff(event_time)<0) or ends[-1]<event_time[-1]:
            raise ValueError('Native and decision clocks disagree')
        delta=np.diff(original[:,5])
        event=np.flatnonzero(delta)
        assigned=np.searchsorted(ends,event_time[event+1],side='left')
        expected=np.zeros(n,dtype=np.float32)
        np.add.at(expected,assigned,delta[event])
        np.testing.assert_array_equal(expected,z['reward'].reshape(-1))
        terminal=z['terminated'].reshape(-1).astype(bool)
        if not terminal[-1] or terminal[:-1].any():
            raise ValueError('Invalid episode boundary')
        actual=np.array([[int(v[k]) for k in ('dir','focus','shoot','bomb')] for v in decisions])
        np.testing.assert_array_equal(actual,z['action'])
    settled=np.fromfile(folder/'native.bin',dtype='<i4').reshape(-1,12)[-1]
    if settled[5]!=-1 or settled[6]!=rank or settled[8]!=0 or settled[2]!=r['terminal_power_units']:
        raise ValueError('Settled original stock disagrees with terminal receipt')
    row=dict(title=11,seed=seed,role=role,policy_sha256=parent,outcome='game_over',difficulty=rank,
             file=str(trace.resolve()),sha256=digest(trace),source_log_sha256=digest(file),
             difficulty_evidence_sha256=r['evidence_sha256']['native.bin'])
    # Public records identify original files by portable paths.
    summary=dict(seed=seed,difficulty=rank,policy_sha256=parent,host=r['host'],lane=r.get('lane'),
                 decisions=n,score=int(settled[1])*10,score_at_final_loss=int(original[-1,1])*10,
                 terminal_spares=r['terminal_spares'],last_loss_power=r['power_at_last_loss_units']/20,
                 terminal_power=r['terminal_power_units']/20,effective_casts=r['effective_confirmed_casts'],
                 first_loss_wall_seconds=r['native_life_losses'][0]['original_ms']/1000,
                 wall_seconds=r['real_wall_seconds'],native_key_matches=r['native_input_matches'],
                 native_key_total=r['native_input_total'],receipt_sha256=digest(file),
                 stock_event_decisions=assigned.tolist(),native_stock_changes=delta[event].tolist())
    return row,summary


def save(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--parent',required=True,type=Path)
    a=parser.parse_args()
    root=a.root.resolve();parent=a.parent.resolve()
    protocol=root/'camera-protocol.json';plan=json.loads(protocol.read_bytes())
    actor=parent/'policy.ts';h=digest(actor)
    if h!=plan['behavior_sha256']:raise ValueError('Unregistered initial actor')
    rows=[];training=[]
    for seed in plan['train_seeds']['11']:
        folder=root/'native11'/f'train-merged-sft-{seed}'
        row,s=audit(folder,'train',seed,h,plan)
        row['file']=(folder/'trajectory.npz').relative_to(root).as_posix()
        rows.append(row);training.append(s)
        print('Original TRAIN verified',len(rows),seed,s['decisions'],flush=True)
    save(root/'training-credits.json',rows);save(root/'verified-stock-reward.json',training)
    policy=root/'policy'
    if not (policy/'report.json').is_file():
        subprocess.run([sys.executable,'-u','-m','scripts.train_full_resolution_camera_ppo','--manifest',root/'training-credits.json',
                        '--protocol',protocol,'--sft',parent/'camera-policy.pt','--behavior',actor,'--out',policy],check=True)
    alt=digest(policy/'policy.ts')
    report=json.loads((policy/'report.json').read_bytes())
    if alt!=report['policy_sha256'] or report['native_behavior_sha256']!=h:raise ValueError('Model lineage differs')
    held=[];arms=(plan['reference_arm'],plan['candidate'])
    gate=root/'stage2-qualification.json'
    wait_file(gate)
    qualification=json.loads(gate.read_bytes())
    if (not qualification['qualified'] or qualification['native_executable_sha256']!=plan['native_executable_sha256']
            or digest(root/qualification['evidence_zip'])!=qualification['evidence_zip_sha256']):
        raise ValueError('Unverified original positive Stage 2 transition')
    for seed in plan['eval_seeds']['11']:
        pair=[]
        for arm,sha in zip(arms,(h,alt)):
            folder=root/'native11'/f'held-{arm}-{seed}'
            _,s=audit(folder,'held',seed,sha,plan)
            stage=verify_stage(folder)
            pair.append(dict(arm=arm,stage2_reached=stage['stage2_reached'],first_stage2=stage['first_stage2'],**s))
        if pair[0]['host']!=pair[1]['host'] or pair[0]['lane']!=pair[1]['lane']:
            raise ValueError('Pair must share its physical lane')
        held+=pair
    out=root/'evaluation';out.mkdir(exist_ok=True)
    save(out/'credits.json',held)
    stats={}
    for rank,label in ((2,'Hard'),(3,'Lunatic')):
        cohort=[r for r in held if r['difficulty']==rank]
        values={arm:[r for r in cohort if r['arm']==arm] for arm in arms}
        pairs=len(values[arms[0]])
        rng=np.random.default_rng(1401126);idx=rng.integers(pairs,size=(20000,pairs))
        result=dict(pairs=pairs,stage2_reached={arm:sum(r['stage2_reached'] for r in values[arm]) for arm in arms},
                    stage2_rate={arm:sum(r['stage2_reached'] for r in values[arm])/pairs for arm in arms},
                    stage_qualification_sha256=digest(gate))
        for field in ('score','first_loss_wall_seconds','wall_seconds','terminal_spares','last_loss_power','terminal_power','effective_casts','decisions'):
            x,y=(np.array([r[field] for r in values[arm]],dtype=float) for arm in arms)
            delta=y-x
            result[field]=dict(means=[float(x.mean()),float(y.mean())],paired_difference=float(delta.mean()),
                               paired_seed_bootstrap95=np.quantile(delta[idx].mean(1),[.025,.975]).tolist())
        stats[label]=result
    save(out/'paired-by-difficulty.json',stats)


if __name__=='__main__':
    main()
