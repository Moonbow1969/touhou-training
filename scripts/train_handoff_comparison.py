"""Compare PPO on post-human-handoff and ordinary-opening TH13 policy trajectories."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scripts.checkpoint_storage import digest
from scripts.normalize_th13_survival_clock import native_progress


def read(f):
    return json.loads(f.read_bytes())


def freeze(f, value):
    if f.is_file() and read(f)!=value:raise ValueError(f'Frozen file differs: {f}')
    f.write_text(json.dumps(value,indent=2)+'\n')


def acquire(root, plan):
    selected=[]
    for s,c in zip(plan['attempt_seeds'],plan['opening_control_seeds'],strict=True):
        suffix=root/'native13'/f'train-humanprefix-visual-{s}'
        control=root/'opening-control'/f'train-four-ppo-{c}'
        if (suffix/'eligible.json').is_file() and (control/'eligible.json').is_file():
            selected.append((suffix,control))
            if len(selected)==4:return selected
        elif not (suffix/'eligible.json').is_file():
            if (suffix/'collection-error.json').is_file():continue
            q=suffix/'qualification.json'
            if q.is_file() and read(q).get('validated_handoff') is False:continue
            return None
        else:return None
    if len(selected)<4:raise RuntimeError('Fewer than four independently qualified paired starts')


def pack(folder, out, policy, prefix):
    r=read(folder/'receipt.json')
    if (r['policy_sha256']!=policy or r['difficulty']!=2 or r['title']!=13 or r['role']!='train'
        or not r['terminal'] or r['continues'] or r['outcome'] not in ('game_over','ending')):
        raise ValueError('Nonqualified full-control policy suffix or opening')
    if prefix and read(folder/'qualification.json').get('validated_handoff') is not True:
        raise ValueError('Human prefix qualification missing')
    f=folder/r['file']
    if digest(f)!=r['sha256'] or digest(folder/'steps.bin')!=r['native_difficulty_trace_sha256']:
        raise ValueError('Original camera or telemetry hash differs')
    with np.load(f,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    raw=np.fromfile(folder/'steps.bin',dtype='<f4').reshape(-1,143)
    if (len(raw)!=r['steps'] or (prefix and raw[0,134]<2995)
        or not np.array_equal(raw[:,110:114],arrays['action']) or not np.array_equal(raw[:,119],arrays['reward'])):
        raise ValueError('Native reward/action trajectory differs from sealed policy rows')
    original=-raw[:,136]+raw[:,137]+2*raw[:,138]+.001*raw[:,140]-.03*raw[:,139]
    np.testing.assert_allclose(original,arrays['reward'],rtol=0,atol=2e-7)
    ticks=native_progress(raw)
    arrays['reward_original']=arrays['reward']
    arrays['native_ticks_earned']=ticks
    arrays['reward']=(arrays['reward']-.001*raw[:,140]+(.001/60)*ticks).astype(np.float32)
    target=out/f"th13-{r['seed']}.npz"
    if not target.exists():np.savez_compressed(target,**arrays)
    else:
        with np.load(target,allow_pickle=False) as existing:
            if set(existing.files)!=set(arrays) or any(not np.array_equal(existing[k],v) for k,v in arrays.items()):
                raise ValueError('Existing packed policy trajectory differs')
    return dict(title=13,seed=r['seed'],role='train',outcome=r['outcome'],file=target.name,sha256=digest(target),
                steps=r['steps'],policy_sha256=policy,difficulty=2,difficulty_evidence_sha256=digest(folder/'steps.bin'),
                source_log_sha256=digest(folder/'receipt.json'),eligibility_sha256=digest(folder/'eligible.json'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True,type=Path)
    d=parser.parse_args().data.resolve();root=d/'human-prefix-curriculum'
    p=read(root/'protocol.json');prior=d/'four-game-online/policy'
    if digest(prior/'policy.ts')!=p['source_policy_sha256']:raise ValueError('Parent actor changed')
    names=['prefix-ppo','opening-ppo']
    seeds=list(range(1281351,1281363))
    comparison=dict(name='TH13 Hard human-prefix vs opening-only online PPO, normal fresh game evaluation',
        reference_arm=names[1],candidate=names[0],eval_seeds={'13':seeds},rank_by_seed={'13':{str(k):2 for k in seeds}},
        controls='Twelve fresh paired ordinary Stage1 openings, Hard Reimu, identical host and original. No human input, continue or helper during evaluation.',
        source=digest(root/'protocol.json'),training_rule='Choose the first four eligible registered suffix attempts in ordinal order and their same-ordinal opening controls. Within each group score all observed complete policy-only sequences, normalize TH13 native clock reward and compute GAE. Then sample min(1536, totals of both arms) decisions per model without replacement using a fixed seed. Same parent, actor/critic and clipping schedule. Keep all 12 attempt outcomes for acquisition-rate reporting.')
    freeze(root/'camera-protocol.json',comparison)
    pairs=acquire(root,p)
    while pairs is None:
        time.sleep(15);pairs=acquire(root,p)
    rows={}
    for i,arm in enumerate(names):
        batch=root/arm/'batch';batch.mkdir(parents=True,exist_ok=True)
        rows[arm]=[pack(pair[i],batch,p['source_policy_sha256'],i==0) for pair in pairs]
        freeze(batch/'credits.json',rows[arm])
    count=min(1536,*[sum(r['steps'] for r in group) for group in rows.values()])
    for arm,records in rows.items():
        folder=root/arm
        fit=dict(learning_rates=read(d/'four-game-online/camera-protocol.json')['learning_rates'],
                 train_seeds={'13':[r['seed'] for r in records]},matched_decisions_per_stratum=count,
                 balanced_title_rank=True,rank_by_seed={'13':{str(r['seed']):2 for r in records}})
        freeze(folder/'fit.json',fit)
        if not (folder/'policy/report.json').is_file():
            subprocess.run(list(map(str,[sys.executable,'-u','-m','scripts.train_full_resolution_camera_ppo',
                '--manifest',folder/'batch/credits.json','--protocol',folder/'fit.json','--parent-actor-critic',prior/'camera-ppo.pt',
                '--behavior',prior/'policy.ts','--out',folder/'policy'])),check=True)
        r=read(folder/'policy/report.json')
        if r['optimizer_decision_sampling']['TH13-Hard']['selected']!=count:raise ValueError('Unmatched optimizer budget')
        print(arm,r['policy_sha256'],count,flush=True)


if __name__=='__main__':
    main()
