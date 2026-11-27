"""Fine-tune the shared visual policy with equal weight on every game and high difficulty."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from scripts.checkpoint_storage import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','plan','reference','out'):
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--th17',nargs='+',type=Path,help='Qualified original TH17 Hard and Lunatic stage cameras from the frozen pilot/stage roster')
    a=p.parse_args()
    root=a.data.resolve()
    plan=json.loads(a.plan.read_bytes())
    inv=json.loads((root/plan['inventory']).read_bytes())
    parent=root/plan['parent']
    report=json.loads((parent/'report.json').read_bytes())
    if digest(parent/'policy.ts')!=plan['parent_sha256'] or digest(parent/'camera-policy.pt')!=report['sft'] or inv['checkpoint_sha256']!=plan['parent_sha256']:
        raise ValueError('Ranked fine-tune parent differs')
    base=json.loads((root/plan['base_split']).read_bytes())['split']
    extra=json.loads((root/plan['expanded_manifest']).read_bytes())['records']
    paths={r['sha256']:root/r['file'] for r in base['train']}
    paths.update({r['archive_sha256']:root/r['path'] for r in extra})
    inventory=[r for r in inv['sources'] if r['difficulty'] in ('Hard','Lunatic')]
    if sum(r['usable_decisions'] for r in inventory)!=plan['existing_high_rank_windows']:
        raise ValueError('Existing training coverage changed')
    train=[]
    exposure={}
    for r in inventory:
        f=paths[r['camera_sha256']]
        if digest(f)!=r['camera_sha256']:raise ValueError('Changed training shard')
        train.append(f)
        key=r['title']+' '+r['difficulty']
        exposure[key]=exposure.get(key,0)+r['usable_decisions']
    if a.th17:
        declared=plan['new_train_sources']
        expected={(r['sha256'],r.get('stage',1)):(r['player_group'],r['difficulty']) for r in declared}
        if len(expected)!=len(declared) or {r['difficulty'] for r in declared}!={'Hard','Lunatic'}:
            raise ValueError('TH17 registration must cover both ranks with distinct source stages')
        seen=set()
        for f in a.th17:
            meta=json.loads(f.with_suffix('.json').read_bytes())
            key=meta['replay_sha256'],meta['stage']
            if key in seen or expected.get(key)!=(meta['player_group'],meta['difficulty']) or digest(f)!=meta['archive_sha256']:
                raise ValueError('TH17 training source differs from frozen roster')
            if meta['title']!='TH17' or not 1<=meta['stage']<=6 or meta['focus_bit']!=8 or meta['split']!='train':
                raise ValueError('Unqualified TH17 stage or input schema')
            if meta['original_software_exe_sha256']!=plan['th17_build']:raise ValueError('TH17 executable differs')
            seen.add(key);train.append(f)
        if seen!=set(expected):raise ValueError('Missing a registered TH17 TRAIN replay stage')
    held=[root/r['file'] for r in base['held']]
    if any(digest(f)!=r['sha256'] for f,r in zip(held,base['held'])):raise ValueError('Held source changed')
    print('START',len(train),'training stages; original three-title source windows',exposure,flush=True)
    subprocess.run([sys.executable,'-u','-m','scripts.train_human_camera_policy','--encoder',str(parent/'camera-policy.pt'),
        '--reference',str(a.reference),'--out',str(a.out),'--steps',str(plan['updates']),'--cpu-replay',
        '--natural-movement-frequency','--balance-titles','--balance-difficulties',
        '--bomb-label-smoothing',str(plan.get('bomb_label_smoothing',.025)),
        *(['--future-movement-weight',str(plan['future_movement_weight'])] if plan.get('future_movement_weight') else []),
        '--train',*map(str,train),'--held',*map(str,held)],check=True)
    result=json.loads((a.out/'report.json').read_bytes())
    if not result['difficulty_balanced'] or not result['title_balanced'] or {r['sha256'] for r in result['train']}!={digest(f) for f in train}:
        raise ValueError('Completed run differs from registered source set')


if __name__=='__main__':
    main()
