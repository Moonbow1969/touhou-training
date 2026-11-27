"""Count authenticated source-policy games in the current five-title online campaign."""
import argparse
from collections import Counter
import json
from pathlib import Path
from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root
    protocol=root/'camera-protocol.json';plan=json.loads(protocol.read_bytes())
    observed={};counts=Counter()
    for t,file in records(root,True,True,include_held_endpoints=True):
        try:row=json.loads(file.read_bytes())
        except (PermissionError,json.JSONDecodeError):continue
        seed=row.get('seed')
        if seed not in plan['train_seeds'].get(str(t),[]):continue
        if row['policy_sha256']!=plan['behavior_sha256'] or row.get('continues',0) or not row.get('terminal',True):
            raise ValueError(f'Unexpected training source {file}')
        key=t,seed
        if key in observed:raise ValueError(f'Duplicate physical training receipt {key}')
        observed[key]=digest(file)
        counts[t,plan['rank_by_seed'][str(t)][str(seed)]]+=1
    print('Game   Hard   Lunatic   registered starts')
    for title,slots in plan['train_seeds'].items():
        t=int(title)
        print(f'TH{t:02}   {counts[t,2]:4}   {counts[t,3]:7}   {len(slots):17}')
    policy=root/'policy/report.json'
    if policy.exists():
        result=json.loads(policy.read_bytes())
        if result['policy_sha256']!=digest(policy.with_name('policy.ts')):raise ValueError('Fitted checkpoint differs')
        print(f"Verified fit: {len(result['credits'])} credits, {result['steps']:,} playable decisions")
    else:
        print('PPO fit pending. The table counts readable full-game receipts before training selection.')
    grade=root/'th12-train-qualified-coverage.json'
    if grade.is_file():
        grades=Counter(v['coverage'] for v in json.loads(grade.read_bytes())['credits'])
        print('TH12 controller coverage:',dict(grades))


if __name__=='__main__':main()
