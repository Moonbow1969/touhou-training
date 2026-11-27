"""Measure predicted stochastic action frequencies on unmodified held human cameras."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.train_human_camera_policy import load_replay


def measure(model, row):
    frames=torch.from_numpy(row['raw'])
    with torch.inference_mode():
        batches=[model(batch.float()/255) for batch in frames.split(64)]
    move=torch.cat([out[0] for out in batches]).softmax(1).numpy().astype(np.float64)
    bomb=torch.cat([out[3] for out in batches]).softmax(1)[:,1].numpy().astype(np.float64)
    valid=np.diff(row['native_frame'])==4
    action=row['action']
    # Independent policy samples on adjacent genuine observations. Across a recording
    # gap there is no adjacent action pair, even when the two retained rows are neighbors.
    expected_switches=(1-(move[:-1]*move[1:]).sum(1))[valid]
    positives=action[:,3].astype(bool)
    return dict(observations=len(action), consecutive_pairs=int(valid.sum()),
                expert_movement_changes=int((action[1:,0]!=action[:-1,0])[valid].sum()),
                expected_policy_movement_changes=float(expected_switches.sum()),
                expert_bomb_windows=int(positives.sum()),expected_policy_bomb_windows=float(bomb.sum()),
                expected_policy_bombs_on_expert_positive=float(bomb[positives].sum()),
                expected_policy_bombs_on_expert_negative=float(bomb[~positives].sum()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs='+',type=Path,required=True)
    p.add_argument('--held',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    if len(set(a.held))!=len(a.held):raise ValueError('Repeated replay shard')
    held=[load_replay(f) for f in a.held]
    results=[]
    for folder in a.models:
        path=folder/'policy.ts'
        report=json.loads((folder/'report.json').read_bytes())
        if digest(path)!=report['torchscript_sha256'] or {r['sha256'] for r in held}!={r['sha256'] for r in report['held']}:
            raise ValueError('Actor or held split differs from original registration')
        model=torch.jit.load(str(path),map_location='cpu').eval()
        rows=[dict(title=r['title'],player=r['group'],sha256=r['sha256'],counts=measure(model,r)) for r in held]
        totals={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
        results.append(dict(policy_sha256=digest(path),totals=totals,replays=rows))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(dict(
        measure='Expected independently sampled actions on actual human-held observations. Movement changes use only adjacent native four-tick pairs. Bomb window means X active anywhere in next four ticks; multiple windows can belong to one physical press.',
        policies=results),indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
