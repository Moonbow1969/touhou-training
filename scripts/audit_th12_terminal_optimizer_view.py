"""Check all sampled actions and prove the optimizer stops at the last consumed playable input."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scripts.checkpoint_storage import digest
from scripts.th12_playable_actions import verify_source
from scripts.train_full_resolution_camera_ppo import CameraActorCritic,CameraPolicy,credit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('receipt','parent','out'):p.add_argument('--'+arg,type=Path,required=True)
    a=p.parse_args()
    data=json.loads(a.receipt.read_bytes())
    trace=a.receipt.parent/data['trajectory']
    check=verify_source(12,data,trace,a.receipt.parent.parent/'raw'/data['episode'])
    if check is None:raise ValueError('No post-terminal input to audit')
    actor=CameraPolicy()
    state=torch.load(a.parent/'camera-policy.pt',weights_only=True,map_location='cpu')
    if any('policy.spatial_context.' in k for k in state):actor.add_spatial_context()
    actor.load_state_dict(state)
    model=CameraActorCritic(actor).eval()
    parent=digest(a.parent/'policy.ts')
    if parent!=data['policy_sha256']:raise ValueError('Wrong behavior policy')
    record=dict(role='train',policy_sha256=parent,outcome='game_over',title=12,seed=data['seed'],
                file=trace.name,sha256=data['trajectory_sha256'],postterminal_input_boundary=check)
    torch.set_num_threads(2)
    arrays,quality=credit(record,a.receipt,model,parent,'cpu')
    n=check['live_actions']
    with np.load(trace) as raw:
        if not all(len(arr)==n for arr in arrays):raise ValueError('Masked action leaked into optimizer')
        expected=raw['reward'][n-1]-raw['value'].reshape(-1)[n-1]
        np.testing.assert_allclose(arrays[3][-1].item(),expected,atol=1e-6)
        np.testing.assert_allclose(arrays[4][-1].item(),raw['reward'][n-1],atol=1e-6)
        if float(raw['reward_parts'][n-1,4])!=-1 or float(raw['reward'][-1])!=0:raise ValueError('Wrong fatal action')
    audit=dict(quality=quality,verified_boundary=check,receipt_sha256=digest(a.receipt),trace_sha256=digest(trace),
               last_playable_advantage=float(arrays[3][-1]),last_playable_return=float(arrays[4][-1]),
               actual_model=digest(a.parent/'policy.ts'),original_game_over_credits=1)
    a.out.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    print(quality,flush=True)


if __name__=='__main__':
    main()
