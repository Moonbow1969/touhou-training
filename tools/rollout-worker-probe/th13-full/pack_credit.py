"""Package a completed native TH13 full-action stream for on-policy PPO."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def pack(folder, role, seed, checkpoint, output):
    data = np.fromfile(folder/"steps.bin", dtype="<f4").reshape(-1,143)
    if not len(data) or not np.isfinite(data).all():
        raise ValueError("Missing, truncated or nonfinite control trace")
    if data[-1,120]!=1 or data[:-1,120].any() or data[:,132].any():
        raise ValueError("Require an uninterrupted full credit with no Continue")
    if not np.array_equal(data[:,122],data[:,123]):
        raise ValueError("X11 input acknowledgement differs from sampled commands")
    if not np.all(data[:,141] == 1) or data[0,125] != 1:
        raise ValueError("Expected a Normal credit starting in Stage 1")
    stage2 = np.flatnonzero(data[:,126] >= 2)
    actions = data[:,110:114].astype(np.int8)
    policy_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    np.savez_compressed(output, features=data[:,:110], action=data[:,110:114].astype(np.int8),
                        logp=data[:,114:118], value=data[:,118], reward=data[:,119],
                        terminated=data[:,120].astype(bool), seconds=data[:,121],
                        keymask=data[:,122].astype(np.uint16), stage_before=data[:,125].astype(np.int16),
                        stage=data[:,126].astype(np.int16), spares=data[:,128].astype(np.int16),
                        bombs=data[:,130].astype(np.int16), native_keys=data[:,124].astype(np.uint16),
                        resource_before=data[:,[127,129]].astype(np.int16), raw_score=data[:,131].astype(np.int32),
                        continues=data[:,132].astype(np.int16), pause_state=data[:,133].astype(np.int16),
                        native_frame=data[:,135].astype(np.int32), reward_components=data[:,136:141],
                        policy_sha256=policy_hash)
    result=dict(title=13, seed=seed, role=role, policy_sha256=policy_hash,
                terminal=True, file=output.name, sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                steps=len(data), stage2=bool(len(stage2)),
                stage2_entry_spares=int(data[stage2[0],128]) if len(stage2) else None,
                stage2_entry_bombs=int(data[stage2[0],130]) if len(stage2) else None,
                last_playable_spares=int(data[-1,127]), last_playable_bombs=int(data[-1,129]),
                remaining_lives=int(data[-1,128]), remaining_bombs=int(data[-1,130]),
                effective_bombs=int(data[:,139].sum()), confirmed_losses=int(data[:,136].sum()), life_earned=int(data[:,137].sum()),
                wall_seconds=float(data[-1,121]), return_sum=float(data[:,119].sum()), median_hold_ms=float(np.median(data[:,140])*1000),
                stage2_entry_seconds=float(data[stage2[0],121]) if len(stage2) else None,
                horizontal_moves=int(((actions[:,0]%3)!=1).sum()), vertical_moves=int(((actions[:,0]//3)!=1).sum()),
                focus_steps=int(actions[:,1].sum()), shoot_steps=int(actions[:,2].sum()), bomb_key_steps=int(actions[:,3].sum()),
                native_key_match=int(((data[:,124].astype(int)&251)==data[:,122]).sum()), x_key_match=len(data),
                continues=0)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("credit_dir",type=Path)
    p.add_argument("checkpoint",type=Path)
    p.add_argument("--role",choices=["train","held"],required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    receipt=pack(args.credit_dir,args.role,args.seed,args.checkpoint,args.out)
    print(json.dumps(receipt,indent=2))
