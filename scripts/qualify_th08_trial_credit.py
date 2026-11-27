"""Qualify original TH08 stock, inputs, screen labels and event-based rewards."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.original_camera_heads import policy_heads
from scripts.run_th08_trial_camera import mask_for
from scripts.train_five_original_games import freeze


def observations(folder, rank):
    rows=[json.loads(line) for line in (folder/'timing.jsonl').read_bytes().splitlines()]
    tool=Path(__file__).resolve().parents[1]/'tools/rollout-worker-probe/th08-trial'
    with np.load(tool/'rank-glyphs.npz') as refs:
        rank_ref={name:refs[name] for name in refs.files}
    with np.load(folder/'native-observers.npz',allow_pickle=False) as observer:
        hud=observer['hud'];patch=observer['prompt']
    if hud.shape!=(len(rows),204,206,3) or hud.dtype!=np.uint8 or patch.shape!=(len(rows),40,218):
        raise ValueError('Incomplete original rank/stock or terminal observations')
    if rows[-1]['phase']!='terminal' or any(x['phase']=='terminal' for x in rows[:-1]):
        raise ValueError('Original continue prompt was not the unique terminal')
    idx=np.array([i for i,row in enumerate(rows) if row['phase']=='policy'])
    if not np.array_equal(idx,np.arange(idx[0],len(rows)-1)):
        raise ValueError('Discontinuous neural inputs')
    playable=rows[idx[0]:-1]
    glyph=hud[idx,170:203,104:205].astype(np.int16)
    r,g,b=np.moveaxis(glyph,-1,0)
    ink=(b>100)&(r-g>15)&(b-g>25)
    sim={k:(ink & mask).sum((1,2))/np.maximum((ink | mask).sum((1,2)),1) for k,mask in rank_ref.items()}
    other='Lunatic' if rank=='Hard' else 'Hard'
    certified=(sim[rank]>.65)&(sim[rank]-sim[other]>.2)
    if certified.mean()<.97:
        raise ValueError('Original difficulty glyph did not confirm the requested difficulty')
    with np.load(tool/'continue-edges.npz') as ref:
        edge=ref['edge'];mask=ref['mask']
    final=patch[-1,3:37,2:215].astype(np.float32)
    contrast=(final-cv2.GaussianBlur(final,(0,0),1.35))[mask];reference=edge[mask]
    correlation=float(contrast@reference/(np.linalg.norm(contrast)*np.linalg.norm(reference)))
    if correlation<.78 or abs(correlation-rows[-1]['continue_structure'])>1e-4:
        raise ValueError('Native continue-screen structure failed independent verification')
    measured=[]
    for y,kind in ((74,'spares'),(90,'bombs')):
        hits=[]
        for x in range(66,179,16):
            chip=hud[:,y-6:y+7,x-6:x+7].astype(np.int16)
            r,g,b=np.moveaxis(chip,3,0)
            lit=(r>125)&(r-g>32)&(r-b>36) if kind=='spares' else (b>110)&(b-r>20)&(b-g>28)
            hits.append(lit.sum((1,2))>=16)
        flags=np.array(hits).T
        counts=[int(v.sum()) if np.all(v[:v.sum()]) else None for v in flags]
        if counts!=[r[kind] for r in rows]:
            raise ValueError(f'Independent {kind} re-read failed')
        measured.append(counts[idx[0]:])
    if measured[0][-1]!=0:
        raise ValueError('Original shows spare lives at terminal')
    return playable,measured,dict(difficulty=rank,matching_hud_observations=int(certified.sum()),
           scored_hud_observations=len(idx),mean_original_rank_similarity=float(sim[rank].mean()),
           terminal_glyph_correlation=correlation,terminal_hud_bombs=measured[1][-1])


def changes(readings,start):
    if readings[:2]!=[start,start]:
        raise ValueError('Expected full starting stock on the first two gameplay images')
    events=[];previous=start
    for i in range(1,len(readings)-1):
        cur=readings[i]
        if cur is not None and cur!=previous and readings[i+1]==cur:
            events.append((i-1,cur-previous));previous=cur
    return events


def qualify(folder,actor,arm_digest,expected_seed,rank,role,out):
    receipt=json.loads((folder/'receipt.json').read_bytes())
    plan=json.loads((folder/'protocol.json').read_bytes())
    if (receipt['role']!=role or receipt['seed']!=expected_seed or receipt['policy_sha256']!=arm_digest
            or receipt['requested_rank']!=rank or receipt['outcome']!='original_continue_prompt'
            or plan['policy']!=arm_digest or plan['role']!=role):
        raise ValueError('Unregistered original behavior, difficulty or terminal')
    for name,key in [('timing.jsonl','timing_sha256'),('behavior.npz','behavior_sha256'),
                     ('native-observers.npz','native_observers_sha256'),('terminal.png','terminal_sha256'),
                     ('chosen-rank.png','rank_evidence_sha256')]:
        if digest(folder/name)!=receipt[key]:
            raise ValueError(f'Changed physical source {name}')
    playable,(lives,bombs),observed=observations(folder,rank)
    with np.load(folder/'behavior.npz',allow_pickle=False) as trace:
        frames,actions,lp,val=(trace[k] for k in ('frames','action','logp','value'))
    n=len(frames)
    if (n!=len(playable) or n!=receipt['physical_decisions'] or
            [mask_for(a.astype(int).tolist()) for a in actions]!=[r['mask'] for r in playable]):
        raise ValueError('Physical key sequence differs from the complete camera trace')
    ns=np.array([r['monotonic_ns'] for r in playable],dtype=np.int64)
    applied=np.array([r['x_applied_ns'] for r in playable],dtype=np.int64)
    if np.any(applied<ns) or np.any(applied[:-1]>=ns[1:]):
        raise ValueError('Game camera to applied key chronology changed')
    prior=np.concatenate((frames[:1],frames[:-1]))
    from scripts.train_replay_camera_encoder import channels
    images=torch.from_numpy(channels(frames,prior)).float()/255
    error=v_error=0.
    with torch.inference_mode():
        for j in range(0,n,48):
            batch=images[j:j+48]
            heads,value=policy_heads(actor(batch),len(batch))
            actual=torch.as_tensor(actions[j:j+48],dtype=torch.long)
            predicted=torch.stack([h.log_softmax(1).gather(1,actual[:,k:k+1])[:,0] for k,h in enumerate(heads)],1)
            error=max(error,float((predicted-torch.from_numpy(lp[j:j+48])).abs().max()))
            v_error=max(v_error,float((value.flatten()-torch.from_numpy(val[j:j+48])).abs().max()))
    if error>8e-4 or v_error>3e-4:
        raise ValueError('Independent behavior model disagrees with original policy outputs')
    # Last playable stock is scored separately from the continued menu display.
    loss_events=changes(lives,2);bomb_events=changes(bombs,3)
    if 2+sum(delta for _,delta in loss_events)!=0 or not all(abs(delta)==1 for _,delta in loss_events):
        raise ValueError('Spare-life changes do not reconcile to native ending stock')
    if 3+sum(delta for _,delta in bomb_events)!=bombs[-2]:
        raise ValueError('Bomb ledger does not reconcile to last playable stock')
    parts=np.zeros((n,3),dtype=np.float32)
    for j,change in loss_events:parts[j,0]=change
    parts[-1,1]=-1.
    for j,change in bomb_events:
        if all(abs(j-i)>6 for i,d in loss_events if d<0):
            parts[j,2]=.03*change
    reward=parts.sum(1);done=np.zeros(n,dtype=bool);done[-1]=True
    out.mkdir(exist_ok=True,parents=True)
    file=out/f'th8-{expected_seed}.npz'
    arrays=dict(frames=frames,action=actions,logp=lp,value=val,reward=reward,terminated=done)
    if file.exists():
        with np.load(file,allow_pickle=False) as saved:
            if set(saved.files)!=set(arrays) or any(not np.array_equal(saved[k],v) for k,v in arrays.items()):
                raise ValueError('Previously packed data differ from original physical replay')
    else:
        np.savez_compressed(file,**arrays)
    record=dict(title=8,seed=expected_seed,policy_sha256=arm_digest,role=role,outcome='game_over',
           difficulty=2 if rank=='Hard' else 3,difficulty_evidence_sha256=receipt['native_observers_sha256'],
           file=file.name,sha256=digest(file),source_log_sha256=digest(folder/'receipt.json'),
           source_actions_sha256=receipt['behavior_sha256'])
    audit=dict(seed=expected_seed,rank=rank,arm_sha256=arm_digest,observed_duration_s=receipt['observed_duration_s'],
           motions=receipt['direction_histogram'],decisions=n,terminal_spares=0,last_playable_bombs=bombs[-2],
           confirmed_deaths=1-sum(min(0,v) for _,v in loss_events),life_events=[list(e) for e in loss_events],bomb_events=[list(e) for e in bomb_events],
           optimizer_reward_by_component=parts.sum(0).tolist(),max_actor_logp_error=error,max_critic_error=v_error,
           applied_ms_median=float(np.median((applied-ns)/1e6)),median_decision_interval_ms=float(np.median(np.diff(ns))/1e6),
           **observed)
    return record,audit


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for arg in ('folder','policy','out'):
        ap.add_argument('--'+arg,type=Path,required=True)
    ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--rank',choices=('Hard','Lunatic'),required=True)
    ap.add_argument('--role',choices=('train','held'),required=True)
    a=ap.parse_args();torch.set_num_threads(2)
    actor=torch.jit.load(str(a.policy),map_location='cpu').eval()
    row,audit=qualify(a.folder,actor,digest(a.policy),a.seed,a.rank,a.role,a.out)
    freeze(a.out/f'{a.seed}-qualification.json',dict(record=row,audit=audit))
    print(audit,flush=True)
