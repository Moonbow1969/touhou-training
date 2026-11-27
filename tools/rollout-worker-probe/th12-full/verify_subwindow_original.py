"""Independently validate native stock, the applied half-actions, original pixels and both learned heads."""
from collections import Counter,defaultdict
from pathlib import Path
import json
import sys
import numpy as np
import torch
import argparse
import hashlib

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
from difficulty import verify_difficulty
from stage_transition import inspect_credit
from verify_subwindow_probe import table,native_mask,x11_mask

def score(raw,model_path,job,arm):
    stage=inspect_credit(raw);wire=table(raw/'subwindow-wire.tsv');ranks=table(raw/'difficulty-at-reset.tsv')
    assert len(ranks)==1 and ranks[0]['rank']==ranks[0]['recorded_rank']==job['rank']
    mark=(raw/'outcome.txt').read_text().split();assert mark[0]=='game_over' and int(mark[1])==len(wire)
    n=len(wire);duration=float(mark[2]);rank=verify_difficulty(raw,job['rank'],duration,'9fa30534d6710abd245dfa64fc222ea6816f960b96974fed536a00ae1c08094b')
    assert stage['stage2_reached'] is not None
    trace=table(raw/'engineering-native.tsv');life=trace[(trace['mode']==0)&(trace['stage']==trace['manager_stage'])&(trace['stage']>0)&(trace['tick']>=0)]
    death=int(np.flatnonzero(life['spares']==-1)[0]);last=life[death];old=life[death-1];first_loss=life[np.flatnonzero(life['spares']<2)[0]]
    assert tuple(life[0][k] for k in ('stage','spares','bombs'))==(1,2,2) and old['spares']==0 and np.all(life['spares'][death:]==-1)
    proof=defaultdict(set)
    for s in life[:death]:
        proof[int(s['stage']),int(s['tick'])].add(int(s['keys'])&251)
    model=torch.jit.load(str(model_path),map_location='cpu').eval();torch.set_num_threads(2)
    with np.load(raw/'model-wire.npz') as z:
        images=z['frames'];actions=z['action'];lp=z['logp'];wire_logits=z['logits']
    assert images.shape==(n,112,96) and actions.shape==(n,2,4) and images.tobytes()==(raw/'observations.u8').read_bytes()
    errors=np.zeros((2,4));maximum_logits=0.
    with torch.inference_mode():
        past=np.concatenate([images[:1],images[:-1]])
        deltas=np.abs(images.astype('i2')-past).astype('u1')
        for i in range(0,n,32):
            observed=torch.from_numpy(np.stack([images[i:i+32],deltas[i:i+32]],1)).float()/255.
            heads=model(observed)
            maximum_logits=max(maximum_logits,float(abs(torch.cat(heads,-1).numpy()-wire_logits[i:i+32]).max()))
            for slot in (0,1):
                for k,h in enumerate(heads):
                    ids=torch.from_numpy(actions[i:i+32,slot,k].copy()).long()[:,None]
                    pred=h[:,slot if arm=='two-subwindows' else 0,:].log_softmax(1).gather(1,ids)[:,0].numpy()
                    errors[slot,k]=max(errors[slot,k],float(abs(pred-lp[i:i+32,slot,k]).max()))
    assert maximum_logits<1e-3 and errors.max()<1e-3
    counts=Counter();changed=Counter();offset=[Counter(),Counter()];held_middle_differs=0;missed_groups=0;prev=None
    partial=[]
    for i,row in enumerate(wire):
        st=int(row['stage']);t=int(row['image_tick'])
        if prev is not None and prev[0]==st:missed_groups+=max(0,(t-prev[1])//4-1)
        prev=st,t;different=native_mask(actions[i,0])!=native_mask(actions[i,1]);held_middle_differs+=int(different)
        if arm=='four-tick':assert not different
        for k,label in enumerate(('first','second')):
            x=x11_mask(actions[i,k]);actual=int(row[label+'_physical']);assert int(row[label+'_requested'])==x and actual in (-1,x)
            if actual<0:partial.append(dict(group=i,slot=k,image_tick=t,transition_tick=int(row['end_tick']),transition_stage=int(row['end_stage']),spares=int(row['end_spares'])))
            else:offset[k][int(row[label+'_applied_tick'])-t-2*k]+=1
            for phase in (1,2):
                intended=t+2*k+phase
                if (st,intended)>=(int(last['stage']),int(last['tick'])):counts['at_or_after_terminal']+=1;continue
                known=proof.get((st,intended),set())
                outcome='unknown' if len(known)!=1 else ('matches' if next(iter(known))==native_mask(actions[i,k]) else 'mismatches')
                counts[outcome]+=1
                if k==1 and different:changed[outcome]+=1
    if any(z['group']!=n-1 and z['spares']>=0 and z['transition_stage']==wire[z['group']]['stage'] for z in partial):
        raise ValueError('Policy withheld a half action in an ordinary live stage')
    last_requested=[]
    for pos in range(max(0,n-2),n):
        r=wire[pos];last_requested.append({k:int(r[k]) for k in wire.dtype.names})
    at_terminal=[{k:int(s[k]) for k in ('stage','tick','keys','spares','bombs','ms')} for s in life[max(0,death-5):death+3]]
    checks={f:digest(raw/f) for f in ('decisions.tsv','observations.u8','subwindow-wire.tsv','engineering-native.tsv','stage-native.tsv','difficulty-native.tsv','identity.txt','model-wire.npz','terminal-at-control.ppm','stage-ranking.png','warmup.json') if (raw/f).is_file()}
    warm=json.loads((raw/'warmup.json').read_bytes());assert warm['rng_state_unchanged'] and not warm['socket_bound'] and len(warm['forward_ms'])==4 and warm['policy_sha256']==digest(model_path)
    assert (raw/'identity.txt').read_text().split()[:3]==['held',str(job['seed']),digest(model_path)]
    output=dict(title=12,seed=job['seed'],lane=job['lane'],host=job['host'],policy=arm,policy_sha256=digest(model_path),
        native_outcome='game_over',terminal_rank=stage['rank'],stage2_reached=stage['stage2_reached'],stage2_review=stage,
        initial_spares=2,initial_bombs=2,remaining_spare_lives=int(old['spares']),last_playable_bombs=int(old['bombs']),
        final_stage=int(last['stage']),final_stage_native_tick=int(last['tick']),first_loss_stage=int(first_loss['stage']),first_loss_tick=int(first_loss['tick']),
        groups=n,proposed_half_actions=2*n,physically_applied_half_actions=sum(sum(z.values()) for z in offset),unexecuted_half_actions=partial,
        original_playable_tick_controls=dict(counts),changed_second_ticks=dict(changed),groups_with_different_half_masks=held_middle_differs,
        absent_four_tick_camera_groups=missed_groups,apply_tick_offsets=[dict(z) for z in offset],last_two_native_decisions=last_requested,
        native_terminal_consumption_neighborhood=at_terminal,maximum_logits_error=maximum_logits,maximum_logp_error_by_slot_head=errors.tolist(),
        complete_input_duration_s=duration,source_sha256=checks,**rank)
    return output

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('records',type=Path,help='Extracted and byte-restored native evidence root')
    cli.add_argument('models',type=Path,help='Directory containing the two frozen *-policy.ts models')
    args=cli.parse_args()
    p=json.loads((args.records/'protocol.json').read_bytes())
    total=0
    for job in p['prespecified_roster']:
        for arm in job['arms']:
            model=args.models/(arm+'-policy.ts')
            assert digest(model)==p['models'][arm]
            raw=args.records/'native12'/f"held-{arm}-{job['seed']}-{job['lane']}"
            physical=score(raw,model,job,arm)
            # IEEE results for batched Torch may vary at the lowest digits across CPUs.
            signed=json.loads((raw/'receipt.json').read_bytes())
            for name in ['maximum_logits_error','maximum_logp_error_by_slot_head']:
                if float(np.max(signed.pop(name)))>=.001 or float(np.max(physical.pop(name)))>=.001:
                    raise ValueError('Model rollout differs from recorded distributions')
            same=json.loads(json.dumps(physical))
            if same!=signed:
                raise ValueError((raw.name,[key for key in set(signed)|set(same) if signed.get(key)!=same.get(key)]))
            total+=1
            print('Replayed original',raw.name,flush=True)
    assert total==32
if __name__=='__main__':main()
