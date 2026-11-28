"""Independently reconcile TH07 stock scorecards, saved replays and CPU actor decisions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image,ImageDraw

def digest(path):return sha256(path.read_bytes()).hexdigest()

def run(root,parent):
    manual=json.loads((root/'original-scorecard-readings.json').read_bytes())
    plan=json.loads((root/'protocol.json').read_bytes())
    names=plan['parent_arm'],plan['candidate']
    models={names[0]:parent,names[1]:root/'policy/policy.ts'}
    torch.set_num_threads(2)
    actors={k:torch.jit.load(str(v),map_location='cpu').eval()for k,v in models.items()}
    parents=dict(actors[names[0]].named_parameters());child=dict(actors[names[1]].named_parameters())
    head_delta=child['camera.policy.7.weight'].detach()-parents['camera.policy.7.weight'].detach()
    head_l2={k:float(val.norm())for k,val in zip(['movement','focus','shoot','bomb'],head_delta.split([9,2,2,2]))}
    assert all(v>0 for v in head_l2.values())
    out=[];rows=[]
    for role,numbers in [('train',plan['train']),('held',plan['held'])]:
        for seed in numbers:
            for k in ([names[0]] if role=='train' else names):
                folder=root/(f'train-{seed}' if role=='train' else f'held-{k}-{seed}')
                receipt=json.loads((folder/'receipt.json').read_bytes())
                assert receipt['policy_sha256']==digest(models[k]) and (receipt['role'],receipt['seed'])==(role,seed)
                for f,d in receipt['evidence_sha256'].items():assert digest(folder/f)==d
                screen=manual[folder.name]
                assert receipt['original_replay_score']==screen['score'] and screen['continue']==0 and screen['miss']==3
                with np.load(folder/'policy-trace.npz') as z,torch.inference_mode():
                    frm=torch.from_numpy(z['frames']).float()/255
                    data=torch.stack([frm,(frm-torch.cat([frm[:1],frm[:-1]])).abs()],1)
                    actions=torch.from_numpy(z['action']).long()
                    lp=torch.from_numpy(z['logp']);val=torch.from_numpy(z['value'])
                    pe=ve=0.
                    for start in range(0,len(frm),64):
                        heads=actors[k](data[start:start+64])
                        logp=torch.stack([h.log_softmax(-1).gather(1,actions[start:start+64,idx:idx+1]).flatten()for idx,h in enumerate(heads[:4])],1)
                        pe=max(pe,float((logp-lp[start:start+64]).abs().max()))
                        ve=max(ve,float((heads[4].flatten()-val[start:start+64]).abs().max()))
                    assert max(pe,ve)<4e-5
                with np.load(folder/'ppo-reward.npz') as z:
                    assert len(z['life_loss_action'])==screen['miss'] and z['terminated'][-1] and z['terminated'].sum()==1
                out.append(dict(role=role,seed=seed,arm=k,actor_sha256=receipt['policy_sha256'],
                    decisions=receipt['actions'],stage2=receipt['stage2_reached'],last_hud_spares=receipt['last_hud_spares'],
                    last_hud_bombs=receipt['last_hud_bombs'],original_result=screen,max_cpu_logp_error=pe,max_cpu_value_error=ve))
                rows.append((folder,k,seed))
    held=[r for r in out if r['role']=='held']
    differences=[]
    for s in plan['held']:
        v=[next(r for r in held if r['arm']==n and r['seed']==s)for n in names]
        differences.append(dict(seed=s,stage2=[r['stage2'] for r in v],score_change=v[1]['original_result']['score']-v[0]['original_result']['score'],
            game_progress_percentage_points=round(v[1]['original_result']['progress_pct']-v[0]['original_result']['progress_pct'],2)))
    panel=Image.new('RGB',(830,270*4))
    for j,(folder,k,seed) in enumerate(rows):
        x,y=(j%2)*415,(j//2)*270
        panel.paste(Image.open(folder/'settled-scorecard.png').crop((25,90,440,333)),(x,y+22))
        ImageDraw.Draw(panel).text((x+10,y+2),f'{folder.name}  |  {k}',fill='white')
    panel.save(root/'eight-original-scorecards.png')
    (root/'result.json').write_text(json.dumps(dict(train_games=4,independent_held_pairs=2,arm_order=names,credits=out,
        last_layer_policy_head_weight_l2_change=head_l2,paired_original_differences=differences,
        interpretation='Two held pairs verify the first learning and deployment update; uncertainty is large. Controller seeds do not fix the native game RNG.'),indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('parent',type=Path)
    a=p.parse_args();run(a.root,a.parent)
