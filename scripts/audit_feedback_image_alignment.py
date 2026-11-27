"""Measure whether the qualified movement model uses aligned images beyond action persistence."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.checkpoint_storage import digest
from scripts.train_causal_replay_history import paths
from scripts.train_camera_direction_feedback import score
from scripts.train_five_original_games import freeze
from scripts.train_human_camera_policy import load_replay


def metrics(matrix, direction, prior, clock):
    matrix=matrix.astype(np.float64)
    matrix/=matrix.sum(axis=2,keepdims=True)
    marginal=np.empty((len(direction),9))
    for i,t in enumerate(clock):
        marginal[i]=matrix[i,4] if i==0 or t-clock[i-1]!=4 else marginal[i-1] @ matrix[i]
    return dict(conditional=score(matrix[np.arange(len(direction)),prior],direction,prior),
                autonomous=score(marginal,direction,prior))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    d=p.parse_args().data.resolve()
    source=d/'four-title-direction-feedback'
    root=source/'image-alignment'
    root.mkdir(exist_ok=True)
    observed=json.loads((source/'report.json').read_bytes())
    plan=json.loads((source/'protocol.json').read_bytes())
    model_file=source/'feedback/policy.ts'
    freeze(root/'design.json',dict(origin='Post-hoc explanatory control after the registered feedback comparison',
        registration_sha256=digest(source/'protocol.json'),feedback_sha256=digest(model_file),
        action_only='Estimate nine-by-nine human transition counts with equal total weight per TRAIN game and rank. Add 0.5 per cell to the effective training counts. One model sees only the preceding direction.',
        image_alignment='Within each held stage permute the learned image-specific transition matrices by one-half of the complete retained stage. Preserve the true previous direction and each stage-specific marginal distribution of images. This destroys their timing relative to human actions. Evaluate with the same episode boundaries and rows as the original development comparison.',
        outcomes='NLL for all directions and true changes, conditioning on human prior or propagating own model distribution on the fixed human images. Original-game performance is measured elsewhere.'))
    counts={}
    held={}
    locator=paths(d)
    for sha,r in observed['split'].items():
        row=load_replay(locator[sha])
        if row['sha256']!=sha:
            raise ValueError('Changed human stage')
        action=row['action'][:,0]
        prev=row['prior'][:,0]
        clock=row['native_frame']
        keep=np.flatnonzero((np.diff(clock)==4)&(prev[1:]==action[:-1]))+1
        if len(keep)!=r['usable']:
            raise ValueError('Eligible human decisions differ')
        if r['role']=='train':
            cell=(r['title'],r['rank'])
            counts.setdefault(cell,np.zeros((9,9)))
            counts[cell]+=np.bincount(9*prev[keep]+action[keep],minlength=81).reshape(9,9)
        else:
            held[sha]=(row['raw'][keep],action[keep],prev[keep],clock[keep],r['title'])
    total=sum(c.sum() for c in counts.values())
    balanced=sum(c/c.sum() for c in counts.values()) * (total/len(counts)) + .5
    trans=balanced/balanced.sum(axis=1,keepdims=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(2)
    net=torch.jit.load(str(model_file),map_location=device).eval()
    # One encoder pass per image; the residual then evaluates all nine possible prior actions.
    results={}
    with torch.inference_mode():
        for sha,(raw,action,prior,clock,title) in held.items():
            chunks=[]
            for x in torch.from_numpy(raw).split(64):
                x=x.to(device).float()/255
                z=net.features(x)
                move=net.head(z)[:,:9]
                chunks.append(torch.stack([net.residual(z,move,torch.full((len(x),),k,device=device,dtype=torch.long)).softmax(1)
                                           for k in range(9)],dim=1).cpu().numpy())
            aligned=np.concatenate(chunks)
            result={arm:metrics(m,action,prior,clock) for arm,m in (
                ('aligned',aligned),('half_stage_shift',np.roll(aligned,len(action)//2,axis=0)),
                ('action_only',np.broadcast_to(trans,aligned.shape)))}
            for mode in ('conditional','autonomous'):
                measure=result['aligned'][mode]['all']
                expected=observed['outcomes']['feedback']['stages'][sha][mode]['all']
                if measure['rows']!=expected['rows'] or abs(measure['nll_sum']-expected['nll_sum'])/measure['rows']>1e-4:
                    raise ValueError('Exported original actor disagrees with the trained development model')
            results[sha]=result
    summaries={}
    for title,files in plan['held'].items():
        summaries[title]={}
        for arm in ('aligned','half_stage_shift','action_only'):
            summaries[title][arm]={}
            for mode in ('conditional','autonomous'):
                table=summaries[title][arm][mode]={}
                for subset in ('all','change','start','turn'):
                    values=[results[f][arm][mode][subset] for f in files]
                    n=sum(v['rows'] for v in values)
                    table[subset]=dict(n=n,nll=sum(v['nll_sum'] for v in values)/n,accuracy=sum(v['correct'] for v in values)/n)
    report=dict(design=digest(root/'design.json'),action_only_counts_by_stratum={str(k):c.tolist() for k,c in counts.items()},
                transition_probability=trans.tolist(),by_title=summaries,by_held_stage=results)
    (root/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for title,t in summaries.items():
        print(title,{arm:round(t[arm]['autonomous']['change']['nll'],3) for arm in t},flush=True)


if __name__=='__main__':
    main()
