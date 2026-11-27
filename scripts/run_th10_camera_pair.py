"""Counterbalance two frozen TorchScript policies on fresh, same-host TH10 original credits."""
import argparse
from pathlib import Path
import numpy as np
import torch

from scripts.run_th10_camera_cohort import collect,checkpoint,verify,digest,save,read_arm,summarize,draw


def switches(folder,host,seeds):
    rows=[]
    for seed in seeds:
        actual_host=('accelerator' if seed%2 else 'momentum') if host=='fleet' else host
        with np.load(folder/f'{actual_host}-{seed}/behavior.npz') as b:
            direction=b['action'][:,0]
            logits=b['camera_logits'][1:,:9].astype(np.float64)
        weights=np.exp(logits-logits.max(1,keepdims=True))
        probs=weights/weights.sum(1,keepdims=True)
        expected=(1-probs[np.arange(len(probs)),direction[:-1]]).sum()
        changed=direction[:-1]!=direction[1:]
        rows.append(dict(seed=seed,n=len(changed),switches=int(changed.sum()),rate=float(changed.mean()),
            expected_switches=float(expected),expected_rate=float(expected/len(changed)),
            stationary_to_motion=int(((direction[:-1]==4)&(direction[1:]!=4)).sum()),
            motion_to_stationary=int(((direction[:-1]!=4)&(direction[1:]==4)).sum()),
            turns_between_non_neutral_directions=int(((direction[:-1]!=4)&(direction[1:]!=4)&changed).sum())))
    n=sum(r['n'] for r in rows);count=sum(r['switches'] for r in rows)
    expected=sum(r['expected_switches'] for r in rows)
    return dict(credits=rows,n=n,switches=count,pooled_rate=count/n,expected_switches=expected,expected_pooled_rate=expected/n)


def run(args):
    if args.reference_name==args.candidate_name:
        raise ValueError('The reference and candidate require distinct arm names.')
    for name in (args.reference_name,args.candidate_name):
        if Path(name).name!=name or name in ('.','..',''):raise ValueError('Arm names must be directory basenames.')
    if len(args.seeds)<2 or len(args.seeds)%2 or len(set(args.seeds))!=len(args.seeds):
        raise ValueError('Supply an even number of distinct controller seeds.')
    if args.concurrency<1:raise ValueError('Concurrency must be positive.')
    args.workers=args.workers.resolve()
    args.out=args.out.resolve()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    arms=[(args.reference_name,args.reference.resolve()),(args.candidate_name,args.candidate.resolve())]
    first,second=np.array_split(args.seeds,2)
    first,second=first.tolist(),second.tolist()
    difficulty=getattr(args,'difficulty',1)
    difficulty_name={0:'Easy',1:'Normal',2:'Hard',3:'Lunatic'}[difficulty]
    design=(f'Original TH10 {difficulty_name} official trial stage 1 with stock lives, four sampled actions and no Continue. '
            f'One {args.host} host per matched pair. First half of controller seeds plays {args.reference_name} '
            f'first; second half plays {args.candidate_name} first. Game RNG and concurrent host load uncontrolled. '
            'The 9-way movement switch rate counts every change between consecutive decisions, including neutral, within each credit.')
    save(args.out/'paired-protocol.json',dict(design=design,host=args.host,seeds=args.seeds,
         first_reference=first,first_candidate=second,holds_ms=120,concurrency=args.concurrency))
    models={}
    for (name,path),block in [(arms[0],first),(arms[1],second),(arms[1],first),(arms[0],second)]:
        history=name==args.candidate_name and args.candidate_history
        if name not in models:
            if name==args.candidate_name and hasattr(args,"validate_candidate"):args.validate_candidate(path)
            models[name]=checkpoint(path,history)
        if len(models)==2 and digest(arms[0][1])==digest(arms[1][1]):
            raise ValueError('Reference and candidate files are identical.')
        slots=args.block_capacity() if hasattr(args,"block_capacity") else args.concurrency
        collect(args,name,path,block,args.out/name,slots,True,history)
    data={};movement={}
    for name,path in arms:
        folder=args.out/name
        verify(folder,path,models[name],args.seeds,'held',host=args.host,history=name==args.candidate_name and args.candidate_history,difficulty=difficulty)
        data[name]=read_arm([folder],sorted(args.seeds))
        movement[name]=switches(folder,args.host,args.seeds)
    a,b=movement[args.reference_name]['credits'],movement[args.candidate_name]['credits']
    dif=np.array([candidate['rate']-reference['rate'] for reference,candidate in zip(a,b)])
    movement['paired_mean_candidate_minus_reference']=float(dif.mean())
    boot=np.random.default_rng(args.seeds[0]).choice(dif,(10000,len(dif))).mean(1)
    movement['paired_bootstrap_95pct_interval']=np.quantile(boot,[.025,.975]).tolist()
    result=summarize(data,design,args.reference_name);result['movement_switches']=movement
    save(args.out/'paired.json',result);draw(data,args.out/'paired.png',difficulty_name)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workers','out','reference','candidate'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--reference-name',required=True);p.add_argument('--candidate-name',required=True)
    p.add_argument('--host',choices=('accelerator','momentum','fleet'),required=True)
    p.add_argument('--difficulty',choices=(0,1,2,3),type=int,default=1)
    p.add_argument('--stage-sensor',choices=('legacy','qualified-v2','qualified-v2-clock'))
    p.add_argument('--candidate-history',action='store_true')
    p.add_argument('--seeds',required=True,nargs='+',type=int)
    p.add_argument('--concurrency',type=int,default=3)
    run(p.parse_args())
