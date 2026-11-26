"""Recompute paired native ranking-tick, usable-stock and wall-time outcomes from sealed original credits."""
import argparse,itertools,json,math
from pathlib import Path
import numpy as np

def compare(root, arms, seeds, rng_seed):
    cohorts={}
    for arm in arms:
        cohort={}
        for seed in seeds:
            folder=root/f'held-{arm}-{seed}';receipt=json.loads((folder/'receipt.json').read_bytes())
            end_file=folder/'native-terminal.json'
            if end_file.exists():
                end=json.loads(end_file.read_bytes());tick=end['native_stage_tick'];score=end['display_score']
            else:
                tick=receipt['native_ranking_tick'];score=receipt['ranking_display_score']
            if not receipt['terminal'] or receipt['continues']!=0:raise ValueError(folder)
            with np.load(folder/'rollout.npz') as z: actor_tick=int(z['native_frame'].max())
            cohort[seed]=[tick,actor_tick,receipt['wall_seconds'],score,receipt['last_playable_bombs']]
        cohorts[arm]=cohort
    indices=np.random.default_rng(rng_seed).integers(0,len(seeds),size=(100000,len(seeds)))
    result={'population':'Full native No-Continue credits on one host, matching controller seeds across compared checkpoints/action cadences, original game RNG uncontrolled. Count stage clock at original ranking after all keys release.','rng_seed':rng_seed,'resamples':len(indices),'pairs':[]}
    for first,second in itertools.combinations(arms,2):
        delta=np.array([np.subtract(cohorts[second][k],cohorts[first][k]) for k in seeds])
        pair=dict(reference=first,challenger=second,n=len(seeds),per_seed=[dict(seed=seed,reference_native_ranking_tick=cohorts[first][seed][0],challenger_native_ranking_tick=cohorts[second][seed][0],native_delta=int(delta[i,0]),wall_delta=float(delta[i,2])) for i,seed in enumerate(seeds)])
        for i,key in enumerate(['native_ranking_tick','last_actor_stage_tick','policy_wall_seconds','native_ranking_score','last_playable_bombs']):
            v=delta[:,i];w=int((v>0).sum());l=int((v<0).sum())
            signp=min(1.,2*sum(math.comb(w+l,j) for j in range(min(w,l)+1))/2**(w+l))
            pair[key]=dict(mean_delta=float(v.mean()),median_delta=float(np.median(v)),paired_percentile_bootstrap_95=np.percentile(v[indices].mean(1),[2.5,97.5]).tolist(),wins=w,losses=l,ties=len(seeds)-w-l,two_sided_exact_sign_p=signp)
        pair['wall_and_native_delta_same_sign']=int((np.sign(delta[:,0])==np.sign(delta[:,2])).sum())
        result['pairs'].append(pair)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('output',type=Path);p.add_argument('--arms',nargs='+',required=True)
    p.add_argument('--start-seed',type=int,required=True);p.add_argument('--n',type=int,required=True);p.add_argument('--rng-seed',type=int,default=261125);args=p.parse_args()
    args.output.write_text(json.dumps(compare(args.root,args.arms,range(args.start_seed,args.start_seed+args.n),args.rng_seed),indent=2)+'\n')
