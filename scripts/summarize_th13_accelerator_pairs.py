"""Summarize preregistered matched TH13 camera credits using original stocks and script ticks."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from scripts.collect_th10_full_action import save


def summarize(root,experiment_name='v2-online'):
    experiment=root.resolve()/experiment_name;out=experiment/'native13'
    plan=json.loads((experiment/'camera-protocol.json').read_bytes())
    seeds=plan['eval_seeds']['13'];rows={};groups={}
    for name in (plan['reference_arm'],plan['candidate']):
        arm=[]
        for seed in seeds:
            folder=out/f'held-{name}-{seed}'
            log=json.loads((folder/'receipt.json').read_bytes())
            assert log['host']=='accelerator' and log['seed']==seed and log['role']=='held'
            assert log['terminal'] and not log['continues'] and log['remaining_lives']==-1
            assert hashlib.sha256((folder/'rollout.npz').read_bytes()).hexdigest()==log['sha256']
            raw=np.fromfile(folder/'steps.bin','<f4').reshape(-1,143)
            assert raw.shape[0]==log['steps'] and raw[0,[125,127,129,141]].tolist()==[1,2,2,1]
            assert raw[-1,120] and not raw[:-1,120].any()
            ticks=np.maximum(0,np.diff(np.r_[raw[0,134],raw[:,135]]))
            ticks[raw[:,125]!=raw[:,126]]=0
            life=np.flatnonzero(raw[:,136]>0)
            movement=raw[:,110];switches=int(np.count_nonzero(np.diff(movement)))
            arm.append(dict(seed=seed,host=log['host'],policy_sha256=log['policy_sha256'],game_sha256=log['game_sha256'],
                stage2=log['stage2'],stage2_spares=log['stage2_entry_spares'],stage2_bombs=log['stage2_entry_bombs'],
                spares=log['remaining_lives'],bombs=log['remaining_bombs'],casts=log['effective_bombs'],
                losses=log['confirmed_losses'],earned_life=log['life_earned'],whole_credit_wall_s=log['wall_seconds'],
                original_script_ticks=int(ticks.sum()),first_loss_script_ticks=int(ticks[:life[0]+1].sum()),
                first_loss_s=float(raw[life[0],121]),original_display_score=int(raw[-1,131])*10,
                decisions=log['steps'],native_key_match=log['native_key_match'],x11_key_match=log['x_key_match'],
                movement_switches=switches,movement_transitions=len(movement)-1))
        assert len(set(a['policy_sha256'] for a in arm))==1
        rows[name]=arm
        groups[name]=dict(credits=len(arm),reached_stage2=sum(r['stage2'] for r in arm),terminal_spares=[r['spares'] for r in arm],
            mean_remaining_bombs=float(np.mean([r['bombs'] for r in arm])),confirmed_casts=sum(r['casts'] for r in arm),
            mean_wall_s=float(np.mean([r['whole_credit_wall_s'] for r in arm])),mean_native_ticks=float(np.mean([r['original_script_ticks'] for r in arm])),
            mean_first_loss_native_ticks=float(np.mean([r['first_loss_script_ticks'] for r in arm])),
            mean_display_score=float(np.mean([r['original_display_score'] for r in arm])),
            switch_rate=sum(r['movement_switches'] for r in arm)/sum(r['movement_transitions'] for r in arm),
            decisions=sum(r['decisions'] for r in arm),native_key_matches=sum(r['native_key_match'] for r in arm))
        n=len(arm);proportion=groups[name]['reached_stage2']/n;z=1.959963984540054
        center=(proportion+z*z/(2*n))/(1+z*z/n)
        radius=z*math.sqrt(proportion*(1-proportion)/n+z*z/(4*n*n))/(1+z*z/n)
        groups[name]['stage2_wilson_95pct']=[max(0.,center-radius),min(1.,center+radius)]
    comparisons={};baseline,candidate=list(rows.values());rng=np.random.default_rng(seeds[0])
    draws=rng.integers(len(seeds),size=(20000,len(seeds)))
    for key in ('stage2','whole_credit_wall_s','original_script_ticks','first_loss_script_ticks','original_display_score','casts','bombs'):
        difference=np.array([float(b[key])-float(a[key]) for a,b in zip(baseline,candidate)])
        comparisons[key]=dict(mean_delta=float(difference.mean()),per_seed=difference.tolist(),positive=int((difference>0).sum()),
            paired_percentile_95pct=np.quantile(difference[draws].mean(1),[.025,.975]).tolist())
    save(out/'paired.json',dict(design='Twelve controller seeds matched on physical Accelerator, Normal official TH13 0.01a trial, all original stocks and sampled movement/focus/shoot/Bomb actions, no Continue; original RNG uncontrolled.',
        native_clock='Nonnegative advances of the stage-local native clock. Stage-entry reset intervals have zero contribution.',
        uncertainty='Exploratory paired seed bootstrap, 20,000 draws, unadjusted percentile intervals.',seeds=seeds,groups=groups,comparisons=comparisons,credits=rows))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--experiment',choices=('v2-online','attention-online'),default='v2-online')
    args=p.parse_args();summarize(args.root,args.experiment)
