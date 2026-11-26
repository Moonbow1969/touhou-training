"""Summarize independent TH13 native credits whose visual outcome was adjudicated."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();rows=[]
    for f in sorted(args.run.glob('pg-*/label.json')):
        credit=json.loads((f.parent/'credit.json').read_text());label=json.loads(f.read_text())
        data=np.load(f.parent/'behavior.npz');r=credit['round']
        phase='training' if r<=4 else 'solo_frozen_evaluation' if r<=6 else 'external_joint_evaluation'
        rows.append(dict(credit=f.parent.name,phase=phase,model=credit['model'],seed=credit['seed'],round=r,
                         policy_fingerprint=credit['weights_sha256'],decisions=credit['decisions'],duration_seconds=credit['actual_seconds'],
                         action_counts=credit['action_count'],observed_loss_reward=label['reward'],status=label['status'],evidence=label['note'],
                         held_spares_at_cutoff=credit['hud']['endpoint']['spare_lives'],held_bombs_at_cutoff=credit['hud']['endpoint']['bombs'],
                         mean_logp=round(float(data['logp'].mean()),6)))
    groups=defaultdict(list)
    for r in rows:groups[(r['phase'],r['model'])].append(r)
    summary={}
    for (phase,model),credits in groups.items():
        vals=[r['observed_loss_reward'] for r in credits if r['status'] in ['active','resolved_trance','game_over']]
        summary[phase+'/'+model]=dict(credits=len(credits),valid=len(vals),returns=vals,mean_return=float(np.mean(vals)) if vals else None,
                                     decisions=sum(r['decisions'] for r in credits),no_observed_loss=sum(v==0 for v in vals))
    s={'title':'TH13 official Trial 0.01a','host':'Endeavour','difficulty':'Normal','player':'Hakurei Reimu','native_ticks_qualified':False,
       'horizon':'60s wall, independent original Stage 1 credit; adjudicated automatic death-trance settlement up to +12s',
       'real_training':'on-policy REINFORCE, native terminal resource signal, image-only categorical horizontal movement; fixed Shoot, no Bomb',
       'unit':'complete credit, original RNG not controlled','groups':summary,'credits':rows}
    args.output.parent.mkdir(exist_ok=True,parents=True);args.output.write_text(json.dumps(s,indent=2)+'\n')
    print(json.dumps(summary))
if __name__=='__main__':main()
