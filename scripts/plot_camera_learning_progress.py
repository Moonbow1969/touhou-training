"""Plot within-cohort paired changes across successive original-game CNN PPO generations."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    definitions=[('SFT → PPO1','camera-evaluation/camera-ppo.json'),
                 ('PPO1 → PPO2','camera-round2/evaluation/camera-ppo2.json'),
                 ('PPO2 → PPO5','campaign/g05/evaluation/camera-ppo5.json')]
    available=[(label,json.loads((a.root/path).read_bytes())) for label,path in definitions if (a.root/path).is_file()]
    if not available:raise ValueError('No fully matched native cohort')
    a.out.mkdir(parents=True,exist_ok=True)
    summaries=[];fig,axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained',sharex=True,sharey=True)
    for ax,t in zip(axes,('10','12','13')):
        for i,(label,table) in enumerate(available):
            r=table[t]
            old=next(v for k,v in r.items() if k!='candidate' and isinstance(v,dict) and 'n' in v)
            new=r['candidate'];delta=r['mean_paired_wall_difference'];lo,hi=r['mean_paired_wall_difference_fixed_host_bootstrap95']
            summaries.append(dict(comparison=label,title=int(t),n_pairs=new['n'],parent_stage2=old['stage2'],candidate_stage2=new['stage2'],
                                  parent_mean_s=old['mean_seconds'],candidate_mean_s=new['mean_seconds'],
                                  difference_s=delta,paired_95_s=[lo,hi],parent_terminal_bombs=old['mean_terminal_bombs'],
                                  candidate_terminal_bombs=new['mean_terminal_bombs'],parent_terminal_power=old['mean_terminal_power'],
                                  candidate_terminal_power=new['mean_terminal_power']))
            ax.plot([lo,hi],[i,i],lw=2.5,color='#30746f');ax.scatter(delta,i,s=46,color='#124b50',zorder=3)
            ax.annotate(f"{delta:+.1f} s   ·   {new['n']} pairs",(delta,i),xytext=(0,11),textcoords='offset points',ha='center',fontsize=8)
        ax.set_title(f'TH{t}');ax.axvline(0,color='#84909a',ls='--',lw=1);ax.grid(axis='x',alpha=.15)
        ax.set_ylim(-.5,len(available)-.36);ax.spines[['top','right','left']].set_visible(False)
        ax.set_xlabel('Change in seconds to credit end')
    axes[0].set_yticks(np.arange(len(available)),[v[0] for v in available]);axes[0].invert_yaxis()
    fig.suptitle('Original-game whole credits  ·  paired wall time and 95% bootstrap intervals')
    fig.savefig(a.out/'paired-generations.png',dpi=175);plt.close(fig)
    (a.out/'paired-generations.json').write_text(json.dumps(summaries,indent=2)+'\n')


if __name__=='__main__':main()
