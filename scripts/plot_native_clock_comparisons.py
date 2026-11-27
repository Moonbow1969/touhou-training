"""Visualize paired changes in original stage-script ticks with host-matched bootstrap intervals."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.score_native_stage_clock import summarize


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    titles={'TH13 · online training title':[], 'TH14 · development transfer title':[]}
    th13=[
        ('SFT → PPO1','camera-evaluation','camera-sft','camera-ppo'),
        ('PPO1 → PPO2','camera-round2/evaluation','camera-ppo','camera-ppo2'),
        ('PPO2 → PPO5','campaign/g05/evaluation','camera-ppo2','camera-ppo5'),
        ('Joint SFT → PPO1','cross-title-online/evaluation','cross-sft','cross-ppo'),
        ('Joint PPO1 → PPO3','cross-campaign/g03/evaluation','camera-cross-ppo','camera-cross-ppo3'),
        ('Progress bonus','native-progress-ablation/evaluation','ppo2','ppo2-progress'),
        ('Replay KL','replay-anchor/evaluation','plain-cross-ppo','anchored-cross-ppo'),
        ('Native-time reward','clock-reward/evaluation','wall-clock','native-clock')]
    for name,folder,left,right in th13:
        file=a.root/folder/'credits.json'
        if file.is_file():
            res=summarize(json.loads(file.read_bytes()),13,[left,right])
            titles['TH13 · online training title'].append((name,res['n'],res['mean_paired_change'],res['paired_fixed_host_bootstrap95']))
    th14=[('SFT → PPO1','primary'),('PPO1 → PPO2','ppo2-comparison'),('TH13-only → joint SFT','transfer-mixture'),
          ('Joint SFT → PPO1','transfer-online'),('Replay KL','replay-anchor'),
          ('Broader SFT corpus','v2-supervised-corpus'),('3k → 9k SFT updates','v2-supervised-budget')]
    for name,folder in th14:
        file=a.root/'zero-shot-th14'/folder/'paired.json'
        if file.is_file():
            r=json.loads(file.read_bytes());n=next(iter(r['arms'].values()))['n']
            titles['TH14 · development transfer title'].append((name,n,r['mean_paired_stage1_tick_difference'],r['paired_bootstrap95_stage1_ticks']))
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,(title,rows) in zip(axes,titles.items()):
        ax.set_title(title);ax.axvline(0,ls='--',lw=1,color='#929ca1')
        for i,(name,n,delta,(lo,hi)) in enumerate(rows):
            ax.plot([lo,hi],[i,i],lw=2,color='#4a6f82');ax.scatter(delta,i,color='#154759',s=38,zorder=3)
        ax.set_yticks(range(len(rows)),[f'{r[0]} ({r[1]} pairs)' for r in rows]);ax.invert_yaxis()
        ax.set_xlabel('Change in furthest native Stage 1 tick');ax.grid(axis='x',alpha=.17)
        ax.spines[['right','top','left']].set_visible(False)
    fig.suptitle('Original whole credits   ·   paired mean and 95% bootstrap interval')
    a.out.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.out/'native-stage-progress.png',dpi=175);plt.close(fig)
    (a.out/'native-stage-progress.json').write_text(json.dumps(titles,indent=2)+'\n')


if __name__=='__main__':
    main()
