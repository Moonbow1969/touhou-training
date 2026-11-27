"""Compare supervised budgets with players as the independent sampling units."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.checkpoint_storage import digest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',nargs=2,required=True,type=Path)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    arms=[json.loads((f/'report.json').read_bytes()) for f in a.models]
    if arms[0]['train']!=arms[1]['train'] or arms[0]['held']!=arms[1]['held'] or arms[0]['encoder']!=arms[1]['encoder']:
        raise ValueError('Model budgets do not share their data and initialization')
    if any(r['torchscript_sha256']!=digest(f/'policy.ts') for r,f in zip(arms,a.models)):raise ValueError('Frozen model differs')
    groups=defaultdict(list)
    for r in arms[0]['held']:groups[r['group']].append(r['name'])
    def scores(arm,names):
        rows=[arm['metrics'][name]['visual_sft'] for name in names]
        def aggregate(key,cond):
            n=sum(row[cond]['n'] for row in rows)
            if key=='correct':return sum(row[cond][key] for row in rows)/n
            return sum(row[cond][key]*row[cond]['n'] for row in rows)/n
        return dict(movement_nll=aggregate('movement_nll','all'),movement_accuracy=aggregate('correct','all'),
                    change_accuracy=aggregate('correct','change'),start_accuracy=aggregate('correct','start'))
    pilots={group:[scores(arm,names) for arm in arms] for group,names in groups.items()}
    rng=np.random.default_rng(19761)
    draws=rng.integers(len(pilots),size=(20000,len(pilots)))
    diff={}
    for k in next(iter(pilots.values()))[0]:
        values=np.array([right[k]-left[k] for left,right in pilots.values()])
        diff[k]=dict(mean_player_delta=values.mean(),paired_player_bootstrap95=np.quantile(values[draws].mean(1),[.025,.975]).tolist(),
                     players_increasing=int((values>0).sum()),players_decreasing=int((values<0).sum()))
    a.out.mkdir(parents=True,exist_ok=True)
    names=list(arms[0]['metrics'])
    report=dict(steps=[r['steps'] for r in arms],source_windows=sum(r['decisions'] for r in arms[0]['train']),
                held_windows=sum(r['decisions'] for r in arms[0]['held']),checkpoints=[r['torchscript_sha256'] for r in arms],
                pooled=[scores(arm,names) for arm in arms],by_player=pilots,paired=diff,
                interval='20,000 resamples of held players, carrying all of each player’s stage data together. Six players give limited external precision.')
    (a.out/'paired.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(8.3,4),layout='constrained')
    for ax,k,title in zip(axes,('movement_nll','movement_accuracy'),('Movement log loss','Movement accuracy')):
        for left,right in pilots.values():ax.plot([0,1],[left[k],right[k]],'o-',color='#89949b',alpha=.7,lw=1)
        means=np.mean([[left[k],right[k]] for left,right in pilots.values()],axis=0)
        ax.plot([0,1],means,'o-',color='#0e6463',lw=3,label='Equal-player mean')
        ax.set(xticks=[0,1],xticklabels=[f"{r['steps']:,} updates" for r in arms],title=title,xlim=(-.12,1.12))
        ax.grid(axis='y',alpha=.25)
    axes[0].legend(frameon=False)
    fig.suptitle(f'Four-title Hard and Lunatic SFT · {len(pilots)} held players from TH10, TH12 and TH13')
    fig.savefig(a.out/'paired.png',dpi=165)
    plt.close(fig)
    print(json.dumps(diff),flush=True)


if __name__=='__main__':main()
