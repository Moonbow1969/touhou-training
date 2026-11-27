"""Pair two imitation models on identical human-held stages, with player-level uncertainty."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.checkpoint_storage import digest


def counts(model, names, condition):
    rows=[model['metrics'][name]['visual_sft'][condition] for name in names]
    n=sum(r['n'] for r in rows)
    if not n:return dict(n=0,correct=0,accuracy=None,nll=None)
    correct=sum(r['correct'] for r in rows)
    return dict(n=n,correct=correct,accuracy=correct/n,nll=sum(r['n']*r['movement_nll'] for r in rows if r['n'])/n)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('reference','candidate','out'):
        p.add_argument('--'+key,required=True,type=Path)
    a=p.parse_args()
    models=[json.loads((d/'report.json').read_bytes()) for d in (a.reference,a.candidate)]
    if models[0]['held']!=models[1]['held'] or any(digest(d/'policy.ts')!=r['torchscript_sha256'] for d,r in zip((a.reference,a.candidate),models)):
        raise ValueError('Checkpoint or held stages differ')
    people={}
    for r in models[0]['held']:people.setdefault(r['group'],[]).append(r['name'])
    names=[r['name'] for r in models[0]['held']]
    results={}
    rng=np.random.default_rng(261128)
    for cond in ('all','change','start','turn'):
        by_person={k:[counts(model,v,cond) for model in models] for k,v in people.items()}
        nll=np.array([y['nll']-x['nll'] for x,y in by_person.values() if x['n']])
        acc=np.array([y['accuracy']-x['accuracy'] for x,y in by_person.values() if x['n']])
        idx=rng.integers(len(acc),size=(20000,len(acc))) if len(acc) else None
        results[cond]=dict(pooled=[counts(model,names,cond) for model in models],by_pilot=by_person,
            mean_pilot_accuracy_change=float(acc.mean()) if len(acc) else None,
            pilot_bootstrap95_accuracy_change=np.quantile(acc[idx].mean(1),[.025,.975]).tolist() if len(acc) else None,
            mean_pilot_nll_change=float(nll.mean()) if len(nll) else None,
            pilot_bootstrap95_nll_change=np.quantile(nll[idx].mean(1),[.025,.975]).tolist() if len(nll) else None)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'paired-human.json').write_text(json.dumps(dict(
        count_unit='Four-tick human action window. Pooled metrics weight windows; paired uncertainty resamples the distinct held players with equal weight. Training pilots and presentation budgets both changed.',
        train_pilots=[len({r['group'] for r in m['train']}) for m in models],
        held_pilots=sorted(people),held_stages=len(names),checkpoints=[m['torchscript_sha256'] for m in models],results=results),indent=2)+'\n',encoding='utf8')
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    for ax,metric,label in zip(axes,('accuracy','nll'),('Movement accuracy','Movement negative log likelihood')):
        for i,(cond,r) in enumerate(results.items()):
            pair=[m[metric] for m in r['pooled']]
            ax.plot(pair,[i,i],color='#bdc4c8',lw=3)
            ax.scatter(pair,[i,i],s=60,c=['#435764','#157469'])
        ax.set_yticks(range(4),['All actions','Change direction or stop/start','Start moving','Turn while moving'])
        ax.invert_yaxis();ax.set_xlabel(label);ax.grid(axis='x',alpha=.2);ax.spines[['top','right','left']].set_visible(False)
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],marker='o',ls='',color=c,label=f"{len({r['group'] for r in m['train']})} training pilots")
             for c,m in zip(('#435764','#157469'),models)]
    fig.legend(handles=handles,loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle(f'Human replay control prediction on {len(people)} held pilots',fontsize=12)
    fig.savefig(a.out/'paired-human.png',dpi=165);plt.close(fig)


if __name__=='__main__':
    main()
