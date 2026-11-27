"""Summarize Stage 2, resources and physical controls in each independently evaluated game/rank cell."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.checkpoint_storage import digest
from scripts.score_native_full_actions import qualified_stage2
from scripts.score_original_camera_run import records
from scripts.verify_th17_original_credit import inspect as check_th17


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out-stem',default='whole-game-controls',choices=('whole-game-controls','whole-game-controls-with-playable-stock'))
    args=p.parse_args();root=args.root.resolve()
    evaluation=root/'evaluation'
    file=evaluation/'credits.json';rows=json.loads(file.read_bytes())
    plan=json.loads((root/'camera-protocol.json').read_bytes())
    old_th17={r['native_log_sha256']:r for r in rows if r['title']==17 and 'last_playable_bombs' not in r}
    if old_th17:
        for title,path in records(root,True,True):
            if title!=17:continue
            original=old_th17.pop(digest(path),None)
            if original is not None:
                original['last_playable_bombs']=check_th17(path,json.loads(path.read_bytes()),original['difficulty'])['last_playable_bombs']
        if old_th17:raise ValueError('Missing TH17 playable resource evidence')
    arms=(plan['reference_arm'],plan['candidate']);cells=defaultdict(lambda:defaultdict(dict))
    for r in rows:
        t,rank,seed,arm=(r[k] for k in ('title','difficulty','seed','arm'))
        if arm not in arms or seed not in plan['eval_seeds'][str(t)] or rank!=plan['rank_by_seed'][str(t)][str(seed)]:
            raise ValueError('Unexpected physical held source')
        if qualified_stage2(r)!=r['stage2_reached'] or not r['original_endpoint_sha256']:raise ValueError('Missing native endpoint evidence')
        if seed in cells[(t,rank)][arm]:raise ValueError('Repeated original game')
        cells[(t,rank)][arm][seed]=r
    summary={};markers=[];labels=[]
    for (title,rank),items in sorted(cells.items()):
        seeds=sorted(items[arms[0]])
        expected=sorted(s for s in plan['eval_seeds'][str(title)] if plan['rank_by_seed'][str(title)][str(s)]==rank)
        if seeds!=expected or seeds!=sorted(items[arms[1]]):raise ValueError('Missing registered pair')
        for s in seeds:
            a,b=(items[arm][s] for arm in arms)
            if (a['host'],a['game_sha256'])!=(b['host'],b['game_sha256']):raise ValueError('Incomparable physical pair')
        out={}
        for arm in arms:
            group=[items[arm][s] for s in seeds]
            def inventory(k):
                vals=[r[k] for r in group if r.get(k) is not None]
                return None if not vals else dict(n=len(vals),mean=float(np.mean(vals)),observed=vals)
            movement=np.array([r['action_counts'][0] for r in group])
            n=movement.sum(1)
            if np.any(n!=np.array([r['decisions'] for r in group])):raise ValueError('Inconsistent native control denominator')
            out[arm]=dict(n=len(group),stage2=sum(r['stage2_reached'] is True for r in group),
                stage_unknown=sum(r['stage2_reached'] is None for r in group),terminal_spares=inventory('terminal_spares'),
                terminal_bombs=inventory('terminal_bombs'),last_playable_bombs=inventory('last_playable_bombs'),terminal_power=inventory('terminal_power'),
                effective_casts=sum(r['effective_bombs'] for r in group),
                decisions=int(n.sum()),fraction_with_vertical_component=float(movement[:,[0,1,2,6,7,8]].sum()/n.sum()),
                fraction_diagonal=float(movement[:,[0,2,6,8]].sum()/n.sum()),
                fraction_neutral=float(movement[:,4].sum()/n.sum()),native_receipts=[r['native_log_sha256'] for r in group])
        key=f"TH{title:02} "+{2:'Hard',3:'Lunatic'}[rank]
        summary[key]=out;labels.append(key);markers.append(out)
    fig,axs=plt.subplots(1,3,figsize=(12.5,max(4.5,len(labels)*.53)),sharey=True,layout='constrained')
    y=np.arange(len(labels));colors=['#3c7793','#b25d39']
    for j,arm in enumerate(arms):
        offset=(-.13,.13)[j];v=[m[arm] for m in markers]
        def markers_for(ax,values):
            ax.scatter(values,y+offset,label=arm,s=31,c=colors[j],marker=('o','D')[j])
        markers_for(axs[0],[100*x['stage2']/x['n'] for x in v])
        markers_for(axs[1],[x['effective_casts']/x['n'] for x in v])
        markers_for(axs[2],[100*x['fraction_with_vertical_component'] for x in v])
    axs[0].set_yticks(y,labels);axs[0].invert_yaxis()
    for ax,label in zip(axs,('Confirmed Stage 2 (%)','Effective casts per game','Decisions with vertical input (%)')):
        ax.set_xlabel(label,fontsize=9);ax.grid(axis='x',alpha=.2)
    axs[0].set_xlim(-4,104);axs[2].set_xlim(-4,104)
    axs[0].legend(loc='lower right',fontsize=8)
    fig.suptitle(f'Ordinary full stock · {len(rows)} original games · {len(rows)//2} matched pairs')
    fig.savefig(evaluation/(args.out_stem+'.png'),dpi=160);plt.close(fig)
    out=dict(credits_sha256=digest(file),reference=arms[0],candidate=arms[1],complete_credits=len(rows),cells=summary)
    (evaluation/(args.out_stem+'.json')).write_text(json.dumps(out,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print('Full original credits',len(rows),[(arm,sum(m[arm]['stage2'] for m in markers)) for arm in arms],flush=True)

if __name__=='__main__':
    main()
