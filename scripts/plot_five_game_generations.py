"""Plot paired deployment effects of two PPO updates across original games and ranks."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.checkpoint_storage import digest
from scripts.score_original_camera_run import records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    d=p.parse_args().data.resolve()
    out=d/'learning-curves'
    games=(10,11,12,13,17)
    summaries={}
    for name in ('five-game-online','five-game-generation2'):
        study=d/name
        plan=json.loads((study/'camera-protocol.json').read_bytes())
        file=study/'evaluation/credits.json'
        scored=json.loads(file.read_bytes())
        by_hash={(r['title'],r['native_log_sha256']):r for r in scored}
        if len(scored)!=160 or len(by_hash)!=160:
            raise ValueError('Expected the sealed five-title full evaluation')
        buckets=defaultdict(dict)
        for t,path in records(study,True,True):
            row=by_hash.get((t,digest(path)))
            if row is None:continue
            r=json.loads(path.read_bytes())
            if row['stage2_reached'] is not False:
                raise ValueError('This plot is defined only for complete Stage 1 Game Over endpoints')
            endpoint=row['latest_observed_stage1_native_tick']
            if t==11:
                native=path.parent/'native.bin'
                if digest(native)!=r['evidence_sha256']['native.bin']:raise ValueError('Changed original TH11 score')
                final=np.fromfile(native,'<i4').reshape(-1,12)[-1]
                if final[5]!=-1 or final[8]!=0:raise ValueError('TH11 score not settled at game over')
                endpoint=int(final[1])*10
            if t==12:
                e=r['covered_native_events']
                raw=path.parent.parent/'raw'/r['episode']/'engineering-native.tsv'
                if digest(raw)!=e['raw_sha256'][raw.name]:raise ValueError('Changed native TH12 evidence')
                with raw.open() as f:
                    fatal=next(x for x in csv.DictReader(f,delimiter='\t') if int(x['spares'])<0 and x['stage']==x['manager_stage']=='1')
                endpoint=int(fatal['tick'])
                if endpoint!=e['death_native_tick']:raise ValueError('Changed TH12 fatal tick')
            if endpoint is None:raise ValueError('Missing original endpoint')
            key=t,row['difficulty'],row['seed']
            pair=buckets[key]
            if row['arm'] in pair:raise ValueError('Duplicate registered endpoint')
            pair[row['arm']]=dict(value=endpoint,host=row['host'],lane=r.get('lane'),receipt=row['native_log_sha256'])
        baseline,candidate=plan['reference_arm'],plan['candidate']
        cells={}
        for t in games:
            for rank in (2,3):
                values=defaultdict(list)
                paired=[]
                for (title,level,seed),arms in sorted(buckets.items()):
                    if (title,level)!=(t,rank):continue
                    a,b=arms[baseline],arms[candidate]
                    if a['host']!=b['host'] or a['lane']!=b['lane']:raise ValueError('Unmatched original hardware')
                    delta=b['value']-a['value'];values[a['host'],a['lane']].append(delta)
                    paired.append(dict(seed=seed,reference=a,candidate=b,difference=delta))
                if len(paired)!=8:raise ValueError('Missing registered pairs')
                rng=np.random.default_rng(261127)
                draws=np.column_stack([rng.choice(v,size=(20000,len(v)),replace=True) for v in values.values()]).mean(1)
                cells[f'{t}-{rank}']=dict(pairs=paired,mean=np.mean([r['difference'] for r in paired]).item(),
                                           percentile95=np.quantile(draws,[.025,.975]).tolist())
        summaries[name]=dict(reference=baseline,candidate=candidate,scored_sha256=digest(file),cells=cells)
    fig,axes=plt.subplots(1,5,figsize=(15.5,4),layout='constrained')
    for ax,t in zip(axes,games):
        ax.axhline(0,color='.45',lw=.9)
        for i,(study,comparison) in enumerate(summaries.items()):
            for pos,rank in enumerate((2,3)):
                r=comparison['cells'][f'{t}-{rank}']
                scale=1e6 if t==11 else 1
                avg=r['mean']/scale;low,high=np.asarray(r['percentile95'])/scale
                ax.errorbar(pos+(i-.5)*.17,avg,yerr=[[avg-low],[high-avg]],fmt='o',capsize=3.5,
                            color=['#326183','#b45227'][i],label=['SFT → PPO 1','PPO 1 → PPO 2'][i] if pos==0 else None)
        ax.set_title('TH'+str(t))
        ax.set_xticks([0,1],['Hard','Lunatic']);ax.set_xlim(-.45,1.45)
        ax.set_ylabel('Score (millions)' if t==11 else 'Native ticks')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Change in original-game progress on eight paired starts per title and difficulty')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=2,frameon=False)
    fig.savefig(out/'five-title-paired-generations.png',dpi=180);plt.close(fig)
    (out/'five-title-paired-generations.json').write_text(json.dumps(dict(studies=summaries,
        methods='Two independently seeded evaluation cohorts. The unit is the paired controller seed within physical host and recorded lane. Percentile intervals resample those pairs within stratum. TH11 uses settled native score; TH12 uses fatal Stage 1 tick; TH10, TH13 and TH17 use last observed native Stage 1 tick. All 320 arm outcomes are verified Game Over in Stage 1. Differences across generations use distinct seed sets. This is descriptive and does not correct for multiple comparisons.'),indent=2)+'\n')


if __name__=='__main__':
    main()
