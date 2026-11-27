"""Compare adjacent four-game PPO generations on matched native full-credit seeds."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.score_native_stage_clock import summarize


def read(f):return json.loads(f.read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    d=parser.parse_args().data
    result={}
    for study in ('four-game-online','four-game-ppo-generation2'):
        root=d/study;rows=read(root/'evaluation/credits.json')
        if len(rows)!=192:raise ValueError('Incomplete original-credit cohort')
        registered=read(root/'camera-protocol.json')
        arms=[registered['reference_arm'],registered['candidate']]
        original={r['trajectory_sha256']:r for r in read(root/'native12/held-statistics.json')['episodes']}
        for row in rows:
            if row['title']==12:
                witness=original[row['original_trace_sha256']]
                if (not witness['sensor_verified'] or witness['arm']!=row['arm'] or
                    witness['seed']!=row['seed'] or witness['rank']!=row['difficulty']):
                    raise ValueError('TH12 clock source mismatch')
                row['latest_observed_stage1_native_tick']=witness['stage1_last_native_tick']
        result[study]={}
        for title in (10,12,13,17):
            cell=result[study][str(title)]={}
            for rank in (2,3):
                included=[r for r in rows if (r['title'],r['difficulty'])==(title,rank)]
                if len(included)!=24:raise ValueError('Expected twelve original pairs')
                if all(r['latest_observed_stage1_native_tick'] is None for r in included):
                    cell[str(rank)]=None
                else:cell[str(rank)]=summarize(included,title,arms)
    dest=d/'learning-curves';dest.mkdir(exist_ok=True)
    (dest/'four-game-native-ppo-changes.json').write_text(json.dumps(result,indent=2)+'\n')
    fig,axes=plt.subplots(2,2,figsize=(9.5,6.2),layout='constrained',sharex=True)
    for ax,t in zip(axes.flat,('10','12','13','17')):
        ax.axvline(0,color='#999999',linewidth=.9)
        for i,(study,by_title) in enumerate(result.items()):
            for j,rank in enumerate(('2','3')):
                point=by_title[t][rank];y=3-2*i-j
                if point is None:
                    ax.text(0,y,'  Clock unavailable',fontsize=9,color='#687781',va='center')
                    continue
                value=point['mean_paired_change'];lo,hi=point['paired_fixed_host_bootstrap95']
                ax.plot([lo,hi],[y,y],linewidth=2.2,color=('#16796d','#8b4f8e')[j])
                ax.plot(value,y,'o',color=('#16796d','#8b4f8e')[j],markersize=6)
        ax.set(yticks=[3,2,1,0],yticklabels=['SFT → PPO1 · Hard','SFT → PPO1 · Lunatic',
                'PPO1 → PPO2 · Hard','PPO1 → PPO2 · Lunatic'],ylim=(-.6,3.6),title='TH'+t)
        ax.grid(axis='x',alpha=.18)
    for ax in axes[-1]:ax.set_xlabel('Change in last observed Stage 1 native tick')
    fig.suptitle('Paired original-game progress across two PPO updates\n12 seeds per game, rank and update · 95% paired bootstrap within host')
    fig.savefig(dest/'four-game-native-ppo-changes.png',dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
