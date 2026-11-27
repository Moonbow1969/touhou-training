"""Plot the paired Stage 1 differences for the discovery and independent replication."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.score_native_stage_clock import summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    data=parser.parse_args().data.resolve()
    folder=data/'th10-lunatic-merged-replication'
    rows=json.loads((data/'four-game-merged-online/qualified-th10/credits.json').read_bytes())
    rows=[r for r in rows if r['difficulty']==3]
    replication=json.loads((folder/'replication.json').read_bytes())
    design=json.loads((folder/'protocol.json').read_bytes())
    a,b=replication['arms']
    if {r['arm']:r['policy_sha256'] for r in rows}!=design['policy_sha256']:
        raise ValueError('Discovery and replication must have identical policies')
    raw=summarize(rows,10,(a,b))
    ticks=raw['per_seed']
    first={s:ticks[b][s]-ticks[a][s] for s in ticks[a]}
    primary=replication['primary']
    second={r['seed']:r['candidate']-r['reference'] for r in primary['pairs']}
    if set(first)&set(second) or set(second)!=set(design['seeds']) or len(first)!=12 or len(second)!=24:
        raise ValueError('Repeated or incomplete cohort')
    fig,ax=plt.subplots(figsize=(8.0,4.1),layout='constrained')
    groups=[(first,raw['mean_paired_change'],raw['paired_fixed_host_bootstrap95']),
            (second,primary['paired_difference'],primary['bootstrap95'])]
    for i,(points,m,ci) in enumerate(groups):
        values=np.array(list(points.values()))
        offsets=np.linspace(-.13,.13,len(values))[np.argsort(np.argsort(values,kind='stable'),kind='stable')]
        ax.scatter(values,i+offsets,s=25,color='#5d8195',alpha=.85)
        ax.errorbar(m,i+.23,xerr=[[m-ci[0]],[ci[1]-m]],fmt='D',ms=6,capsize=5,linewidth=2.5,color='#12634f')
        ax.text(m,i+.32,f'{m:+.0f}  [{ci[0]:+.0f}, {ci[1]:+.0f}]',ha='center',va='center',fontsize=10)
    ax.axvline(0,c='#657179',ls='--',lw=1)
    ax.set_yticks([0,1],['Original cohort\n12 pairs','Fresh replication\n24 pairs'])
    ax.set_ylim(1.51,-.30)
    ax.set_xlabel('Paired difference in final Stage 1 tick (four-game PPO minus SFT)')
    ax.set_title('TH10 Lunatic: the initial survival gain did not replicate',loc='left',fontweight='bold')
    ax.spines[['top','right','left']].set_visible(False)
    ax.grid(axis='x',alpha=.2)
    fig.text(.015,.006,'Each dot is one controller seed. Diamonds show paired means and 95% bootstrap intervals.',fontsize=8)
    fig.savefig(folder/'paired-replication.png',dpi=170,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)
    print([(len(g),round(m,2),ci) for g,m,ci in groups],flush=True)


if __name__=='__main__':
    main()
