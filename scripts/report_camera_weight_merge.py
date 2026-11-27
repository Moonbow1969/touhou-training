"""Compare frozen merge controls on identical photos, with one result per held pilot."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.report_shared_title_training import scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root
    labels=['Parent','Joint','Specialist','Joint, scaled','Merged']
    colors=['#777777','#376297','#8f6a91','#d59036','#087a65']
    results={}
    fig,axes=plt.subplots(2,2,figsize=(10,6.2),constrained_layout=True)
    for ax,t in zip(axes.flat,('TH10','TH12','TH13','TH17')):
        def load(folder):
            return json.loads((root/folder/'evaluation'/t/'held.json').read_bytes())
        primary=load('merged-equal')
        controls=load('joint-matched-distance')
        models=[controls['policy'],controls['four-title'],primary[t+'-only'],
                controls['joint-matched-distance'],controls['merged-equal']]
        expected={k:r['sha256'] for k,r in primary['four-title']['replays'].items()}
        results[t]={}
        for i,(label,m) in enumerate(zip(labels,models)):
            if {k:r['sha256'] for k,r in m['replays'].items()}!=expected:
                raise ValueError('Unpaired camera observations')
            groups=defaultdict(list)
            for r in m['replays'].values():groups[r['group']].append(r['aligned'])
            all_rows=[r for group in groups.values() for r in group]
            results[t][label]=dict(checkpoint=m['checkpoint'],
                    pooled={c:scores(all_rows,c) for c in ('all','change','start')},
                    players={k:{c:scores(v,c) for c in ('all','change','start')} for k,v in groups.items()})
            accuracy=results[t][label]['pooled']['all']['accuracy']*100
            ax.scatter(i,accuracy,color=colors[i],s=65,zorder=3)
            ax.annotate(f'{accuracy:.2f}',(i,accuracy),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
        y=[results[t][k]['pooled']['all']['accuracy']*100 for k in labels]
        ax.set(xlim=(-.5,4.5),ylim=(np.floor(min(y))-1,np.ceil(max(y))+1.1),title=t,
               xticks=range(len(labels)),xticklabels=labels,ylabel='Movement accuracy (%)')
        ax.tick_params(axis='x',labelsize=9)
        ax.grid(axis='y',alpha=.25)
    fig.suptitle('Same held replay decisions, five frozen policies')
    fig.savefig(root/'weight-merge-controls.png',dpi=170)
    plt.close(fig)
    (root/'weight-merge-controls.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
