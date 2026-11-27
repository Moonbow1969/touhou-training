"""Report paired player-level multi-title and single-title imitation metrics."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def scores(rows, condition):
    n = sum(row[condition]['n'] for row in rows)
    return dict(n=n, accuracy=sum(row[condition]['correct'] for row in rows)/n,
                nll=sum(row[condition]['movement_nll']*row[condition]['n'] for row in rows)/n)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    root=parser.parse_args().root
    result={}
    fig,axes=plt.subplots(1,2,figsize=(10,5),constrained_layout=True)
    for i,title in enumerate(('TH10','TH12','TH13','TH17')):
        report=json.loads((root/'evaluation'/title/'held.json').read_bytes())
        arms=('four-title',title+'-only')
        by_arm={}
        for arm in arms:
            grouped=defaultdict(list)
            for r in report[arm]['replays'].values():
                grouped[r['group']].append(r['aligned'])
            by_arm[arm]={'players':{p:{c:scores(rows,c) for c in ('all','change','start')}
                                   for p,rows in grouped.items()},
                         'pooled':{c:scores([r for rows in grouped.values() for r in rows],c)
                                   for c in ('all','change','start')}}
        result[title]=by_arm
        for j,cond in enumerate(('all','change')):
            ax=axes[j]
            for p in by_arm[arms[0]]['players']:
                ax.plot([i-.17,i+.17],[100*by_arm[a]['players'][p][cond]['accuracy'] for a in arms],
                        color='#899aab',alpha=.8,marker='o',markersize=4)
            ax.plot([i-.17,i+.17],[100*by_arm[a]['pooled'][cond]['accuracy'] for a in arms],
                    color='#086d64',marker='D',linewidth=2,label='Pooled windows' if i==0 else None)
    for ax,heading in zip(axes,('All decisions','Direction changed')):
        ax.set_xticks(range(4),list(result))
        ax.set_title(heading)
        ax.set_ylabel('Expert movement accuracy (%)')
        ax.grid(axis='y',alpha=.23)
    fig.suptitle('Shared policy (left) and same-title specialist (right)\nGray lines pair individual held players')
    axes[1].legend(frameon=False)
    fig.savefig(root/'shared-vs-specialists.png',dpi=170)
    plt.close(fig)
    (root/'matched-player-results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
