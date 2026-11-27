"""Plot the five-game critic and initial policy-gradient diagnostics."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    root=p.parse_args().root
    def cells(path):
        return json.loads((root/path).read_bytes())['cells']
    gradients=cells('gradient-by-credit-count.json')
    critic=cells('critic-qualification/report.json')
    names=sorted(critic)
    labels=[k.split('-')[0]+' '+{'2':'H','3':'L'}[k[-1]] for k in names]
    x=np.arange(len(labels))
    fig,(a,b)=plt.subplots(2,1,figsize=(11,7.4),layout='constrained',sharex=True)
    for n,color in zip((2,4,8),('#8c8c8c','#4377ad','#a6462f')):
        a.plot(x,[gradients[k]['comparisons'][str(n)]['median'] for k in names],'o-',label=f'{n} credits per group',color=color)
    a.set_title('Agreement between policy gradients from disjoint original-game credits',loc='left')
    a.set_ylabel('Median cosine similarity')
    a.legend(ncol=3,loc='upper left',fontsize=9)
    for delta,key,label,color in ((-.23,'native_critic','Frozen native critic','#a6462f'),(0,'clock','Elapsed decisions','#3b718c'),(.23,'clock_camera','Elapsed decisions + vision','#51916b')):
        b.bar(x+delta,[critic[k]['transition_r_squared'][key] for k in names],.22,label=label,color=color)
    b.set_title('Discounted terminal return prediction on held training trajectories',loc='left')
    b.set_ylabel('Out-of-fold R²')
    b.legend(ncol=3,loc='lower right',fontsize=9)
    for ax in (a,b):
        ax.axhline(0,color='black',lw=.9)
        ax.grid(axis='y',alpha=.18)
    b.set_xticks(x,labels,rotation=25)
    fig.suptitle('Five original titles · 160 Hard and Lunatic training credits · 78,740 playable decisions',fontsize=14)
    fig.savefig(root/'critic-qualification/critic-and-gradient.png',dpi=155)
    plt.close(fig)

if __name__=='__main__':
    main()
