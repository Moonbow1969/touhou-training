"""Plot the realized and probability-implied movement switching in two independent native studies."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scripts.collect_th10_full_action import save
from scripts.run_th10_camera_pair import switches


def plot(root):
    cases=[('A  ·  Sampling correction','frequency-native10',['original','corrected'],['Human SFT','Frequency SFT'],range(757001,757007)),
           ('B  ·  Learned movement persistence','transition-native10',['frequency','transition'],['Frequency SFT','Learned hold'],range(758001,758007))]
    fig,axes=plt.subplots(1,2,figsize=(8.5,4.6),sharey=True,layout='constrained')
    output={}
    for ax,(title,sub,folders,labels,seeds) in zip(axes,cases):
        stats=[switches(root/sub/name,'accelerator',seeds) for name in folders];output[sub]=dict(zip(folders,stats))
        actual=np.array([[r['rate'] for r in arm['credits']] for arm in stats]).T*100
        expected=np.array([[r['expected_rate'] for r in arm['credits']] for arm in stats]).T*100
        for i,(a,e) in enumerate(zip(actual,expected)):
            shift=(i-2.5)*.014
            ax.plot([shift,1+shift],a,color='#91a4b1',linewidth=1,alpha=.85)
            ax.scatter([shift,1+shift],a,s=22,color='#27678d',zorder=3,label='Original-game actions' if i==0 else None)
            ax.scatter([shift,1+shift],e,s=30,facecolors='none',edgecolors='#a84d2d',linewidths=1,zorder=4,
                       label='Model expectation' if i==0 else None)
        pooled=[s['pooled_rate']*100 for s in stats]
        ax.scatter([0,1],pooled,marker='_',s=450,color='black',zorder=5,linewidths=2.5)
        ax.set_title(title,fontsize=11);ax.set_xticks([0,1],labels);ax.set_xlim(-.20,1.2);ax.grid(axis='y',alpha=.18)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Actions changing movement direction (%)');axes[0].set_ylim(0,85)
    axes[0].legend(fontsize=8,loc='lower left')
    fig.suptitle('Movement persistence in original TH10 Normal',fontsize=14)
    fig.supxlabel('Each line pairs one complete credit by controller seed. Each study uses six fresh seed pairs.',fontsize=9)
    fig.savefig(root/'th10-action-persistence.png',dpi=180);plt.close(fig)
    save(root/'th10-action-persistence-data.json',output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    plot(p.parse_args().root)
