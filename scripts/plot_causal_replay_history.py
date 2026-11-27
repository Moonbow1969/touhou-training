"""Plot matched frozen human direction changes for the temporal-camera comparison."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    root=p.parse_args().root
    data=json.loads((root/'report.json').read_bytes())
    titles=list(data['pooled_by_title']);x=np.arange(len(titles))
    fig,axes=plt.subplots(1,2,figsize=(9.5,4),layout='constrained')
    for ax,field in zip(axes,('accuracy','nll')):
        for offset,name,color,marker in ((-.15,'base','#666666','o'),(0,'repeated-current','#ca803a','s'),(.15,'chronological','#267886','D')):
            ys=[data['pooled_by_title'][t][name]['change'][field] for t in titles]
            ax.scatter(x+offset,np.array(ys)*(100 if field=='accuracy' else 1),label=name,
                       color=color,marker=marker,s=48,zorder=3)
        ax.set_xticks(x,titles);ax.grid(axis='y',alpha=.25);ax.spines[['right','top']].set_visible(False)
    axes[0].set_ylabel('Correct direction on human changes (%)')
    axes[1].set_ylabel('Direction-change negative log likelihood')
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Eight consecutive camera observations on held human direction changes')
    fig.savefig(root/'held-changes.png',dpi=165)
    plt.close(fig)


if __name__=='__main__':
    main()
