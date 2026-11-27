"""Compare equal-player human imitation metrics at native tick resolution."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study',type=Path)
    root=parser.parse_args().study
    report=json.loads((root/'report.json').read_bytes());rows=report['by_stage']
    titles=sorted(report['movement_nll_difference_by_title']);methods=('four-tick','two-subwindows')
    def by_player(stage_rows,method):
        m=[r['metrics'][method] for r in stage_rows]
        return np.array([sum(x['per_head_mean_nll'][0]*x['ticks'] for x in m)/sum(x['ticks'] for x in m),
            100*sum(x['per_head_correct'][0] for x in m)/sum(x['ticks'] for x in m),
            100*sum(x['moved_within_window_correct'] for x in m)/sum(x['moved_within_window_ticks'] for x in m)])
    values={}
    for t in titles:
        per_title=[r for r in rows if r['title']==t]
        for arm in methods:
            values[t,arm]=np.mean([by_player([r for r in per_title if r['player']==u],arm) for u in sorted({r['player'] for r in per_title})],axis=0)
        if not np.isclose(values[t,methods[1]][0]-values[t,methods[0]][0],report['movement_nll_difference_by_title'][t]):raise ValueError('Changed pre-registered player weights')
    fig,axes=plt.subplots(1,3,figsize=(12.5,3.8),layout='constrained')
    for i,ax in enumerate(axes):
        for t,col in zip(titles,('#2c6e91','#be6a36','#527e4b','#885d91')):
            ax.plot((0,1),[values[t,arm][i] for arm in methods],marker='o',lw=2,color=col,label=t)
        ax.set_xticks([0,1],['One command','Two commands']);ax.set_xlim(-.18,1.18)
        ax.grid(axis='y',alpha=.22);ax.spines[['top','right']].set_visible(False)
        ax.set_title(('Movement NLL per tick','All-direction accuracy (%)','Accuracy after an in-window turn (%)')[i],fontsize=10)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,frameon=False)
    fig.suptitle('Same image · same training budget · previous development players')
    fig.savefig(root/'native-tick-imitation.png',dpi=155)
    plt.close(fig)

if __name__=='__main__':
    main()
